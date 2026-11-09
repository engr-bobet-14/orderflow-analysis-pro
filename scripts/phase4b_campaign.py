"""Phase 4b sequential host driver. All experiment entry points require sealed host approvals."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.phase4_policy import PhaseLayout, canonical_bytes, source_state
from scripts.phase4_approvals import require_phase4b_approval, verify_phase4b_gates
from orderflow_system.data.paired_dataset import sha256_file
from scripts import optimize_orderflow as opt, phase4_studies as p4

BAR_TYPES=('fixed_tick_imbalance','fixed_volume_imbalance')


def stage_ceiling(manifest,stage):
    value=manifest.get('max_stage_seconds',{}).get(stage)
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
        raise ValueError('finite positive registered stage ceiling required')
    return value


def stage_progress(layout,digest,stage,*,arm=None,trial=None,marker):
    return write_json(layout,layout.checkpoints/digest/f'{stage}.progress.json',{
        'stage':stage,'arm':arm,'trial':trial,'marker':marker,'utc':datetime.now(timezone.utc).isoformat()})


LAUNCH_ENV='PHASE4B_LAUNCH'
LOCK_ENV='PHASE4B_LOCK_FDS'
# Rev 13 C7 termination kinds. Consuming kinds end the stage under the sealed manifest.
# A3 (owner decision 2026-11-08): S1 only. Its RSS includes both production trade caches' file-backed
# residency while two arms are open (measured 4.0 GiB mapped, 0.46 GiB private); private stays 4 GiB.
S1_RSS_LIMIT_BYTES=12*1024**3


def stage_rss_limit(stage):
    from scripts.phase4_resources import PRIVATE_BUDGET
    return S1_RSS_LIMIT_BYTES if stage=='S1' else PRIVATE_BUDGET


def stage_min_available(stage):
    """Host memory required at admission: the stage RSS limit above 4 GiB plus the unchanged 8 GiB margin."""
    from scripts.phase4_resources import MIN_AVAILABLE,PRIVATE_BUDGET
    limit=stage_rss_limit(stage)
    return MIN_AVAILABLE+limit if limit>PRIVATE_BUDGET else MIN_AVAILABLE


def enforce_stage_memory(measurement,stage):
    from scripts.phase4_resources import enforce_memory
    return enforce_memory(measurement,0,stage_rss_limit(stage))


CONSUMING_KINDS=('timeout_confirmed','study_allowance_exhausted','integrity_stop','strict_memory_stop')
UNKNOWN='external_or_unknown_termination'
KILL_AFTER_SECONDS=30
# A sleep-inclusive clock, like the external timeout's: CLOCK_MONOTONIC continues across sleep on darwin,
# CLOCK_BOOTTIME does on Linux (time.monotonic on darwin is mach_absolute_time and pauses during sleep).
_CLOCK=getattr(time,'CLOCK_BOOTTIME',time.CLOCK_MONOTONIC)


def _now():
    return time.clock_gettime(_CLOCK)


def new_launch():
    import uuid
    return {'utc':datetime.now(timezone.utc).isoformat(),'monotonic':_now(),'attempt':uuid.uuid4().hex}


def launch_from_env():
    raw=os.environ.get(LAUNCH_ENV)
    if raw is None:return None
    launch=json.loads(raw)
    if set(launch)!={'utc','monotonic','attempt'}:raise ValueError('invalid stage launch context')
    return launch


def _termination_dir(layout,digest,stage):
    return layout.checkpoints/digest/'terminations'/stage


def _since_launch(launch):
    return None if launch is None or launch.get('monotonic') is None else _now()-launch['monotonic']


def classify(stop=None,returncode=None,elapsed=None,ceiling=None):
    """Only observed evidence at or above its own ceiling confirms a timeout; the cause is never guessed.

    GNU timeout --foreground (rev 13 C7 b): SIGTERM at the deadline gives exit 124; its kill-after SIGKILL,
    KILL_AFTER_SECONDS later, makes timeout itself exit 137. -SIGKILL is never timeout evidence: without
    --foreground timeout SIGKILLs its own process group (itself included), which any outside SIGKILL mimics.
    """
    if stop is not None:
        if stop.get('reason')=='study_allowance_exhausted':return 'study_allowance_exhausted'
        if stop.get('reason')=='integrity_stop':return 'integrity_stop'
        if stop.get('reason')=='memory':return 'strict_memory_stop'  # first-party watchdog record only
        observed=stop.get('charge_basis',('upper_bound' if stop.get('reconciled') else 'observed'))=='observed'
        if (observed and not stop.get('reconciled') and stop.get('reason','').startswith('max_')
                and stop.get('measured_seconds') is not None and stop.get('limit_seconds') is not None
                and stop['measured_seconds']>=stop['limit_seconds']):
            return 'timeout_confirmed'
        return UNKNOWN
    if elapsed is None or ceiling is None:return UNKNOWN
    if returncode==124 and elapsed>=ceiling:return 'timeout_confirmed'
    if returncode==137 and elapsed>=ceiling+KILL_AFTER_SECONDS:return 'timeout_confirmed'
    return UNKNOWN


_UNSET=object()


def record_termination(layout,manifest,digest,stage,*,stop=None,returncode=None,memory=None,signal_number=None,
                       elapsed=_UNSET,attempt=None):
    """Write-ahead, per launch attempt; the first record of an attempt wins (concurrent watchdogs wait).

    ``elapsed`` is the stage elapsed from L observed by the caller (the driver measures it at child exit).
    """
    launch=manifest.get('_launch');attempt=attempt or (launch or {}).get('attempt','unlaunched')
    directory=layout.checkpoints/digest
    progress_path=directory/f'{stage}.progress.json'
    progress=json.loads(progress_path.read_bytes()) if progress_path.exists() else {}
    arm=(f"BTCUSDT:{BAR_TYPES[stage=='S2v']}:full" if stage in ('S2t','S2v') else None)
    elapsed=_since_launch(launch) if elapsed is _UNSET else elapsed
    ceiling=None
    if stop is None:
        try:ceiling=stage_ceiling(manifest,stage)
        except (ValueError,KeyError):ceiling=None
    kind=classify(stop,returncode,elapsed,ceiling)
    stop=stop or {}
    if returncode is not None and returncode<0:signal_number=-returncode
    duration=stop.get('measured_seconds',elapsed if returncode is not None else None)
    boundaries=stop.get('boundaries') or ({'prior_charge_seconds':0.,'current_seconds':stop['measured_seconds']}
        if stop.get('reason') in ('max_trial_seconds','max_stage_seconds') and 'measured_seconds' in stop else
        {'launch_monotonic':(launch or {}).get('monotonic'),'current_seconds':elapsed})
    if (stop.get('reason')=='max_trial_seconds' and not stop.get('boundaries') and stop.get('live')
            and not stop.get('reconciled') and stop.get('measured_seconds') is not None and stop.get('utc')):
        # Rev 13 C7 audit: the trial clock's own start, from the stop's own detection instant (valid when a
        # persisted stop is replayed under a later launch). Upper-bound (reconciled) charges get none.
        started=datetime.fromisoformat(stop['utc'])-timedelta(seconds=stop['measured_seconds'])
        boundaries={**boundaries,'trial_started_utc':started.isoformat(),
                    'trial_start_basis':'stop utc - measured_seconds (perf_counter)'}
    record=opt.json_safe({'kind':kind,'campaign_sha256':digest,'stage':stage,'attempt':attempt,'L':launch,
        'arm':progress.get('arm') or arm,'trial':stop.get('trial',progress.get('trial')),
        'reason':stop.get('reason') or ('process_exit' if returncode is not None else signal_name(signal_number)),
        'elapsed_seconds':elapsed,'duration_seconds':duration,'boundaries':boundaries,
        'ceiling_seconds':stop.get('limit_seconds',ceiling),
        'ceiling_inputs':manifest.get('runtime_ceilings',{}).get('inputs') if kind=='timeout_confirmed' else None,
        'charge_basis':stop.get('charge_basis',('upper_bound' if stop.get('reconciled') else 'observed')),
        'observed_exit_code':returncode,'observed_signal':signal_name(signal_number),
        'last_progress_marker':progress.get('marker','stage launch'),'last_memory_sample':memory,
        'cause':'ceiling' if kind=='timeout_confirmed' else stop.get('cause','unknown' if kind==UNKNOWN else kind),
        'utc':datetime.now(timezone.utc).isoformat()})
    envelope={'record':record,'sha256':hashlib.sha256(canonical_bytes(record)).hexdigest()}
    folder=_termination_dir(layout,digest,stage)
    with opt.file_lock(layout.checked_write(folder/'termination.lock')):
        try:write_json(layout,folder/f'{attempt}.pending.json',envelope,exclusive=True)
        except FileExistsError:pass  # A concurrent watchdog won; read its complete marker below.
    return _repair(layout,digest,stage,attempt)


def durable_audit(layout,manifest,digest,stage,attempts=3,**kwargs):
    """Watchdog audits retry transient write failures before the process exits."""
    for index in range(attempts):
        try:return record_termination(layout,manifest,digest,stage,**kwargs)
        except Exception:
            if index+1==attempts:raise
            time.sleep(.05)


def signal_name(number):
    if number is None:return None
    try:return signal.Signals(number).name
    except ValueError:return str(number)


def _repair(layout,digest,stage,attempt):
    from scripts.phase4_campaign import append_log,atomic_bytes
    folder=_termination_dir(layout,digest,stage)
    pending=layout.checked_write(folder/f'{attempt}.pending.json')
    with opt.file_lock(layout.checked_write(folder/'termination.lock')):
        envelope=json.loads(pending.read_bytes());record=envelope['record']
        if (hashlib.sha256(canonical_bytes(record)).hexdigest()!=envelope['sha256']
                or record['campaign_sha256']!=digest or record['stage']!=stage or record['attempt']!=attempt):
            raise ValueError('termination write-ahead record hash/identity mismatch')
        final=layout.checked_write(folder/f'{attempt}.json')
        if final.exists():
            if json.loads(final.read_bytes())!=record:raise ValueError('termination audit changed')
        else:write_json(layout,final,record,exclusive=True)
        log=layout.checked_write(layout.output/'campaign-log.jsonl')
        lines=log.read_bytes().splitlines(keepends=True) if log.exists() else []
        events=[]
        for index,line in enumerate(lines):
            try:events.append(json.loads(line))
            except ValueError:
                if index!=len(lines)-1:raise ValueError('corrupt campaign log before termination recovery')
                atomic_bytes(log,b''.join(lines[:index]),layout=layout)  # Repair only a torn final append.
        if not any(e.get('event')=='termination_audit' and e.get('audit_sha256')==envelope['sha256'] for e in events):
            append_log(log,{'event':'termination_audit','audit_sha256':envelope['sha256'],**record})
        return record


def terminations(layout,digest,stage):
    """Every recorded termination of a stage, repaired, oldest first. A final without its marker is refused."""
    folder=layout.checkpoints/digest/'terminations'/stage
    if not folder.exists():return []
    pending={p.name[:-len('.pending.json')] for p in folder.glob('*.pending.json')}
    finals={p.name[:-len('.json')] for p in folder.glob('*.json') if not p.name.endswith('.pending.json')}
    if finals-pending:raise ValueError('termination pending marker missing for '+', '.join(sorted(finals-pending)))
    records=[_repair(layout,digest,stage,attempt) for attempt in pending]
    return sorted(records,key=lambda r:(r['utc'],r['attempt']))


def recover_termination_audit(layout,digest,stage):
    records=terminations(layout,digest,stage)
    return records[-1] if records else None


def _classify_orphans(layout,manifest,digest,stage,records):
    """A started attempt with no checkpoint and no record ended unclassified (e.g. the driver died)."""
    attempts=layout.checkpoints/digest/'attempts'
    if not attempts.exists() or (layout.checkpoints/digest/f'{stage}.json').exists():return records
    known={r['attempt'] for r in records}|{r['attempt'][:-len('-postflight')] for r in records
                                             if r['attempt'].endswith('-postflight')}
    for path in sorted(attempts.glob('*.json')):
        started=json.loads(path.read_bytes())
        attempt=started.get('attempt')
        if started.get('stage')!=stage or started.get('campaign_sha256',digest)!=digest or not attempt or attempt in known:
            continue
        record_termination(layout,{**manifest,'_launch':{'utc':started.get('utc'),'monotonic':None,'attempt':attempt}},
            digest,stage,stop={'reason':'unclassified_end','cause':'unknown'},elapsed=None)
        known.add(attempt)
    return terminations(layout,digest,stage)


def termination_admission(layout,manifest,digest,stage,recovery_selector,argv,*,consume=True):
    """None for a fresh stage; the consumed recovery approval for an owner-approved recovery."""
    from scripts.phase4_approvals import require_recovery_approval,consumed_recoveries
    records=_classify_orphans(layout,manifest,digest,stage,terminations(layout,digest,stage))
    consuming=[r for r in records if r['kind'] in CONSUMING_KINDS]
    if consuming:
        raise ValueError(f"stage consumed by {consuming[0]['kind']}; new registration required")
    done=consumed_recoveries(layout)
    open_unknown=[r for r in records if r['kind']==UNKNOWN
                  and hashlib.sha256(canonical_bytes(r)).hexdigest() not in done]
    if not open_unknown:
        if recovery_selector is not None:raise ValueError('no unrecovered termination for this recovery approval')
        return None
    if len(open_unknown)>1:raise ValueError('several unrecovered terminations; owner recovery decision required')
    if recovery_selector is None:
        raise ValueError('external/unknown termination: explicit owner recovery approval required')
    return require_recovery_approval(layout,recovery_selector,digest,stage,open_unknown[0],argv,consume=consume)


def recovery_admission(layout,manifest,digest,stage,selector,argv):
    """Host proof gates are re-verified before an owner recovery approval is consumed."""
    try:gates,basis=verify_phase4b_gates(layout,digest,stage,with_basis=True)
    except (ValueError,OSError) as exc:raise ValueError('recovery requires the host proof gates: '+str(exc)) from exc
    approval=termination_admission(layout,manifest,digest,stage,selector,argv)
    if approval is None:raise ValueError('no unrecovered termination for this recovery approval')
    return {**approval,'gates_sha256':gates,'inspection_basis':basis}


@contextmanager
def stage_watchdog(layout,manifest,digest,stage):
    """Stage deadline is L + max_stage_seconds, so it fires no later than the external timeout.

    The SIGTERM handler only records the signal; the watchdog thread writes the audit and exits, so a
    signal landing while the main thread holds the termination lock cannot deadlock the process.
    """
    import threading
    launch=manifest.get('_launch') or new_launch()
    manifest['_launch']=launch
    ceiling=stage_ceiling(manifest,stage);done=threading.Event();signalled=threading.Event();received={}
    def sample():
        try:
            from scripts.phase4_resources import current_memory
            value=current_memory();return {'rss_bytes':value.get('rss_bytes'),'private_bytes':value.get('private_bytes')}
        except Exception:return None
    def timeout_stop(elapsed):
        return {'reason':'max_stage_seconds','measured_seconds':elapsed,'limit_seconds':ceiling,'live':True,
                'charge_basis':'observed','boundaries':{'launch_monotonic':launch['monotonic'],
                'prior_charge_seconds':0.,'current_seconds':elapsed}}
    def fire(stop,signal_number=None):
        try:durable_audit(layout,manifest,digest,stage,stop=stop,memory=sample(),signal_number=signal_number)
        finally:os._exit(70)
    def watch():
        cadence=min(.25,max(ceiling/20,.005))
        while True:
            triggered=signalled.wait(cadence)
            if triggered:
                signum,elapsed=received['signum'],received['elapsed']
                fire(timeout_stop(elapsed) if elapsed>=ceiling else {'reason':signal_name(signum),'cause':'unknown'},
                     signum)
            if done.is_set():return
            elapsed=_now()-launch['monotonic']
            if elapsed>=ceiling:fire(timeout_stop(elapsed))
    def terminate(signum,frame):
        received.update(signum=signum,elapsed=_now()-launch['monotonic']);signalled.set()
    previous=signal.signal(signal.SIGTERM,terminate)
    guard=threading.Thread(target=watch,daemon=True);guard.start()
    try:yield launch
    finally:
        done.set();guard.join();signal.signal(signal.SIGTERM,previous)


def canonical_origin_lock_path(origin):
    from scripts.phase4_approvals import PHASE4_CAMPAIGN_SHA256
    root=Path(origin['root'])
    if not root.is_absolute() or origin.get('sha256')!=PHASE4_CAMPAIGN_SHA256:
        raise ValueError('canonical Phase 4 origin required for the execution lock')
    return root/'output/phase4/campaign-execution.lock'


@contextmanager
def canonical_execution_lock(origin):
    """Hold Phase 4's own execution lock: read-only open, never created, written or chmod-ed."""
    import fcntl
    path=canonical_origin_lock_path(origin)
    try:fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    except OSError as exc:raise ValueError('canonical Phase 4 execution lock missing or symlinked') from exc
    try:
        import stat as st
        if not st.S_ISREG(os.fstat(fd).st_mode):raise ValueError('canonical Phase 4 execution lock is not a regular file')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('canonical Phase 4 execution lock is held') from exc
        yield fd
    finally:os.close(fd)


@contextmanager
def _held_lock(path):
    import fcntl
    fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('another phase4b stage/driver is active') from exc
        yield fd
    finally:os.close(fd)


@contextmanager
def stage_locks(layout,manifest):
    """Fixed order: canonical Phase 4 lock, 4b execution lock, 4b driver lock. Held through post-flight."""
    layout.checkpoints.mkdir(parents=True,exist_ok=True)
    with canonical_execution_lock(manifest['phase4_origin']) as phase4:
        with _held_lock(layout.checked_write(layout.checkpoints/'execution.lock')) as execution:
            with _held_lock(layout.checked_write(layout.checkpoints/'driver.lock')) as driver:
                yield {'phase4':phase4,'execution':execution,'driver':driver}


def require_inherited_locks(layout,manifest):
    """Prove each named fd is the driver's held lock: a fresh open cannot lock it, the inherited fd can."""
    import fcntl
    raw=os.environ.get(LOCK_ENV)
    if raw is None:raise ValueError('stage child requires the driver-held locks')
    fds=json.loads(raw)
    paths={'phase4':canonical_origin_lock_path(manifest['phase4_origin']),
           'execution':layout.checkpoints/'execution.lock','driver':layout.checkpoints/'driver.lock'}
    if set(fds)!=set(paths):raise ValueError('stage child lock set incomplete')
    for name,path in paths.items():
        info=os.fstat(fds[name]);target=os.stat(path,follow_symlinks=False)
        if (info.st_dev,info.st_ino)!=(target.st_dev,target.st_ino):raise ValueError('inherited lock mismatch: '+name)
        probe=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            try:
                fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass  # Held, as required.
            else:
                fcntl.flock(probe,fcntl.LOCK_UN)
                raise ValueError(f'{name} lock not held: child did not inherit the driver lock')
        finally:os.close(probe)
        try:fcntl.flock(fds[name],fcntl.LOCK_EX|fcntl.LOCK_NB)  # Same open description as the holder.
        except BlockingIOError as exc:raise ValueError(f'{name} lock fd was not inherited from its holder') from exc


def read_manifest(layout):
    raw=layout.manifest.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if (Path(str(layout.manifest)+'.sha256').read_text().strip()!=digest
            or json.loads(raw).get('phase')!=layout.phase):
        raise ValueError('campaign phase/seal mismatch')
    manifest=json.loads(raw);p4.sequential_campaign(manifest)
    if manifest['source_state']!=source_state():raise ValueError('campaign source_state changed')
    if layout.phase=='phase4b':
        from scripts.phase4b_runtime import require_runtime_fingerprint
        require_runtime_fingerprint(manifest.get('runtime_fingerprint'))
    return manifest,digest


def write_json(layout,path,value,*,exclusive=False):
    value=opt.json_safe(value)
    path=layout.checked_write(path);path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        # Atomic create: a crash or failed fsync never leaves a torn record at ``path``.
        import uuid
        temp=path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
        try:
            with temp.open('xb') as handle:
                handle.write(canonical_bytes(value));handle.flush()
                os.fsync(handle.fileno())
            os.link(temp,path)  # FileExistsError if another writer won.
        finally:
            try:temp.unlink()
            except FileNotFoundError:pass
    else:
        from scripts.phase4_campaign import atomic_bytes
        atomic_bytes(path,canonical_bytes(value),layout=layout)
    return path


# Phase 4's sealed owner baseline (46 files); its listed absolute paths are the only ones verified in place.
PHASE4_PROTECTED_BASELINE_SHA256='f9412c6c104852c4c175b90b589618bbe2ba77d8c3705ab55eefef902a0d590a'
PHASE4_PROTECTED_COUNT=46
MARKET_DATA_ROOT=Path('/Volumes/Market-Data')


def protected_roots(root):
    """Market-Data, the main orderflow checkout (even from a worktree) and the sibling bars repository."""
    import subprocess
    common=subprocess.run(['git','-C',str(root),'rev-parse','--path-format=absolute','--git-common-dir'],
        check=True,capture_output=True,text=True).stdout.strip()
    main=Path(common).parent
    return MARKET_DATA_ROOT,main,main.parent/'information-driven-bars'


def phase4_protected_files(baseline,roots,*,sealed_sha256,count):
    """Verify each sealed-baseline entry at its exact original path; never rebase, follow symlinks or widen."""
    import stat
    data=Path(baseline).read_bytes()
    if hashlib.sha256(data).hexdigest()!=sealed_sha256:
        raise ValueError('protected baseline is not the sealed Phase 4 baseline')
    entries={}
    for line in data.decode().splitlines():
        if not line.strip():continue
        digest,name=line.split(maxsplit=1);name=name.strip()
        if name in entries:raise ValueError('duplicate protected path: '+name)
        entries[name]=digest
    if len(entries)!=count:raise ValueError(f'protected baseline count {len(entries)} != {count}')
    verified={}
    for name,digest in entries.items():
        path=Path(name)
        if not path.is_absolute() or os.path.normpath(name)!=name:
            raise ValueError('protected path is not an exact absolute path: '+name)
        if not any(path!=root and path.is_relative_to(root) for root in roots):
            raise ValueError('protected path outside authorized roots: '+name)
        # ponytail: check-then-hash, not O_NOFOLLOW per component; the baseline is sha-pinned and read-only.
        if os.path.realpath(name)!=name:raise ValueError('protected path traverses a symlink: '+name)
        if not stat.S_ISREG(os.lstat(name).st_mode):raise ValueError('protected path is not a regular file: '+name)
        if sha256_file(path)!=digest:raise ValueError('protected file changed: '+name)
        verified[name]=digest
    return verified


def protected_hashes(layout,manifest):
    from scripts.phase4_campaign import protected_snapshot
    from scripts.trade_feature_cache import BUILDER_FILES
    from scripts.phase4_benchmark import DATASET_ID
    protected=phase4_protected_files(layout.root/'output/phase4/build/protected-before.sha256',
        protected_roots(layout.root),sealed_sha256=PHASE4_PROTECTED_BASELINE_SHA256,count=PHASE4_PROTECTED_COUNT)
    historical={item['path']:item['sha256'] for item in manifest['snapshots']}
    for path,digest in historical.items():
        if sha256_file(Path(path))!=digest:raise ValueError('registered historical snapshot changed')
    return {**protected,**historical,
        **protected_snapshot([layout.root/'output/phase4',*(layout.root/p for p in BUILDER_FILES),
            layout.root/'output/optimize/trade_cache'/DATASET_ID/'COMPLETE.json'])}


def preflight(layout,manifest):
    from scripts.phase4_policy import assert_no_conflicts, mount_ready
    from scripts.phase4_resources import require_memory_plan,enforce_current
    if not mount_ready():raise ValueError('Market-Data mount missing')
    assert_no_conflicts();require_memory_plan([{'mapped_files':{}}]);enforce_current()
    protected_hashes(layout,manifest)


@contextmanager
def file_timing(layout,digest,label,ceiling,*,audit=None,rss_limit=None):
    """Reuse the existing heartbeat, private/RSS bounds and live ceiling without a study DB."""
    import time
    class FileAttributes:
        number=0
        def __init__(self,path):
            self.path=layout.checked_write(path)
            self.user_attrs=json.loads(self.path.read_bytes()) if self.path.exists() else {}
            if self.user_attrs.get('phase4_runtime_stop') is not None:
                raise ValueError('durable runtime stop requires an owner decision/new registration')
        def set_user_attr(self,key,value):
            self.user_attrs[key]=value
            write_json(layout,self.path,self.user_attrs)
    directory=layout.checkpoints/digest/'runtime'/label
    trial=FileAttributes(directory/'trial.json');study=FileAttributes(directory/'study.json')
    with opt.phase4_trial_timing(trial,study,0.,time.perf_counter(),args=SimpleNamespace(
            max_trial_seconds=ceiling,phase4_timeout_audit=audit,phase4_rss_limit_bytes=rss_limit)):
        yield


def run_stage(layout,stage,owner_approved,*,operation=None,argv=None,finalize=True,recovery=None):
    """Write-ahead single-use stage admission; callbacks let tests use only synthetic data.

    The timed region ends at the pending result. Post-flight integrity verification (``postflight_stage``)
    runs inline when ``finalize`` and otherwise in the driver's separate, untimed child (rev 13, R12-01).
    """
    manifest,digest=read_manifest(layout)
    if stage not in manifest['stages']:raise ValueError('stage not registered in this manifest')
    launch=launch_from_env()
    if operation is None and launch is None:
        raise ValueError('production stages run only via the driver (launch context and inherited locks required)')
    if launch is not None:manifest['_launch']=launch
    checkpoint=layout.checked_write(layout.checkpoints/digest/f'{stage}.json')
    lock=layout.checked_write(layout.checkpoints/'execution.lock')
    locks=(nullcontext() if operation is None else
           opt.file_lock(lock,nonblocking=True,message='another phase4b stage is active'))
    with locks:
        if operation is None:require_inherited_locks(layout,manifest)
        if checkpoint.exists():raise ValueError('stage already completed')
        attempts=layout.checked_write(checkpoint.parent/'attempts')
        recovered=None
        if recovery is not None:
            # The driver consumed the owner's recovery approval for exactly this launch attempt.
            recovered=_recovery_marker(layout,digest,stage,recovery,launch)
        else:
            termination_admission(layout,manifest,digest,stage,None,argv)
        marker=attempts/f'{owner_approved}.json' if recovered is None else attempts/f"recovery-{recovered['recovery_sha256']}.json"
        if marker.exists():raise ValueError('stage approval already used')
        verified=(require_phase4b_approval(layout,owner_approved,digest,stage,argv,consume=False) if recovered is None
                  else {'sha256':recovered['recovery_sha256'],'gates_sha256':recovered.get('gates_sha256'),
                        'inspection_basis':recovered.get('inspection_basis')})
        preflight(layout,manifest)
        # The production preflight is strict; synthetic tests replace it, never the writer policy.
        before=protected_hashes(layout,manifest) if operation is None else {
            s['path']:sha256_file(Path(s['path'])) for s in manifest['snapshots']}
        if stage in ('S2t','S2v'):
            benchmark=layout.output/'s1/benchmark.json'
            if sha256_file(benchmark)!=manifest['benchmark_sha256']:raise ValueError('S1 benchmark changed')
        if stage=='S3':
            completed_s2(layout,digest)
        if recovered is None:require_phase4b_approval(layout,owner_approved,digest,stage,argv,consume=True)
        record={'stage':stage,'phase':layout.phase,'campaign_sha256':digest,'owner_approval_sha256':verified['sha256'],
            'gates_sha256':verified['gates_sha256'],'inspection_basis':verified.get('inspection_basis'),
            'source_state':source_state(),'status':'started',
            'attempt':(launch or {}).get('attempt','unlaunched'),'recovery':recovered,
            'utc':datetime.now(timezone.utc).isoformat()}
        from scripts.phase4b_runtime import runtime_fingerprint,require_runtime_fingerprint
        runtime=runtime_fingerprint() if stage=='S1' else manifest.get('runtime_fingerprint')
        if stage!='S1':require_runtime_fingerprint(runtime)
        record['runtime_fingerprint']=runtime
        write_json(layout,marker,record,exclusive=True)
        log=layout.checked_write(layout.output/'campaign-log.jsonl')
        from scripts.phase4_campaign import append_log
        append_log(log,record)
        try:
            manifest={**manifest,'_approval_sha256':verified['sha256'],'_gates_sha256':verified['gates_sha256']}
            stage_progress(layout,digest,stage,marker='stage start')
            with stage_watchdog(layout,manifest,digest,stage) if operation is None else nullcontext():
                with (file_timing(layout,digest,stage,None,rss_limit=stage_rss_limit(stage))
                      if operation is None else nullcontext()):
                    result=(operation or execute_stage)(stage,manifest,digest,layout)
            result=opt.json_safe(result)
            record.update(status='pending_postflight',result=result,protected_before=before,
                protected_mode='full' if operation is None else 'snapshots')
            write_json(layout,_pending_result(layout,digest,stage),record,exclusive=True)
        except BaseException as exc:
            record.update(status='stopped',error=str(exc))
            write_json(layout,checkpoint.with_suffix('.stopped.json'),record)
            append_log(log,record)
            raise
    if not finalize:return record
    return postflight_stage(layout,stage,synthetic=operation is not None)


def _pending_result(layout,digest,stage):
    return layout.checked_write(layout.checkpoints/digest/f'{stage}.result.pending.json')


def _recovery_marker(layout,digest,stage,recovery_sha,launch):
    from scripts.phase4_approvals import recovery_root,trusted_path
    path=recovery_root(layout)/f'{recovery_sha}.admitted.json';trusted_path(path)
    marker=json.loads(path.read_bytes())
    if (marker['campaign_sha256']!=digest or marker['stage']!=stage or launch is None
            or marker['attempt']!=launch['attempt']):
        raise ValueError('recovery admission does not match this launch attempt')
    return marker


def postflight_stage(layout,stage,*,synthetic=False):
    """Untimed integrity verification; only this writes a complete checkpoint (rev 13, R12-01)."""
    from scripts.phase4_campaign import append_log
    manifest,digest=read_manifest(layout)
    pending=_pending_result(layout,digest,stage)
    if not pending.exists():raise ValueError('no pending stage result for post-flight')
    record=json.loads(pending.read_bytes())
    checkpoint=layout.checked_write(layout.checkpoints/digest/f'{stage}.json')
    if checkpoint.exists():raise ValueError('stage already completed')
    log=layout.checked_write(layout.output/'campaign-log.jsonl')
    from scripts.phase4b_runtime import require_runtime_fingerprint
    try:
        if source_state()!=record['source_state']:raise ValueError('integrity: source changed during stage')
        require_runtime_fingerprint(record['runtime_fingerprint'])
        before=record['protected_before']
        # Production post-flight always verifies the full protected set; the pending file cannot opt out.
        snapshots=synthetic and record.get('protected_mode')=='snapshots'
        current=({path:sha256_file(Path(path)) for path in before} if snapshots
                 else protected_hashes(layout,manifest))
        if current!=before:raise ValueError('integrity: protected inputs changed during stage')
        if record.get('recovery') is not None:
            _recovery_marker(layout,digest,stage,record['recovery']['recovery_sha256'],{'attempt':record['attempt']})
        else:
            # The stage approval record must be unchanged after the run (it is already consumed).
            from scripts.phase4_approvals import trusted_path
            approval=layout.approval_root/(record['owner_approval_sha256']+'.json')
            if not snapshots:
                trusted_path(approval)
                if sha256_file(approval)!=record['owner_approval_sha256']:raise ValueError('integrity: approval changed')
    except (ValueError,OSError) as exc:
        # The current launch's attempt id: a post-flight-only recovery must not collide with the original.
        launch=launch_from_env() or {'utc':record.get('utc'),'monotonic':None,'attempt':record['attempt']}
        durable_audit(layout,{**manifest,'_launch':launch},digest,stage,attempt=launch['attempt']+'-postflight',
            stop={'reason':'integrity_stop','cause':str(exc)})
        record.update(status='stopped',error=str(exc))
        write_json(layout,checkpoint.with_suffix('.stopped.json'),record);append_log(log,record)
        raise ValueError('post-flight integrity stop: '+str(exc)) from exc
    result=record['result']
    if stage=='S1':
        result={**result,'campaign_sha256':digest,'runtime_fingerprint':record['runtime_fingerprint']}
        benchmark=write_json(layout,layout.output/'s1/benchmark.json',result,exclusive=True)
        seal=layout.checked_write(str(benchmark)+'.sha256')
        with seal.open('xb') as handle:handle.write((sha256_file(benchmark)+'\n').encode())
    record.update(status='complete',result=result,protected_after=before)
    append_log(log,record);write_json(layout,checkpoint,record,exclusive=True)
    return record


def open_prefix(layout,manifest,bt,stage=None):
    """Historical caches are opened read-only for validation; strategy gets SearchPrefix only."""
    from scripts import diagnose_trades as diagnose
    from scripts.trade_feature_cache import production_generation,generation_directory
    from scripts.phase4_benchmark import DATASET_ID
    from scripts import phase4_resources as resources
    resources.enforce_current(0,stage_rss_limit(stage))
    arm=manifest['arms'][f'BTCUSDT:{bt}:full'];spec=arm['comparator']
    raw=Path(spec['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('comparator hash mismatch')
    artifact=json.loads(raw)
    source_db=layout.root/'output/optimize/phase4/studies.db'
    identity=opt.read_study_attrs(source_db,artifact['study'])['identity']
    if identity['trade_signals']!='disabled':raise ValueError('Stage B comparator must be bar-only')
    cache,meta=diagnose.find_cached_bars(layout.root/'output/optimize/phase4/cache',identity['data_fp'])
    if meta['symbol']!='BTCUSDT' or meta['bar_type']!=bt:raise ValueError('bar cache comparator identity mismatch')
    from scripts.dataset_policy import ensure_dev_window
    ensure_dev_window(date.fromisoformat(meta['start']),date.fromisoformat(meta['end']))
    generation=production_generation(generation_directory(layout.root/'output/optimize/trade_cache',DATASET_ID))
    trade_path=Path(generation['lanes'][bt])
    resources.require_memory_plan([{'mapped_files':resources.mapped_cache_files(trade_path)}])
    args=SimpleNamespace(**{k:manifest[k] for k in ('holdout_fraction','embargo','n_splits')})
    prefix=p4.SearchPrefix.open(cache,bt,trade_path,args,production=True,dataset_id=DATASET_ID,bar_type=bt)
    prefix.metadata=meta
    prefix.input_paths={'bars':str(cache),'trade':str(trade_path),'dataset_id':DATASET_ID}
    folds=opt.build_chronological_folds(prefix.bars,manifest['n_splits'],manifest['embargo'])
    return prefix,p4.FrozenComparator(artifact['params'],artifact['replay_digest']),folds


def campaign_arm(args,study_name=None):
    layout=args.phase_layout;manifest,digest=read_manifest(layout)
    if args.campaign_identity!=digest or layout.db_path(args.study_db)!=Path(manifest['phase4b_db']):
        raise ValueError('study campaign/database mismatch')
    key=p4.arm_key(args);arm=manifest['arms'][key]
    for name in ('seed','fee_bps','slippage_bps','min_total_trades','min_trades_per_fold','n_splits','embargo','holdout_fraction'):
        if getattr(args,name)!=manifest[name]:raise ValueError('registered setting mismatch: '+name)
    if args.n_trials!=arm['budget'] or args.n_jobs!=1 or not args.no_warm_start:
        raise ValueError('registered budget/sequential/cold-start mismatch')
    for name in ('max_trial_seconds','max_study_seconds'):
        if getattr(args,name)!=arm[name]:raise ValueError('registered runtime limit mismatch')
    if layout.studies_db.exists():
        matches=[name for name in opt.list_study_names(layout.studies_db)
            if (opt.read_study_attrs(layout.studies_db,name).get('identity',{}).get('campaign_sha256')==digest
                and opt.read_study_attrs(layout.studies_db,name).get('identity',{}).get('arm')==key)]
        if len(matches)>1 or matches and study_name is not None and matches!=[study_name]:
            raise ValueError('second study for the same arm refused')
    return arm


def run_study(layout,manifest,digest,bt,prefix,folds):
    """The host-only Optuna path reuses all Phase 4 watchdog/write-ahead safeguards."""
    arm=manifest['arms'][f'BTCUSDT:{bt}:full']
    args=SimpleNamespace(**{k:manifest[k] for k in ('seed','fee_bps','slippage_bps','min_total_trades',
        'min_trades_per_fold','n_splits','embargo','holdout_fraction')},
        phase='phase4b',phase4=True,phase_layout=layout,symbol='BTCUSDT',bar_type=bt,n_jobs=1,no_warm_start=True,
        n_trials=arm['budget'],study_db=layout.studies_db,trade_signals='full',campaign_manifest=layout.manifest,
        campaign_identity=digest,space=p4.FULL_STRATEGY_SPACE,comparator_identity=arm['comparator'],
        trade_updates=prefix.trade_updates,trade_cache=Path(prefix.input_paths['trade']),
        trade_cache_identity=prefix.trade_cache_identity,start_date=date(2019,9,10),end_date=date(2023,12,31),
        max_trial_seconds=arm['max_trial_seconds'],max_study_seconds=arm['max_study_seconds'],
        approval_sha256=manifest['_approval_sha256'],gates_sha256=manifest['_gates_sha256'],
        phase4_timeout_audit=lambda stop:record_termination(layout,manifest,digest,'S2t' if bt==BAR_TYPES[0] else 'S2v',stop=stop),
        phase4_termination_audit=lambda stop:record_termination(layout,manifest,digest,'S2t' if bt==BAR_TYPES[0] else 'S2v',stop=stop),
        phase4_launch=manifest.get('_launch'),
        phase4_progress=lambda trial:stage_progress(layout,digest,'S2t' if bt==BAR_TYPES[0] else 'S2v',
            arm=f'BTCUSDT:{bt}:full',trial=trial.number,marker=f'trial {trial.number}'))
    name,identity=opt.study_identity(args,prefix.metadata)
    lock=layout.checked_write(opt.study_lock_path(layout.studies_db,name))
    with opt.file_lock(lock,nonblocking=True),opt.file_lock(layout.checked_write(str(lock)+'.workers'),nonblocking=True):
        from scripts import phase4b_accounting as accounting
        accounting.prepare(layout.studies_db,name)  # Before open_study's Phase 4 admission (review issue 2).
        study=opt.open_study(args,name,identity)
        accounting.reconcile(study,args)  # Accounting only: estimates are never promoted (rev 13).
        stop=study.user_attrs.get('phase4_runtime_stop')
        if stop is not None:
            args.phase4_termination_audit(stop)
            raise ValueError('study has a durable runtime stop; new registration required')
        accounting.admit(study,args)
        args.phase4_reconciled_stop_hook=lambda study,args:None
        args.phase4_runtime_stop_hook=accounting.runtime_stop
        opt.optimize_loop(study,prefix,folds,args)
        stop=study.user_attrs.get('phase4_runtime_stop')
        if stop is not None:
            raise ValueError('study stopped by resource/runtime ceiling; new registration required')
        if len(opt.complete_trials(study))!=arm['budget']:raise ValueError('study budget incomplete')
    return {'study':name,'complete':len(opt.complete_trials(study)),'identity':identity}


def holdout_replay(prefix,sample,expected,manifest,interval):
    """Called only after G4b-N/allocation checks under the look lock."""
    from scripts.trade_feature_cache import CachedTradeUpdates
    paths=prefix.input_paths
    bars=opt.CachedBars(paths['bars'],prefix.bars[0].bar_type.value)
    updates=CachedTradeUpdates(paths['trade'],bars,production=True,dataset_id=paths['dataset_id'],bar_type=bars.bar_type.value)
    params=p4.full_strategy_params(sample.params)
    args=SimpleNamespace(phase='phase4b')
    n_search=len(prefix.bars)
    start=n_search+int(len(bars)*manifest['embargo'])
    if (bars[start].seq!=interval['start_seq'] or bars[-1].seq!=interval['end_seq']
            or interval['dataset_id']!=paths['dataset_id']):
        raise ValueError('allocated holdout interval differs from registered cache')
    result=opt.run_backtest(bars,opt.build_instrument_config(params),params['price_proximity_pct'],bars.bar_type,
        prefix_ends=(n_search-1,),trade_updates=updates,trade_config=p4.effective_trade_config(args,params))
    if result['prefix_digests'][n_search-1]!=expected:raise ValueError('holdout search-prefix digest mismatch')
    full=opt.simulate_trades_risk_based(result['ledger'],fee_bps=5.,slippage_bps=1.)
    metrics,trades=opt.score_window(full,bars,start,len(bars)-1,11)
    from scripts.phase4_report import expectancy_r
    er=expectancy_r(trades)
    return {**metrics,'net_expectancy_r':er,'G3':bool(er is not None and er>0 and len(trades)>=11
        and not metrics['account_blown']),'conditional_on_full_history_bar_calibration':True,
        'replay_digest':result['replay_digest'],'search_prefix_digest':expected,
        'replay_identity':{'sampled_params':dict(sample.params),
            'inert_params':{'absorption_min_delta_pct':params['absorption_min_delta_pct']}}}


def execute_stage(stage,manifest,digest,layout):
    if stage=='S1':
        lanes={}
        for bt in BAR_TYPES:
            prefix,comparator,_=open_prefix(layout,manifest,bt,stage)
            def replay_guard(label):
                stage_progress(layout,digest,stage,arm=f'BTCUSDT:{bt}:full',marker=label)
                def audit(stop):
                    return record_termination(layout,manifest,digest,stage,stop={k:v for k,v in stop.items() if k!='trial'})
                return file_timing(layout,digest,f'{bt}-{label}',
                    manifest['arms'][f'BTCUSDT:{bt}:full']['max_trial_seconds'],audit=audit,rss_limit=stage_rss_limit(stage))
            lanes[bt]=p4.run_s1(prefix,comparator,guard=replay_guard)
        return {'lanes':lanes}
    if stage in ('S2t','S2v'):
        bt=BAR_TYPES[stage=='S2v'];prefix,_,folds=open_prefix(layout,manifest,bt)
        return run_study(layout,manifest,digest,bt,prefix,folds)
    from scripts import phase4_report as report
    completed,not_reported=completed_s2(layout,digest)
    snapshots=report.phase4b_report_snapshots(layout,manifest,digest,
        reported_arms={f'BTCUSDT:{bt}:full' for bt in completed})
    reports,looks={},{}
    for bt in completed:
        prefix,comparator,folds=open_prefix(layout,manifest,bt)
        names=[name for name in opt.list_study_names(snapshots[-1]['path'])
            if opt.read_study_attrs(snapshots[-1]['path'],name).get('identity',{}).get('arm')==f'BTCUSDT:{bt}:full']
        if len(names)!=1:raise ValueError('report requires exactly one study per arm')
        trials=opt.read_study_trials(snapshots[-1]['path'],names[0])
        def progress(marker,trial=None):
            stage_progress(layout,digest,stage,arm=f'BTCUSDT:{bt}:full',trial=trial,marker=marker)
        result=report.build_full_strategy_report(prefix,trials,comparator,snapshots,folds,progress=progress)
        result.update(campaign_sha256=digest,bar_type=bt,database_snapshots=snapshots)
        write_json(layout,layout.output/'reports'/f'{bt}.json',result)
        reports[bt]=result
        if result['G4b_N']:
            progress('holdout admission',result['trial'])
            best=next(t for t in trials if t['number']==result['trial'])
            values={k:v for k,v in best['attrs']['params'].items() if k in p4.FULL_STRATEGY_SPACE}
            # Read only the registered interval metadata, never a holdout bar, before allocation.
            interval=manifest['holdout_intervals'][bt]
            reservation={'trial':result['trial'],'replay_digest':result['replay_digest'],
                'params_sha256':opt.fingerprint(values),'complete':manifest['arms'][f'BTCUSDT:{bt}:full']['budget']}
            looks[bt]=p4.phase4b_reserved_look(layout,manifest,digest,interval,reservation,result,
                lambda:holdout_replay(prefix,p4.FullStrategySample(values),result['replay_digest'],manifest,interval))
    return {'reports':reports,'not_reported':not_reported,'looks':looks,'conclusion':None if looks else
        'no evidence of net profitability for the full strategy under registered costs'}


def s2_status(layout,digest,bar_type):
    stage='S2t' if bar_type==BAR_TYPES[0] else 'S2v'
    directory=layout.checkpoints/digest
    records=terminations(layout,digest,stage)
    consuming=[r for r in records if r['kind'] in CONSUMING_KINDS]
    if consuming:  # Durable: a stale complete checkpoint never overrules a consuming termination.
        kind=consuming[0]['kind']
        return 'breached' if kind=='timeout_confirmed' else kind
    complete=directory/(stage+'.json')
    if complete.exists():return json.loads(complete.read_bytes()).get('status','unavailable')
    if (directory/f'{stage}.result.pending.json').exists():return 'pending_postflight'
    if records:return 'unknown_termination'
    stopped=directory/(stage+'.stopped.json')
    if stopped.exists():return json.loads(stopped.read_bytes()).get('status','unavailable')
    return 'not_started'


def completed_s2(layout,digest):
    statuses={bt:s2_status(layout,digest,bt) for bt in BAR_TYPES}
    completed=[bt for bt,status in statuses.items() if status=='complete']
    if not completed:raise ValueError('S3 requires at least one completed S2 checkpoint')
    return completed,{bt:'S2 '+status for bt,status in statuses.items() if status!='complete'}


def first_party_memory_stop(layout,manifest,digest,stage):
    """Only a memory stop already written by the Phase 4 watchdog may name a cause."""
    runtime=layout.checkpoints/digest/'runtime'
    labels=[stage] if stage!='S1' else [stage,*BAR_TYPES]
    if runtime.exists():
        for folder in runtime.iterdir():
            if not any(folder.name==l or folder.name.startswith(l+'-') for l in labels):continue
            for name in ('study.json','trial.json'):
                path=folder/name
                if path.exists():
                    stop=json.loads(path.read_bytes()).get('phase4_runtime_stop')
                    if stop and stop.get('reason')=='memory':return stop
    if stage in ('S2t','S2v') and layout.studies_db.exists():
        arm=f"BTCUSDT:{BAR_TYPES[stage=='S2v']}:full"
        for name in opt.list_study_names(layout.studies_db):
            attrs=opt.read_study_attrs(layout.studies_db,name)
            if attrs.get('identity',{}).get('arm')==arm and attrs.get('identity',{}).get('campaign_sha256')==digest:
                stop=attrs.get('phase4_runtime_stop')
                if stop and stop.get('reason')=='memory':return stop
    return None


def _strip(argv,flag):
    out=[];i=0
    while i<len(argv):
        if argv[i]==flag:i+=2
        elif argv[i].startswith(flag+'='):i+=1
        else:out.append(argv[i]);i+=1
    return out


def launch_stage(layout,manifest,digest,stage,child,fds,launch,timeout,*,postflight_only=False):
    """Timed child from L; classification from evidence observed at exit; untimed post-flight child (rev 13)."""
    from scripts import measure_rss
    from scripts.phase4_resources import MemoryBoundsError
    from scripts import phase4_resources as resources
    env={**os.environ,LAUNCH_ENV:json.dumps(launch),LOCK_ENV:json.dumps(fds)}
    attempt={**manifest,'_launch':launch}
    measurement={'returncode':0}
    if not postflight_only:
        ceiling=stage_ceiling(manifest,stage)
        remaining=max(1,math.ceil(ceiling-(_now()-launch['monotonic'])))
        stage_progress(layout,digest,stage,marker='stage launch')
        measurement=measure_rss.run_measured([timeout,'--foreground','--signal=TERM',f'--kill-after={KILL_AFTER_SECONDS}',str(remaining),
            *child],memory=True,env=env,pass_fds=tuple(fds.values()),worker=True)
        elapsed=_now()-launch['monotonic']  # Observed at child exit, before any scan or write.
        # Raw evidence from the worker (review I2); peak_rss_bytes is the wrapper's wait4 rusage.
        memory=measurement.get('last_memory_sample') or ({k:measurement.get(k) for k in
            ('peak_rss_bytes','peak_private_bytes') if k in measurement} or None)
        if memory and 'peak_rss_bytes' in measurement:memory={**memory,'peak_rss_bytes':measurement['peak_rss_bytes']}
        if measurement['returncode']:
            memory_stop=first_party_memory_stop(layout,manifest,digest,stage)
            record=(record_termination(layout,attempt,digest,stage,stop=memory_stop,memory=memory,elapsed=elapsed)
                    if memory_stop else record_termination(layout,attempt,digest,stage,
                    returncode=measurement['returncode'],memory=memory,elapsed=elapsed))
            raise ValueError(f"stage process ended: {record['kind']}")
        try:enforce_stage_memory(measurement,stage)
        except (MemoryBoundsError,ValueError) as exc:
            record_termination(layout,attempt,digest,stage,stop={'reason':'memory','cause':str(exc)},
                               memory=memory,elapsed=elapsed)
            raise ValueError('stage process memory bounds exceeded: strict_memory_stop') from exc
    post=measure_rss.run_measured([sys.executable,str(Path(__file__).resolve()),'postflight','--phase',layout.phase,
        '--stage',stage],memory=False,env=env,pass_fds=tuple(fds.values()))
    if post['returncode']:
        # Post-flight is untimed: its end is never compared with the stage ceiling (elapsed=None -> unknown).
        record_termination(layout,attempt,digest,stage,returncode=post['returncode'],elapsed=None,
                           attempt=launch['attempt']+'-postflight')  # First wins: an integrity_stop stays.
        raise ValueError('post-flight integrity verification did not complete')
    return measurement


def main(argv=None,*,recovery_argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('register','run','worker','postflight'))
    parser.add_argument('--phase',choices=('phase4b-s1','phase4b'),required=True)
    parser.add_argument('--stage',choices=('S1','S2t','S2v','S3'))
    parser.add_argument('--registration',type=Path,help='host-reviewed snapshots, comparators, phase4_origin JSON')
    parser.add_argument('--owner-approved')
    parser.add_argument('--recovery-approved',help='owner recovery record after an external/unknown termination')
    parser.add_argument('--recovery',help=argparse.SUPPRESS)
    cli=parser.parse_args(argv);layout=PhaseLayout(cli.phase)
    try:
        if cli.command=='register':
            from scripts.phase4_campaign import register
            values=json.loads(cli.registration.read_bytes())
            return register(layout.manifest,values['snapshots'],phase=cli.phase,layout=layout,
                comparators=values['comparators'],phase4_origin=values['phase4_origin'],
                benchmark=layout.output/'s1/benchmark.json' if cli.phase=='phase4b' else None,
                holdout_intervals=values.get('holdout_intervals'),stage_limits=values.get('stage_limits',()),
                trial_limits=values.get('trial_limits',()),study_limits=values.get('study_limits',()))
        if not cli.stage:raise ValueError('a registered stage is required')
        if cli.command=='worker':
            return run_stage(layout,cli.stage,cli.owner_approved,finalize=False,recovery=cli.recovery)
        if cli.command=='postflight':
            manifest,_=read_manifest(layout);require_inherited_locks(layout,manifest)
            return postflight_stage(layout,cli.stage)
        manifest,digest=read_manifest(layout)
        timeout=shutil.which('timeout') or shutil.which('gtimeout')
        if not timeout:raise ValueError('external GNU timeout command (--foreground) is required')
        child=[sys.executable,str(Path(__file__).resolve()),'worker','--phase',cli.phase,'--stage',cli.stage]
        # Locks first: contention with another driver never consumes an owner approval.
        with stage_locks(layout,manifest) as fds:
            launch=new_launch()
            if stage_min_available(cli.stage)>stage_min_available(None):
                # A3: host memory for the raised S1 limit is checked before any approval (fresh or recovery)
                # is consumed.
                from scripts.phase4_resources import require_memory_plan
                require_memory_plan([{'mapped_files':{}}],stage_min_available(cli.stage))
            if cli.recovery_approved:
                from scripts.phase4_approvals import recovery_root
                driver_argv=_strip(list(recovery_argv if recovery_argv is not None else
                                        getattr(sys,'orig_argv',sys.argv)),'--recovery-approved')
                approval=recovery_admission(layout,manifest,digest,cli.stage,cli.recovery_approved,driver_argv)
                path=recovery_root(layout)/f"{approval['sha256']}.admitted.json"
                fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                with os.fdopen(fd,'wb') as handle:
                    handle.write(canonical_bytes({'recovery_sha256':approval['sha256'],'campaign_sha256':digest,
                        'stage':cli.stage,'attempt':launch['attempt'],'gates_sha256':approval['gates_sha256'],
                        'inspection_basis':approval.get('inspection_basis'),
                        'termination_sha256':approval['record']['termination_sha256']}))
                    handle.flush();os.fsync(handle.fileno())
                pending=(layout.checkpoints/digest/f'{cli.stage}.result.pending.json').exists()
                return launch_stage(layout,manifest,digest,cli.stage,[*child,'--recovery',approval['sha256']],
                                    fds,launch,timeout,postflight_only=pending)
            termination_admission(layout,manifest,digest,cli.stage,None,None)
            from scripts.phase4_approvals import verify_phase4b_approval
            matches=[]
            for path in layout.approval_root.glob('*.json'):
                if len(path.stem)!=64:continue
                try:verify_phase4b_approval(layout,path.stem,digest,cli.stage,child)
                except ValueError:continue
                matches.append(path.stem)
            if len(matches)!=1:raise ValueError('exactly one argv-bound host child approval required')
            require_phase4b_approval(layout,cli.owner_approved,digest,cli.stage,consume=True)
            return launch_stage(layout,manifest,digest,cli.stage,[*child,'--owner-approved',matches[0]],
                                fds,launch,timeout)
    except (OSError,ValueError,RuntimeError) as exc:parser.error(str(exc))


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
