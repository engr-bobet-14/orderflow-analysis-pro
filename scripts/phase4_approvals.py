"""Read host-owned approvals and log their use; never create approval records."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import re
import stat
import sys

from scripts.phase4_policy import ROOT, canonical_bytes, source_state

PLAN_SHA256 = '30bde18198de534eaad6b5d47cff0fdbe89f706fe6350093c25b31726ddee258'
PLAN_PATH = ROOT/'docs/plans/phase4-optuna-trade-signals-plan.md'
GATES = ('U','FH1','FH2','FH3','FH4','R')


def approval_root():
    return Path(pwd.getpwuid(os.getuid()).pw_dir)/'.orderflow-approvals/phase4'


def trusted_path(path, *, directory=False):
    if path.is_symlink() or not path.exists():
        raise ValueError('host approval record/directory missing or symlinked')
    info=path.stat()
    if info.st_uid!=os.getuid() or info.st_mode & 0o022:
        raise ValueError('approval paths must be owned by current uid and not group/world-writable')
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError('invalid approval path type')


def consume_approval(approval):
    """Full builds are single-use; campaign admissions append one audit line per use."""
    root=approval_root()
    trusted_path(root.parent,directory=True)
    trusted_path(root,directory=True)
    trusted_path(root/(approval['sha256']+'.json'))
    path=root/(approval['sha256']+'.consumed')
    single=approval['record']['scope']=='full_history_tfc'
    if path.exists() or path.is_symlink():
        trusted_path(path)
    flags=os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW|(os.O_EXCL if single else os.O_APPEND)
    try:
        fd=os.open(path,flags,0o600)
    except FileExistsError as exc:
        raise ValueError('full-history approval already consumed') from exc
    with os.fdopen(fd,'ab') as handle:
        handle.write(canonical_bytes({'approval_sha256':approval['sha256'],
            'command_argv':approval['record']['command_argv'],'pid':os.getpid(),
            'used_utc':datetime.now(timezone.utc).isoformat()}))
        handle.flush()
        os.fsync(handle.fileno())


def command_argv(argv=None):
    """The selector is excluded to avoid embedding a record's hash in its own bytes."""
    values = list(argv if argv is not None else getattr(sys,'orig_argv',[sys.executable,*sys.argv]))
    result=[]
    i=0
    while i < len(values):
        if values[i]=='--owner-approved':
            if i+1 == len(values):
                raise ValueError('approval selector requires a record hash')
            i+=2
        elif values[i].startswith('--owner-approved='):
            i+=1
        else:
            result.append(values[i]);i+=1
    return result


def _state(record,campaign_sha256=None):
    if hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()!=PLAN_SHA256 or record.get('plan_sha256')!=PLAN_SHA256:
        raise ValueError('approval plan hash mismatch')
    current=source_state()
    if record.get('source_state')!=current or record.get('base_commit')!=current['base']:
        raise ValueError('approval source state/base commit mismatch')
    if campaign_sha256 is not None and record.get('campaign_sha256')!=campaign_sha256:
        raise ValueError('approval campaign hash mismatch')
    try:
        created=datetime.fromisoformat(record['created_utc'].replace('Z','+00:00'))
        if created.tzinfo is None or created.utcoffset().total_seconds()!=0:
            raise ValueError('not UTC')
    except (KeyError,TypeError,ValueError,AttributeError) as exc:
        raise ValueError('approval created_utc must be a UTC timestamp') from exc


def verify_approval(scope,selector,campaign_sha256=None,argv=None):
    if not isinstance(selector,str) or not re.fullmatch('[0-9a-f]{64}',selector):
        raise ValueError('host approval requires a content-addressed record hash')
    root=approval_root()
    path=root/(selector+'.json')
    trusted_path(root.parent,directory=True)
    trusted_path(root,directory=True)
    trusted_path(path)
    raw=path.read_bytes()
    try:
        record=json.loads(raw)
    except (ValueError,UnicodeError) as exc:
        raise ValueError('invalid host approval record') from exc
    if not isinstance(record,dict) or canonical_bytes(record)!=raw or hashlib.sha256(raw).hexdigest()!=selector:
        raise ValueError('approval name/content hash mismatch')
    fields={'scope','plan_sha256','command_argv','base_commit','source_state','created_utc'}
    if scope=='campaign':
        fields.add('campaign_sha256')
    if set(record)!=fields or record.get('scope')!=scope or scope not in ('full_history_tfc','campaign'):
        raise ValueError('approval scope/fields mismatch')
    _state(record,campaign_sha256)
    if record['command_argv']!=command_argv(argv):
        raise ValueError('approval command argv mismatch')
    if scope=='full_history_tfc' and ((root/(selector+'.consumed')).exists() or (root/(selector+'.consumed')).is_symlink()):
        raise ValueError('full-history approval already consumed')
    return {'sha256':selector,'record':record}


def require_campaign_approval(selector,campaign_sha256,argv=None):
    approval=verify_approval('campaign',selector,campaign_sha256,argv)
    path=approval_root()/f'gates-{campaign_sha256}.json'
    trusted_path(path)
    raw=path.read_bytes()
    try:
        gates=json.loads(raw)
    except (ValueError,UnicodeError) as exc:
        raise ValueError('invalid host campaign gates record') from exc
    if not isinstance(gates,dict):
        raise ValueError('invalid host campaign gates record')
    _state(gates,campaign_sha256)
    if any(gates.get(key)!='pass' for key in GATES):
        raise ValueError('campaign requires host U, FH1-FH4 and inspection gates')
    try:
        verify_campaign_artifacts(gates,campaign_sha256)
    except (OSError,KeyError,TypeError,AttributeError) as exc:
        raise ValueError(f'invalid campaign gate artifacts: {exc}') from exc
    consume_approval(approval)
    return {**approval,'gates_sha256':hashlib.sha256(raw).hexdigest()}


def _bound_json(path,digest):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError(f'campaign artifact hash mismatch: {path}')
    return json.loads(raw)


def verify_runtime_decision(selector,identity,stop):
    """A host decision acknowledges this exact stop; it never waives campaign gates."""
    from scripts.optimize_orderflow import fingerprint
    if not isinstance(selector,str) or not re.fullmatch('[0-9a-f]{64}',selector):
        raise ValueError('runtime decision requires a content-addressed record hash')
    root=approval_root()
    for path in (root.parent,root):trusted_path(path,directory=True)
    path=root/f'runtime-decision-{selector}.json';trusted_path(path)
    raw=path.read_bytes();record=json.loads(raw)
    fields={'plan_sha256','study_identity_sha256','stop_sha256','decision','created_utc'}
    if (not isinstance(record,dict) or set(record)!=fields or canonical_bytes(record)!=raw
            or hashlib.sha256(raw).hexdigest()!=selector):
        raise ValueError('runtime decision name/content hash or fields mismatch')
    if (hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()!=PLAN_SHA256 or record['plan_sha256']!=PLAN_SHA256
            or record['study_identity_sha256']!=fingerprint(identity) or record['stop_sha256']!=fingerprint(stop)
            or record['decision']!='clear_and_restart_budget'):
        raise ValueError('runtime decision identity, stop or decision mismatch')
    try:
        created=datetime.fromisoformat(record['created_utc'].replace('Z','+00:00'))
        if created.tzinfo is None or created.utcoffset().total_seconds()!=0:raise ValueError('not UTC')
    except (TypeError,ValueError,AttributeError) as exc:
        raise ValueError('runtime decision created_utc must be UTC') from exc
    return {'sha256':selector,'record':record}


def _resource_gates(value,path=()):
    """Include conservative-bound gates as well as the fitted estimates."""
    result={}
    if isinstance(value,dict):
        for key,item in value.items():
            if key=='resource_gates':
                if not isinstance(item,dict) or not item or any(type(v) is not bool for v in item.values()):
                    raise ValueError('invalid benchmark resource_gates')
                for name,passed in item.items():result['.'.join((*path,name))]=passed
            else:
                result.update(_resource_gates(item,(*path,key)))
    elif isinstance(value,list):
        for index,item in enumerate(value):
            result.update(_resource_gates(item,(*path,str(index))))
    return result


def _waived_gates(gates,campaign_sha256,failed):
    waived=set()
    selectors=gates.get('waivers',[])
    if not isinstance(selectors,list):raise ValueError('waivers must be a list of hashes')
    for selector in selectors:
        if not isinstance(selector,str) or not re.fullmatch('[0-9a-f]{64}',selector):
            raise ValueError('invalid waiver selector')
        path=approval_root()/f'waiver-{selector}.json'
        trusted_path(path)
        raw=path.read_bytes();record=json.loads(raw)
        fields={'gate_path','benchmark_sha256','plan_sha256','campaign_sha256','replacement_gate','created_utc'}
        if (not isinstance(record,dict) or set(record)!=fields or canonical_bytes(record)!=raw
                or hashlib.sha256(raw).hexdigest()!=selector):
            raise ValueError('waiver name/content hash or fields mismatch')
        name=record['gate_path']
        if (name not in failed or name in waived or record['benchmark_sha256']!=gates['benchmark_sha256']
                or record['plan_sha256']!=PLAN_SHA256 or record['campaign_sha256']!=campaign_sha256):
            raise ValueError('unknown, stale or mismatched waiver')
        try:
            created=datetime.fromisoformat(record['created_utc'].replace('Z','+00:00'))
            if created.tzinfo is None or created.utcoffset().total_seconds()!=0:raise ValueError('not UTC')
            replacement=record['replacement_gate']
            if set(replacement)!={'name','threshold_spec','measured_artifact_path','measured_artifact_sha256'} or not replacement['name']:
                raise ValueError('replacement fields')
            measured=_bound_json(Path(replacement['measured_artifact_path']),replacement['measured_artifact_sha256'])
            spec=replacement['threshold_spec']
            if set(spec)!={'metric','operator','value'} or spec['operator'] not in ('<=','<','>=','>','=='):
                raise ValueError('threshold spec')
            if not isinstance(spec['metric'],list) or not spec['metric']:raise ValueError('metric path')
            value=measured
            for key in spec['metric']:value=value[key]
            threshold=spec['value']
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in (value,threshold)):
                raise ValueError('finite numeric threshold required')
            passed={'<=':value<=threshold,'<':value<threshold,'>=':value>=threshold,'>':value>threshold,'==':value==threshold}[spec['operator']]
            if not passed:raise ValueError('replacement threshold failed')
        except (OSError,KeyError,TypeError,ValueError,AttributeError) as exc:
            raise ValueError(f'waiver replacement invalid: {exc}') from exc
        waived.add(name)
    return waived


def verify_campaign_artifacts(gates,campaign_sha256):
    """Read certification evidence before opening any study or mapping a cache."""
    from scripts.trade_feature_cache import production_generation,builder_fingerprint
    from scripts.phase4_benchmark import DATASET_ID
    generation_path=ROOT/'output/optimize/trade_cache'/DATASET_ID
    paths={'benchmark':ROOT/'output/phase4/benchmark/benchmark.json',
           'complete':generation_path/'COMPLETE.json','build_report':ROOT/'output/phase4/cache-build.json',
           'verify_report':ROOT/'output/phase4/fullhistory/verify.stdout'}
    data={name:_bound_json(path,gates.get(name+'_sha256')) for name,path in paths.items()}
    resource=_resource_gates(data['benchmark'])
    required={'arms_under_3x','build_under_4h','disk_under_40GiB','worker_under_4GiB','sum_under_24GiB'}
    if not required<={p.rsplit('.',1)[-1] for p in resource}:raise ValueError('benchmark resource_gates missing')
    failed={name for name,passed in resource.items() if not passed}
    waived=_waived_gates(gates,campaign_sha256,failed)
    if failed-waived:raise ValueError('benchmark gates failed: '+', '.join(sorted(failed-waived)))
    generation=production_generation(generation_path)
    build,verified=data['build_report'],data['verify_report']['lanes']
    lanes=generation['lanes']
    if (build.get('full_history') is not True or build.get('mode')!='production'
            or build.get('dataset_id')!=generation['dataset_id'] or build.get('root_hash')!=generation['root_hash']
            or build.get('cache_paths')!=lanes or set(verified)!=set(lanes.values())
            or set(build['lanes'])!=set(lanes)):
        raise ValueError('certified reports differ from current COMPLETE generation')
    refs=json.loads((ROOT/'tests/fixtures/phase4_reference_digests.json').read_bytes())
    for bt,path in lanes.items():
        meta=json.loads((Path(path)/'manifest.json').read_bytes())
        off,all_on=refs['full_off']['lanes'][bt],refs['full_all']['lanes'][bt]
        if (meta['builder_fp']!=builder_fingerprint() or meta['start'] is not None or meta['end'] is not None
                or meta['n_bars']!=off['bars'] or meta['trade_digest']!=off['trade_digest']
                or meta['trade_digest']!=all_on['trade_digest'] or build['lanes'][bt]['trade_digest']!=meta['trade_digest']
                or build['lanes'][bt]['replay_digest']!=off['replay_digest']
                or any(verified[path]['off'][k]!=off[k] for k in ('replay_digest',))
                or any(verified[path]['all'][k]!=all_on[k] for k in ('replay_digest','raw_signal_digest'))):
            raise ValueError(f'certified digest mismatch: {bt}')


def approval_for_command(campaign_sha256,argv):
    """A driver may use only a matching host record for each child, never mint one."""
    matches=[]
    for path in approval_root().glob('*.json'):
        if not re.fullmatch('[0-9a-f]{64}',path.stem):
            continue
        try:
            verify_approval('campaign',path.stem,campaign_sha256,argv)
        except ValueError:
            continue
        matches.append(path.stem)
    if len(matches)!=1:
        raise ValueError('exactly one host approval record is required for this child command')
    return matches[0]
