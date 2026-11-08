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
PHASE4B_PLAN_SHA256 = '75b22452c0b168f43fd91704a7aee23a4cb1c84c637f4a8f775b87be13857b7f'
PHASE4B_PLAN_PATH = ROOT/'docs/plans/phase4b-full-strategy-plan-rev13.md'
PHASE4_CAMPAIGN_SHA256 = 'b54076c64820ed75cfbaff35a311b1ffecd294944c4f5c1d1189c8c60df32434'
# Amendment A2 (owner, 2026-11-08): S1 calibration may run on the interim review. R is never recorded as
# 'pass' for it; the exception value is accepted only for the phase4b-s1 manifest and an owner record.
PHASE4B_S1_EXCEPTION_PATH = ROOT/'docs/plans/phase4b-s1-inspection-exception-A2.md'
PHASE4B_A1_PATH = ROOT/'docs/plans/phase4b-proof-amendment-A1-search-only-real-data.md'
S1_EXCEPTION_R = 'owner_exception_interim_review'
EXCEPTION_FIELDS = {'kind','phase','stage','campaign_sha256','plan_sha256','source_state','amendment_sha256',
                    'interim_review_sha256','owner_statement','created_utc'}


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


def consume_approval(approval, *, layout=None):
    """Full builds are single-use; campaign admissions append one audit line per use."""
    root=approval_root() if layout is None else layout.approval_root
    trusted_path(root.parent,directory=True)
    trusted_path(root,directory=True)
    trusted_path(root/(approval['sha256']+'.json'))
    path=root/(approval['sha256']+'.consumed')
    single=approval['record']['scope']=='full_history_tfc' or approval['record'].get('phase') in ('phase4b','phase4b-s1')
    if path.exists() or path.is_symlink():
        trusted_path(path)
    flags=os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW|(os.O_EXCL if single else os.O_APPEND)
    try:
        fd=os.open(path,flags,0o600)
    except FileExistsError as exc:
        raise ValueError('full-history approval already consumed' if layout is None else
                         'single-use approval already consumed') from exc
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


def phase4_exposure_state(origin, bar_type):
    """Resolve the canonical origin independently of the executing checkout."""
    try:
        root = Path(origin['root'])
        manifest = Path(origin['manifest_path'])
        exposure = Path(origin['exposure_path'])
        if (not root.is_absolute() or manifest != root/'output/phase4/campaign.json'
                or exposure != root/'output/phase4/holdout/exposures.json'):
            return 'unavailable'
        seal=Path(str(manifest)+'.sha256')
        if (origin['sha256']!=PHASE4_CAMPAIGN_SHA256 or manifest.is_symlink() or not manifest.is_file()
                or seal.is_symlink() or not seal.is_file() or seal.read_text().strip()!=PHASE4_CAMPAIGN_SHA256):
            return 'unavailable'
        raw = manifest.read_bytes()
        if hashlib.sha256(raw).hexdigest() != PHASE4_CAMPAIGN_SHA256 or json.loads(raw)['phase'] != 'phase4':
            return 'unavailable'
        if exposure.is_symlink():
            return 'unavailable'
        if not exposure.exists():
            return 'verified_absent'
        data = json.loads(exposure.read_bytes())
        if not isinstance(data, dict):
            return 'unavailable'
        return 'present' if any(key.startswith(f'BTCUSDT:{bar_type}:') for key in data) else 'verified_absent'
    except (OSError, KeyError, TypeError, ValueError):
        return 'unavailable'


def allocation_path(layout, interval):
    if (set(interval) != {'dataset_id','symbol','bar_type','start_seq','end_seq'}
            or interval['symbol'] != 'BTCUSDT'
            or interval['bar_type'] not in ('fixed_tick_imbalance','fixed_volume_imbalance')
            or not re.fullmatch(r'[A-Za-z0-9_-]+', interval['dataset_id'])
            or any(type(interval[k]) is not int or interval[k] < 0 for k in ('start_seq','end_seq'))
            or interval['start_seq'] > interval['end_seq']):
        raise ValueError('invalid holdout allocation interval')
    name = '{dataset_id}__{symbol}__{bar_type}__{start_seq}-{end_seq}.json'.format(**interval)
    return layout.allocation_root/name


def install_holdout_allocation(layout, interval, campaign_sha256, origin):
    """Host-only installation primitive. Campaign execution never calls this."""
    if layout.phase != 'phase4b' or not re.fullmatch('[0-9a-f]{64}', campaign_sha256):
        raise ValueError('allocation requires a phase4b campaign hash')
    if phase4_exposure_state(origin, interval['bar_type']) != 'verified_absent':
        raise ValueError('canonical Phase 4 exposure is present or unavailable')
    path = allocation_path(layout, interval)
    allocation = {**interval,'phase':'phase4b','campaign_sha256':campaign_sha256,'phase4_origin':origin}
    raw = canonical_bytes({'allocation':allocation,'sha256':hashlib.sha256(canonical_bytes(allocation)).hexdigest()})
    path.parent.mkdir(parents=True, exist_ok=True)
    trusted_path(path.parent, directory=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(raw);handle.flush();os.fsync(handle.fileno())
    return path


def require_holdout_allocation(layout, interval, campaign_sha256, origin):
    path = allocation_path(layout, interval)
    try:
        trusted_path(path.parent, directory=True)
        trusted_path(path)
        raw = path.read_bytes()
        envelope = json.loads(raw)
        allocation = envelope['allocation']
        if (path.stat().st_mode & 0o777 != 0o600 or canonical_bytes(envelope) != raw
                or hashlib.sha256(canonical_bytes(allocation)).hexdigest() != envelope['sha256']
                or allocation != {**interval,'phase':'phase4b','campaign_sha256':campaign_sha256,'phase4_origin':origin}):
            raise ValueError('holdout allocation content/hash mismatch')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError('holdout allocation missing, untrusted or mismatched') from exc
    state = phase4_exposure_state(origin, interval['bar_type'])
    if state != 'verified_absent':
        raise ValueError('canonical Phase 4 exposure/origin is '+state)
    return allocation


def verify_phase4b_approval(layout, selector, campaign_sha256, stage, argv=None):
    """Use the same canonical, uid-owned, source/argv-bound host records as Phase 4."""
    if not isinstance(selector,str) or not re.fullmatch('[0-9a-f]{64}',selector):
        raise ValueError('host approval requires a content-addressed record hash')
    root = layout.approval_root
    for path in (root.parent,root):trusted_path(path,directory=True)
    path = root/(selector+'.json');trusted_path(path)
    raw=path.read_bytes();record=json.loads(raw)
    fields={'scope','phase','stage','plan_sha256','campaign_sha256','source_state','base_commit','command_argv','created_utc'}
    if (not isinstance(record,dict) or set(record)!=fields or canonical_bytes(record)!=raw
            or hashlib.sha256(raw).hexdigest()!=selector):
        raise ValueError('approval fields/content hash mismatch')
    if (layout.phase not in ('phase4b','phase4b-s1') or record['scope']!='campaign'
            or record['phase']!=layout.phase or record['stage']!=stage
            or stage not in (('S1',) if layout.phase=='phase4b-s1' else ('S2t','S2v','S3'))):
        raise ValueError('approval phase/stage mismatch')
    if (record['plan_sha256']!=PHASE4B_PLAN_SHA256
            or hashlib.sha256(PHASE4B_PLAN_PATH.read_bytes()).hexdigest()!=PHASE4B_PLAN_SHA256):
        raise ValueError('approval plan hash mismatch')
    current=source_state()
    if record['source_state']!=current or record['base_commit']!=current['base'] or record['campaign_sha256']!=campaign_sha256:
        raise ValueError('approval source/campaign mismatch')
    try:
        created=datetime.fromisoformat(record['created_utc'].replace('Z','+00:00'))
        if created.tzinfo is None or created.utcoffset().total_seconds()!=0:raise ValueError('not UTC')
    except (KeyError,TypeError,ValueError,AttributeError) as exc:
        raise ValueError('approval created_utc must be UTC') from exc
    if record['command_argv']!=command_argv(argv):
        raise ValueError('approval command argv mismatch')
    return {'sha256':selector,'record':record}


def require_phase4b_approval(layout, selector, campaign_sha256, stage, argv=None, *, consume=False):
    approval=verify_phase4b_approval(layout,selector,campaign_sha256,stage,argv)
    gates_sha256,basis=verify_phase4b_gates(layout,campaign_sha256,stage,with_basis=True)
    if consume:consume_approval(approval,layout=layout)
    return {**approval,'gates_sha256':gates_sha256,'inspection_basis':basis}


def inspection_basis(layout, campaign_sha256):
    """What R rested on, for the stage record and campaign log (A2): never inferred from the gates sha alone."""
    return _basis(json.loads((layout.approval_root/f'gates-{campaign_sha256}.json').read_bytes()))


def _basis(gates):
    by_role={a.get('role'):a for a in gates.get('artifacts',[])}
    return {'R':gates.get('R'),
            'owner_exception_sha256':by_role.get('owner_exception',{}).get('sha256'),
            'interim_review_sha256':by_role.get('interim_review',{}).get('sha256')}


def _utc(value, what):
    try:
        created=datetime.fromisoformat(str(value).replace('Z','+00:00'))
        if created.tzinfo is None or created.utcoffset().total_seconds()!=0 or 'T' not in str(value):
            raise ValueError('not UTC')
    except (TypeError,ValueError) as exc:
        raise ValueError(what+' created_utc must be an ISO UTC timestamp') from exc


def verify_proof_artifacts(by_role):
    """A1: the unit artifact excludes real-data tests; the real_data artifact is the guarded search-only run."""
    from scripts import phase4b_proof as proof
    try:
        unit=json.loads(Path(by_role['unit']['path']).read_bytes())
        real=json.loads(Path(by_role['real_data']['path']).read_bytes())
    except (OSError,ValueError,KeyError) as exc:
        raise ValueError('phase4b proof artifacts must be A1 JSON records') from exc
    if (unit.get('argv') or [None])[1:]!=proof.proof_argv('unit')[1:]:
        raise ValueError('unit proof must be exactly the A1 command (-m "not real_data")')
    if unit.get('failed')!=0 or unit.get('errors')!=0:
        raise ValueError('unit proof must have no failures or errors')
    selection=proof.real_data_selection();argv=real.get('argv') or []
    if argv[-len(selection):]!=selection:raise ValueError('real_data proof must be the A1 command')
    collected=real.get('collected') or []
    if (len(collected)!=proof.EXPECTED_COLLECTED or len(set(collected))!=len(collected)
            or any(c.startswith(d) for c in collected for d in proof.DESELECTED)):
        raise ValueError('real_data proof collected nodes do not match A1')
    if real.get('horizon_utc')!=proof.HORIZON_UTC:raise ValueError('real_data proof horizon mismatch')
    if real.get('amendment_sha256')!=hashlib.sha256(PHASE4B_A1_PATH.read_bytes()).hexdigest():
        raise ValueError('real_data proof amendment A1 hash mismatch')
    if (real.get('failed'),real.get('errors'),real.get('skipped'),real.get('passed'))!=(0,0,0,proof.EXPECTED_COLLECTED):
        raise ValueError('real_data proof must pass every collected node (failed/skipped/errors = 0)')


def verify_phase4b_gates(layout, campaign_sha256, stage=None, *, with_basis=False):
    """Host proof gates (plan hash, U/AC1/AC7/R, proof artifacts); returns the gates record sha256."""
    if hashlib.sha256(PHASE4B_PLAN_PATH.read_bytes()).hexdigest()!=PHASE4B_PLAN_SHA256:
        raise ValueError('approval plan hash mismatch: the plan file changed')
    path=layout.approval_root/f'gates-{campaign_sha256}.json';trusted_path(path)
    raw=path.read_bytes();gates=json.loads(raw)
    exception=gates.get('R')==S1_EXCEPTION_R and layout.phase=='phase4b-s1' and stage=='S1'
    if (gates.get('plan_sha256')!=PHASE4B_PLAN_SHA256 or gates.get('campaign_sha256')!=campaign_sha256
            or gates.get('source_state')!=source_state()
            or any(gates.get(key)!='pass' for key in ('U','AC1','AC7'))
            or not (gates.get('R')=='pass' or exception)):
        raise ValueError('phase4b requires host unit, digest, preservation/real-data and inspection gates')
    artifacts=gates.get('artifacts',[])
    roles=[a.get('role') for a in artifacts]
    expected=({'unit','digests','protected','real_data','interim_review','owner_exception'} if exception
              else {'unit','digests','protected','real_data','inspection'})
    if set(roles)!=expected or len(roles)!=len(expected):
        raise ValueError('phase4b host proof artifacts incomplete')
    for artifact in artifacts:
        raw_artifact=Path(artifact['path']).read_bytes()
        if hashlib.sha256(raw_artifact).hexdigest()!=artifact['sha256']:
            raise ValueError('host proof artifact hash mismatch')
    verify_proof_artifacts({a['role']:a for a in artifacts})
    if exception:
        by_role={a['role']:a for a in artifacts}
        verify_inspection_exception(layout,by_role['owner_exception'],campaign_sha256,
                                    by_role['interim_review']['sha256'])
    digest=hashlib.sha256(raw).hexdigest()
    return (digest,_basis(gates)) if with_basis else digest


def exception_root(layout):
    return layout.approval_root/'exceptions'


def install_inspection_exception(layout, record):
    """Host-only, after the owner's explicit decision (A2): content-addressed, 0600, O_EXCL, S1 only."""
    if not isinstance(record,dict) or set(record)!=EXCEPTION_FIELDS or record['kind']!='inspection_exception':
        raise ValueError('inspection exception fields invalid')
    if layout.phase!='phase4b-s1' or record['phase']!='phase4b-s1':
        raise ValueError('inspection exception is limited to the phase4b-s1 manifest')
    if record['stage']!='S1':raise ValueError('inspection exception is limited to stage S1')
    _utc(record['created_utc'],'inspection exception')
    raw=canonical_bytes(record);root=exception_root(layout)
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    path=root/(hashlib.sha256(raw).hexdigest()+'.json')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as handle:
        handle.write(raw);handle.flush();os.fsync(handle.fileno())
    return path


def verify_inspection_exception(layout, artifact, campaign_sha256, interim_review_sha256):
    root=exception_root(layout);path=Path(artifact['path'])
    try:
        for directory in (root.parent,root):trusted_path(directory,directory=True)
        trusted_path(path)
    except ValueError as exc:
        raise ValueError('inspection exception record untrusted: '+str(exc)) from exc
    if path.parent.resolve()!=root.resolve() or path.stat().st_mode & 0o077 or root.stat().st_mode & 0o077:
        raise ValueError('inspection exception record must be private and inside the exception root')
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    record=json.loads(raw)
    if (path.name!=digest+'.json' or artifact['sha256']!=digest or not isinstance(record,dict)
            or set(record)!=EXCEPTION_FIELDS or canonical_bytes(record)!=raw):
        raise ValueError('inspection exception record content mismatch')
    policy=hashlib.sha256(PHASE4B_S1_EXCEPTION_PATH.read_bytes()).hexdigest()
    if (record['kind']!='inspection_exception' or record['phase']!=layout.phase or record['stage']!='S1'
            or record['campaign_sha256']!=campaign_sha256 or record['plan_sha256']!=PHASE4B_PLAN_SHA256
            or record['source_state']!=source_state() or record['amendment_sha256']!=policy
            or record['interim_review_sha256']!=interim_review_sha256):
        raise ValueError('inspection exception record binding mismatch')
    _utc(record['created_utc'],'inspection exception')
    return record


# Rev 13: only an external/unknown termination is recoverable, and only with an owner record.
RECOVERABLE_KIND = 'external_or_unknown_termination'
RECOVERY_FIELDS = {'kind','phase','campaign_sha256','stage','termination_sha256','runtime_fingerprint','command_argv'}


def recovery_root(layout):
    return layout.approval_root/'recovery'


def install_recovery_approval(layout, record):
    """Host-only, after the owner's explicit decision: content-addressed, 0600, O_EXCL."""
    if not isinstance(record,dict) or set(record)!=RECOVERY_FIELDS or record['kind']!='recovery':
        raise ValueError('recovery approval fields invalid')
    raw=canonical_bytes(record);root=recovery_root(layout)
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    path=root/(hashlib.sha256(raw).hexdigest()+'.json')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as handle:
        handle.write(raw);handle.flush();os.fsync(handle.fileno())
    return path


def require_recovery_approval(layout, selector, campaign_sha256, stage, termination, argv, *, consume=False):
    """Bind manifest, stage, termination record, runtime fingerprint and exact argv; single use."""
    from scripts.phase4b_runtime import runtime_fingerprint
    if not isinstance(selector,str) or not re.fullmatch('[0-9a-f]{64}',selector):
        raise ValueError('recovery approval requires a content-addressed record hash')
    if termination.get('kind')!=RECOVERABLE_KIND:
        raise ValueError('no recovery for termination kind '+str(termination.get('kind')))
    root=recovery_root(layout);path=root/(selector+'.json')
    try:
        for directory in (layout.approval_root,root):trusted_path(directory,directory=True)
        trusted_path(path)
    except ValueError as exc:raise ValueError('recovery approval not trusted: '+str(exc)) from exc
    if path.stat().st_mode&0o077:raise ValueError('recovery approval not trusted: mode must be 0600')
    raw=path.read_bytes()
    try:record=json.loads(raw)
    except ValueError as exc:raise ValueError('recovery approval invalid') from exc
    if canonical_bytes(record)!=raw or hashlib.sha256(raw).hexdigest()!=selector or set(record)!=RECOVERY_FIELDS:
        raise ValueError('recovery approval content hash/fields mismatch')
    expected={'kind':'recovery','phase':layout.phase,'campaign_sha256':campaign_sha256,'stage':stage,
        'termination_sha256':hashlib.sha256(canonical_bytes(termination)).hexdigest(),
        'runtime_fingerprint':runtime_fingerprint(),'command_argv':command_argv(argv)}
    for key,value in expected.items():
        if record[key]!=value:raise ValueError('recovery approval '+key+' mismatch')
    consumed=root/(selector+'.consumed')
    if consumed.exists() or consumed.is_symlink():raise ValueError('recovery approval already consumed')
    if consume:
        fd=os.open(consumed,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as handle:
            handle.write(canonical_bytes({'recovery_sha256':selector,'termination_sha256':expected['termination_sha256'],
                'stage':stage,'campaign_sha256':campaign_sha256,'pid':os.getpid(),
                'used_utc':datetime.now(timezone.utc).isoformat()}))
            handle.flush();os.fsync(handle.fileno())
    return {'sha256':selector,'record':record}


def consumed_recoveries(layout):
    """termination_sha256 -> consumed recovery marker, for admission decisions."""
    root=recovery_root(layout);found={}
    if root.exists():
        for path in root.glob('*.consumed'):
            approval=root/(path.name[:-len('.consumed')]+'.json')
            try:
                trusted_path(path);trusted_path(approval)
                raw=approval.read_bytes();record=json.loads(raw);marker=json.loads(path.read_bytes())
            except (ValueError,OSError):continue
            # Only a marker for a genuine, content-addressed approval that binds this termination counts.
            if (hashlib.sha256(raw).hexdigest()!=approval.stem or record.get('kind')!='recovery'
                    or marker.get('termination_sha256')!=record.get('termination_sha256')):
                continue
            found[marker['termination_sha256']]=marker
    return found
