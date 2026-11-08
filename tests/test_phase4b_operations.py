"""C7/C8b: temporary namespaces, sealed limits, approval and exposure isolation."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import phase4_policy as policy, phase4_campaign as campaign, phase4_approvals as approvals
from scripts import phase4_studies as p4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervals():
    return {bt:{'dataset_id':'synthetic','symbol':'BTCUSDT','bar_type':bt,'start_seq':81,'end_seq':100}
            for bt in ('fixed_tick_imbalance','fixed_volume_imbalance')}


def benchmark_lane():
    return {'replays':{name:{'runtime_seconds':200.} for name in p4.S1_REPLAY_NAMES},
        'trial_equivalent_seconds':[200.,200.,200.],
        'worst_case_probe':{'params':p4.full_strategy_probe(),'runtime_seconds':400.}}


def fixture_inputs(tmp_path,monkeypatch=None):
    layout = policy.PhaseLayout('phase4b', root=tmp_path, approvals_base=tmp_path/'approvals')
    snapshots = []
    for rel in ('output/optimize/studies.db','output/optimize/main/studies.db',
                'output/optimize/smoke/studies.db','output/optimize/phase4/studies.db'):
        path = tmp_path/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fake snapshot')
        snapshots.append({'path':str(path),'sha256':sha(path)})
    comparators = {}
    for bt in ('fixed_tick_imbalance','fixed_volume_imbalance'):
        path = tmp_path/'output/phase4/frozen'/f'{bt}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'study':bt,'params':{},'replay_digest':'digest'}))
        comparators[bt] = {'path':str(path),'sha256':sha(path),'study':bt}
    origin_root = tmp_path/'origin'
    manifest = origin_root/'output/phase4/campaign.json'
    manifest.parent.mkdir(parents=True);manifest.write_bytes(policy.canonical_bytes({'phase':'phase4'}))
    Path(str(manifest)+'.sha256').write_text(sha(manifest)+'\n')
    if monkeypatch is not None:monkeypatch.setattr(approvals,'PHASE4_CAMPAIGN_SHA256',sha(manifest))
    origin = {'root':str(origin_root),'manifest_path':str(manifest),'sha256':sha(manifest),
              'exposure_path':str(origin_root/'output/phase4/holdout/exposures.json')}
    return layout,snapshots,comparators,origin


def completed_arm(layout,digest,bt='fixed_tick_imbalance'):
    path=layout.checkpoints/digest/('S2t.json' if bt=='fixed_tick_imbalance' else 'S2v.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'status':'complete'}))


def test_c7_phase_layout_write_and_database_policy(tmp_path):
    layout = policy.PhaseLayout('phase4b',root=tmp_path,approvals_base=tmp_path/'approvals')
    assert layout.studies_db == tmp_path/'output/optimize/phase4b/studies.db'
    assert layout.manifest == tmp_path/'output/phase4b/campaign.json'
    assert layout.approval_root == tmp_path/'approvals/phase4b'
    for rel in ('output/phase4b/a','output/optimize/phase4b/a','output/duckdb_tmp/a'):
        assert layout.checked_write(rel) == tmp_path/rel
    for rel in ('output/phase4/a','output/optimize/phase4/studies.db','output/optimize/trade_cache/a',
                'output/optimize/phase4b/../phase4/a'):
        with pytest.raises(ValueError):layout.checked_write(rel)
    with pytest.raises(ValueError):layout.db_path(tmp_path/'output/optimize/phase4b/other.db')
    old = policy.PhaseLayout('phase4',root=tmp_path)
    assert old.write_roots == policy.WRITE_ROOTS
    assert old.db_path(tmp_path/'output/optimize/phase4/other.db').name == 'other.db'
    (tmp_path/'output/phase4b').mkdir(parents=True)
    (tmp_path/'output/phase4b/link').symlink_to(tmp_path/'output/phase4',target_is_directory=True)
    with pytest.raises(ValueError):layout.checked_write('output/phase4b/link/a')


def test_c7_derive_ceilings_formula_and_invalid_inputs():
    benchmark={'lanes':{bt:benchmark_lane() for bt in intervals()}}
    result = campaign.derive_ceilings(benchmark)
    assert result['arms']['fixed_tick_imbalance'] == {'max_trial_seconds':600,'max_study_seconds':75900}
    for values in ([],[0,1,2],[float('nan'),1,2],[True,1,2]):
        benchmark['lanes']['fixed_tick_imbalance']['trial_equivalent_seconds']=values
        with pytest.raises(ValueError):campaign.derive_ceilings(benchmark)


def test_c7_two_sealed_registrations_and_tampered_benchmark(tmp_path,monkeypatch):
    layout,snapshots,comparators,origin = fixture_inputs(tmp_path,monkeypatch)
    monkeypatch.setattr(campaign,'source_state',lambda:{'base':'base','sha256':'source','dirty':False})
    s1layout = policy.PhaseLayout('phase4b-s1',root=tmp_path,approvals_base=tmp_path/'approvals')
    before = {str(p):sha(p) for p in (tmp_path/'output/phase4').rglob('*') if p.is_file()}
    s1 = campaign.register(s1layout.manifest,snapshots,phase='phase4b-s1',layout=s1layout,
                           comparators=comparators,phase4_origin=origin)
    assert s1['stages'] == ['S1'] and s1['max_stage_seconds'] == {'S1':14400}
    assert all(arm['max_trial_seconds']==1800 for arm in s1['arms'].values())
    benchmark = layout.output/'s1/benchmark.json'
    from scripts.phase4b_runtime import runtime_fingerprint
    runtime=runtime_fingerprint()
    benchmark.write_bytes(policy.canonical_bytes({'campaign_sha256':sha(s1layout.manifest),'runtime_fingerprint':runtime,
        'lanes':{bt:benchmark_lane() for bt in comparators}}))
    Path(str(benchmark)+'.sha256').write_text(sha(benchmark)+'\n')
    checkpoint=layout.checkpoints/sha(s1layout.manifest)/'S1.json'
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    checkpoint.write_bytes(policy.canonical_bytes({'phase':'phase4b-s1','stage':'S1','status':'complete',
        'campaign_sha256':sha(s1layout.manifest),'source_state':s1['source_state'],'runtime_fingerprint':runtime,
        'result':json.loads(benchmark.read_bytes())}))
    m = campaign.register(layout.manifest,snapshots,phase='phase4b',layout=layout,comparators=comparators,
                          phase4_origin=origin,benchmark=benchmark,holdout_intervals=intervals())
    assert m['stages'] == ['S2t','S2v','S3'] and m['phase4b_db'] == str(layout.studies_db)
    assert all(a['budget']==300 and a['max_trial_seconds']==600 for a in m['arms'].values())
    assert m['benchmark_sha256'] == sha(benchmark)
    assert {str(p):sha(p) for p in (tmp_path/'output/phase4').rglob('*') if p.is_file()} == before
    with pytest.raises(ValueError,match='limit'):
        campaign.register(layout.manifest,snapshots,phase='phase4b',layout=layout,
            comparators=comparators,phase4_origin=origin,benchmark=benchmark,
            trial_limits=[(next(iter(m['arms'])), 999)],holdout_intervals=intervals())
    benchmark.write_text('{}')
    with pytest.raises(ValueError,match='benchmark'):
        campaign.register(layout.manifest,snapshots,phase='phase4b',layout=layout,
            comparators=comparators,phase4_origin=origin,benchmark=benchmark,holdout_intervals=intervals())


@pytest.mark.parametrize('state', ['absent','other_campaign','reserved','complete','missing_origin','tampered_origin','recovery'])
def test_c8b_allocation_refusals_before_open_and_on_recovery(tmp_path,monkeypatch,state):
    layout,_,_,origin = fixture_inputs(tmp_path,monkeypatch)
    bt = 'fixed_tick_imbalance'
    interval = {'dataset_id':'synthetic','symbol':'BTCUSDT','bar_type':bt,'start_seq':81,'end_seq':100}
    digest = 'b'*64
    completed_arm(layout,digest)
    manifest = {'phase':'phase4b','phase4_origin':origin}
    if state != 'absent':
        approvals.install_holdout_allocation(layout,interval,
            'a'*64 if state=='other_campaign' else digest,origin)
    if state in ('reserved','complete','recovery'):
        path = Path(origin['exposure_path']);path.parent.mkdir(parents=True)
        path.write_text(json.dumps({f'BTCUSDT:{bt}:off':{'status':state}}))
    if state == 'missing_origin':Path(origin['manifest_path']).unlink()
    if state == 'tampered_origin':Path(origin['manifest_path']).write_text('{}')
    if state == 'recovery':
        layout.exposure.parent.mkdir(parents=True,exist_ok=True)
        layout.exposure.write_text(json.dumps({f'BTCUSDT:{bt}:full':{'candidate':{'replay_digest':'d'},'status':'reserved'}}))
    with pytest.raises(ValueError,match='allocation|origin|exposure'):
        p4.phase4b_reserved_look(layout,manifest,digest,interval,{'replay_digest':'d'},
            {'G4b_N':True,'gates':{str(i):True for i in range(1,11)},'replay_digest':'d',
             'campaign_sha256':digest,'bar_type':bt},
            lambda:pytest.fail('opened holdout before admission'))


def test_c8b_allocation_content_hash_permissions_exclusive_and_pass(tmp_path,monkeypatch):
    layout,_,_,origin = fixture_inputs(tmp_path,monkeypatch)
    completed_arm(layout,'b'*64)
    interval = {'dataset_id':'synthetic','symbol':'BTCUSDT','bar_type':'fixed_tick_imbalance','start_seq':81,'end_seq':100}
    path = approvals.install_holdout_allocation(layout,interval,'b'*64,origin)
    assert path.stat().st_mode & 0o777 == 0o600
    assert 'holdout-allocation' in str(path) and not path.is_relative_to(layout.output)
    with pytest.raises(FileExistsError):approvals.install_holdout_allocation(layout,interval,'b'*64,origin)
    calls=[]
    args=(layout,{'phase':'phase4b','phase4_origin':origin},'b'*64,interval,{'replay_digest':'d'},
          {'G4b_N':True,'gates':{str(i):True for i in range(1,11)},'replay_digest':'d',
           'campaign_sha256':'b'*64,'bar_type':'fixed_tick_imbalance'})
    first=p4.phase4b_reserved_look(*args,lambda:(calls.append(1) or {'G3':True}))
    assert p4.phase4b_reserved_look(*args,lambda:pytest.fail('second look'))==first
    assert calls==[1]
    raw=json.loads(path.read_text());raw['allocation']['campaign_sha256']='c'*64;path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='allocation'):
        p4.phase4b_reserved_look(*args,lambda:pytest.fail('tampered allocation'))


@pytest.mark.parametrize('phase,stage',[('phase4b-s1','S1'),('phase4b','S2t'),('phase4b','S2v'),('phase4b','S3')])
def test_c7_stage_approval_phase_argv_and_single_use(tmp_path,monkeypatch,phase,stage):
    layout=policy.PhaseLayout(phase,root=tmp_path,approvals_base=tmp_path/'approvals')
    state={'base':'base','sha256':'source','dirty':False}
    monkeypatch.setattr(approvals,'source_state',lambda:state)
    layout.approval_root.mkdir(parents=True)
    argv=['python','driver',stage]
    record={'scope':'campaign','phase':phase,'stage':stage,'plan_sha256':approvals.PHASE4B_PLAN_SHA256,
        'campaign_sha256':'c'*64,'source_state':state,'base_commit':'base','command_argv':argv,
        'created_utc':'2026-11-07T00:00:00Z'}
    raw=policy.canonical_bytes(record);selector=hashlib.sha256(raw).hexdigest()
    path=layout.approval_root/(selector+'.json');path.write_bytes(raw)
    verified=approvals.verify_phase4b_approval(layout,selector,'c'*64,stage,argv)
    with pytest.raises(ValueError,match='stage|phase|argv'):
        approvals.verify_phase4b_approval(layout,selector,'c'*64,'S2v' if stage!='S2v' else 'S2t',argv)
    with pytest.raises(ValueError,match='argv'):
        approvals.verify_phase4b_approval(layout,selector,'c'*64,stage,['different'])
    approvals.consume_approval(verified,layout=layout)
    with pytest.raises(ValueError,match='consumed'):
        approvals.consume_approval(verified,layout=layout)


def test_c7_phase4_registration_bytes_match_prebuild(tmp_path,monkeypatch):
    import ast
    import subprocess
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    monkeypatch.setattr(policy,'ROOT',tmp_path)
    monkeypatch.setattr(campaign,'source_state',lambda:{'base':'base','sha256':'source','dirty':False})
    snapshots=[]
    for i in range(3):
        path=tmp_path/f'legacy{i}.db';path.write_bytes(b'synthetic')
        snapshots.append({'path':str(path),'sha256':sha(path)})
    tree=ast.parse(subprocess.check_output(['git','show','95bde47:scripts/phase4_campaign.py'],text=True))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='register')
    namespace=dict(vars(campaign));exec(compile(ast.Module(body=[fn],type_ignores=[]),'<prebuild register>','exec'),namespace)
    path=tmp_path/'output/phase4/campaign.json'
    namespace['register'](path,snapshots)
    expected=path.read_bytes();seal=Path(str(path)+'.sha256').read_bytes()
    path.unlink();Path(str(path)+'.sha256').unlink()
    campaign.register(path,snapshots,phase='phase4')
    assert path.read_bytes()==expected and Path(str(path)+'.sha256').read_bytes()==seal


def test_c8b_gate_cannot_authorize_another_candidate(tmp_path,monkeypatch):
    layout,_,_,origin=fixture_inputs(tmp_path,monkeypatch)
    completed_arm(layout,'b'*64)
    interval={'dataset_id':'synthetic','symbol':'BTCUSDT','bar_type':'fixed_tick_imbalance','start_seq':81,'end_seq':100}
    approvals.install_holdout_allocation(layout,interval,'b'*64,origin)
    report={'G4b_N':True,'gates':{str(i):True for i in range(1,11)},'replay_digest':'different',
            'campaign_sha256':'b'*64,'bar_type':interval['bar_type']}
    with pytest.raises(ValueError,match='candidate|gate'):
        p4.phase4b_reserved_look(layout,{'phase':'phase4b','phase4_origin':origin},'b'*64,interval,
            {'replay_digest':'expected'},report,lambda:pytest.fail('unbound gate authorized look'))


def test_c8b_broken_origin_exposure_symlink_is_unavailable(tmp_path,monkeypatch):
    _,_,_,origin=fixture_inputs(tmp_path,monkeypatch)
    path=Path(origin['exposure_path']);path.parent.mkdir(parents=True)
    path.symlink_to(tmp_path/'missing')
    assert approvals.phase4_exposure_state(origin,'fixed_tick_imbalance')=='unavailable'


def test_c7_legacy_optimizer_cannot_open_phase4b_study(monkeypatch):
    from types import SimpleNamespace
    from scripts import optimize_orderflow as opt
    monkeypatch.setattr(opt,'_stored_identity',lambda *a:{'phase':'phase4b'})
    with pytest.raises(ValueError,match='legacy'):
        p4.legacy_study_guard(SimpleNamespace(phase4=False,study_db='synthetic'),'study')


def test_c7_registration_rebases_all_historical_source_paths(tmp_path,monkeypatch):
    layout,snapshots,comparators,origin=fixture_inputs(tmp_path,monkeypatch)
    layout=policy.PhaseLayout('phase4b-s1',root=tmp_path,approvals_base=tmp_path/'approvals')
    monkeypatch.setattr(campaign,'source_state',lambda:{'base':'base','sha256':'source','dirty':False})
    original_snapshots=[{**s,'path':str(Path('/original/checkout')/Path(s['path']).relative_to(tmp_path))} for s in snapshots]
    original_comparators={bt:{**s,'path':str(Path('/original/checkout')/Path(s['path']).relative_to(tmp_path))}
        for bt,s in comparators.items()}
    result=campaign.register(layout.manifest,original_snapshots,phase=layout.phase,layout=layout,
        comparators=original_comparators,phase4_origin=origin)
    assert result['snapshots']==snapshots
    assert all(Path(arm['comparator']['path']).is_relative_to(tmp_path) for arm in result['arms'].values())


def test_c7_full_registration_requires_completed_s1_checkpoint(tmp_path,monkeypatch):
    layout,snapshots,comparators,origin=fixture_inputs(tmp_path,monkeypatch)
    monkeypatch.setattr(campaign,'source_state',lambda:{'base':'base','sha256':'source','dirty':False})
    s1=policy.PhaseLayout('phase4b-s1',root=tmp_path,approvals_base=tmp_path/'approvals')
    campaign.register(s1.manifest,snapshots,phase=s1.phase,layout=s1,comparators=comparators,phase4_origin=origin)
    benchmark=layout.output/'s1/benchmark.json'
    benchmark.write_bytes(policy.canonical_bytes({'campaign_sha256':sha(s1.manifest),
        'lanes':{bt:benchmark_lane() for bt in comparators}}))
    Path(str(benchmark)+'.sha256').write_text(sha(benchmark)+'\n')
    with pytest.raises(ValueError,match='checkpoint'):
        campaign.register(layout.manifest,snapshots,phase='phase4b',layout=layout,comparators=comparators,
            phase4_origin=origin,benchmark=benchmark,holdout_intervals=intervals())
