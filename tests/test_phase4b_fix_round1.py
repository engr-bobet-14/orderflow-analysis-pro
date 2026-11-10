"""Rev 11 R1–R6 regressions. Fixtures and writes are synthetic and temporary."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from orderflow_system.analytics.footprint import aggregate_levels, diagonal_imbalances, stacked_runs
from scripts import phase4_studies as studies, phase4_report as report


def probe_params():
    return {**studies.full_strategy_defaults(),
        'ts_stacked_min_levels':1,'ts_imbalance_bucket_mult':1,'ts_imbalance_min_level_share':.002,
        'ts_imbalance_ratio':2.,'ts_min_level_share':.02,'ts_min_side_ratio':.55,
        'ts_min_reject_bps':.5,'ts_max_through_bps':4.,'ts_level_zone_frac':.50,
        'ts_contrarian_ratio':1.5,'ts_contrarian_zone_frac':.50,'ts_contrarian_min_level_share':.005,
        'signal_cooldown_seconds':5.,'min_composite_score':10.,'price_proximity_pct':.005}


def lane_fixture(probe=400.):
    return {'replays':{name:{'runtime_seconds':200.} for name in (
        'defaults','initiative_off','k1','E2_off','P2_bar_vwap','stageB_bar_params')},
        'trial_equivalent_seconds':[200.,200.,200.],
        'worst_case_probe':{'params':probe_params(),'runtime_seconds':probe}}


def benchmark_fixture():
    return {'lanes':{'fixed_tick_imbalance':lane_fixture(), 'fixed_volume_imbalance':lane_fixture(500.)}}


@pytest.mark.parametrize('levels,expected', [
    ({0:(10,0),2:(0,40)},[(1,'sell'),(2,'buy')]),
    ({10:(0,40),11:(20,0)},[]),  # no pair across either edge
    ({0:(0,1),3:(1,0)},[]),     # each inside pair is both-zero
    ({5:(10,40)},[]),
    ({},[]),
])
def test_r3_complete_range_diagonal_oracle(levels,expected):
    assert diagonal_imbalances(levels,3.,.01,100)==expected


def test_r3_gap_prints_extend_and_zero_pairs_break_stacks():
    prints=diagonal_imbalances({0:(50,0),1:(50,0),3:(10,0)},3.,.01,110)
    assert prints==[(1,'sell'),(2,'sell')]
    assert stacked_runs(prints,1)=={'buy':0,'sell':2}
    broken=diagonal_imbalances({0:(50,0),2:(20,0),3:(10,0)},3.,.01,80)
    assert broken==[(1,'sell'),(3,'sell')]
    assert stacked_runs(broken,1)['sell']==1
    buys=diagonal_imbalances({0:(0,0),2:(0,20),3:(0,30)},3.,.01,50)
    assert buys==[(2,'buy'),(3,'buy')]
    assert stacked_runs(buys,1)['buy']==2


def test_r3_aggregate_before_gap_detection_and_keep_min_share():
    levels=aggregate_levels(([0,100,400,600],[1,0,40,40],[30,20,0,0]),2)
    assert levels=={0:(50,1),2:(0,40),3:(0,40)}
    prints=diagonal_imbalances(levels,3.,.01,131)
    assert prints==[(1,'sell'),(2,'buy'),(3,'buy')]
    assert stacked_runs(prints,1)=={'buy':2,'sell':1}
    assert diagonal_imbalances({0:(1,0),2:(0,1)},3.,.02,100)==[]


@pytest.mark.parametrize('overrides', [
    {'initiative_imbalance':'bonus'}, {'initiative_imbalance':'required'},
    {'exhaustion_contrarian':'bonus'}, {'absorption_source':'bar'},
    {'imbalance_ratio':2.}, {'framing_profile_source':'trades'},
    {'initiative_imbalance':'off','exhaustion_contrarian':'off'},
])
def test_r4_refuse_unregistered_final_signal_overrides(overrides,monkeypatch):
    monkeypatch.setattr(studies,'_typed_replay',lambda *a:pytest.fail('invalid config reached replay'))
    with pytest.raises(ValueError,match='override|bonus|phase4b'):
        studies.replay_full_strategy(studies.FullStrategySample(studies.full_strategy_defaults()),None,
                                     signal_overrides=overrides)


def test_r4_effective_config_validates_after_ts_mapping():
    with pytest.raises(ValueError,match='bonus'):
        studies.effective_trade_config(SimpleNamespace(phase='phase4b'),{'ts_initiative_imbalance':'bonus'})


@pytest.mark.parametrize('overrides',[{'initiative_imbalance':'off'},{'exhaustion_contrarian':'off'},
                                     {'framing_profile_source':'bar_vwap'},{}])
def test_r4_registered_s1_overrides_remain_available(overrides,monkeypatch):
    configs=[]
    monkeypatch.setattr(studies,'_typed_replay',lambda value,kind,prefix,params,config,*a:
        (configs.append(config.signals) or {'replay_digest':'synthetic'}))
    studies.replay_full_strategy(studies.FullStrategySample(studies.full_strategy_defaults()),None,
                                 signal_overrides=overrides)
    assert configs[0].initiative_imbalance!='bonus'
    assert all(getattr(configs[0],name)==value for name,value in overrides.items())


def test_r6_stressed_quantity_digest_and_length_replace_vector(monkeypatch):
    from tests.test_ledger import replay
    from scripts import optimize_orderflow as opt, diagnose_trades as diagnose
    ledger,_=replay()
    expected=opt.simulate_trades_risk_based(ledger,fee_bps=5.,slippage_bps=2.)
    monkeypatch.setattr(diagnose,'scored_only',lambda frame,*a:frame)
    result=report.slippage_resimulation(ledger,[],[])
    assert 'quantities' not in result
    assert result['quantities_length']==len(expected)
    assert result['quantities_sha256']==hashlib.sha256(expected.quantity.to_numpy(dtype='<f8').tobytes()).hexdigest()
    assert result['net_expectancy_r']==float((expected.net_pnl/expected.dollar_risk).mean())


def test_r1_r2_all_ten_timings_drive_limits_and_s3_replay_budget():
    from scripts import phase4_campaign as campaign
    result=campaign.derive_ceilings(benchmark_fixture())
    tick=result['arms']['fixed_tick_imbalance']
    assert tick=={'max_trial_seconds':600,'max_study_seconds':75900}
    assert result['arms']['fixed_volume_imbalance']['max_trial_seconds']==750
    assert result['max_stage_seconds']['S2t']==tick['max_study_seconds']+1200
    assert result['max_stage_seconds']['S3']==67825
    assert result['max_stage_seconds']['S3'] >= 1.3*500*(3+1+46+1/.8)*2+1200
    assert result['inputs']['M']==500. and result['inputs']['N']==300
    assert len(result['inputs']['timings']['fixed_tick_imbalance'])==10


@pytest.mark.parametrize('missing',range(10))
def test_r2_missing_any_timing_or_probe_is_refused(missing):
    from scripts import phase4_campaign as campaign
    bench=benchmark_fixture();lane=bench['lanes']['fixed_tick_imbalance']
    if missing<6:del lane['replays'][list(lane['replays'])[missing]]
    elif missing<9:lane['trial_equivalent_seconds'].pop(missing-6)
    else:del lane['worst_case_probe']
    with pytest.raises(ValueError,match='timing|probe|benchmark'):
        campaign.derive_ceilings(bench)


def test_r2_s1_runs_registered_probe_and_collects_ten_timings(monkeypatch):
    from tests.test_pipeline import synth_bars
    from tests.test_optimize_orderflow import params
    from scripts import optimize_orderflow as opt, diagnose_trades as diagnose
    prefix=studies.SearchPrefix(synth_bars(30),[None]*30,24)
    benchmarks=[]
    monkeypatch.setattr(studies,'replay_full_strategy',lambda *a,**kw:{'ledger':None,'signal_counts':{},'replay_digest':'d'})
    monkeypatch.setattr(studies,'benchmark_trial_equivalent',lambda prefix,values,**kw:benchmarks.append(dict(values)))
    monkeypatch.setattr(opt,'simulate_trades_risk_based',lambda *a,**kw:None)
    monkeypatch.setattr(diagnose,'add_gross',lambda value:value)
    monkeypatch.setattr(diagnose,'decompose',lambda value:{})
    result=studies.run_s1(prefix,studies.FrozenComparator(params(),'d'))
    assert benchmarks==[studies.full_strategy_defaults()]*3+[probe_params()]
    assert result['worst_case_probe']['params']==probe_params()
    assert len(studies.s1_timings(result))==10


def test_r5_phase4b_main_definitions_are_available_at_dispatch(tmp_path,monkeypatch,capsys):
    import inspect
    import runpy
    import sys
    from scripts import phase4b_campaign as driver, phase4_policy as policy
    from tests.test_phase4b_operations import fixture_inputs
    _,snapshots,comparators,origin=fixture_inputs(tmp_path,monkeypatch)
    layout=policy.PhaseLayout('phase4b-s1',root=tmp_path,approvals_base=tmp_path/'approvals')
    def dispatch(argv):
        namespace=inspect.currentframe().f_back.f_globals
        return namespace['register'](layout.manifest,snapshots,phase=layout.phase,layout=layout,
            comparators=comparators,phase4_origin=origin)
    monkeypatch.setattr(driver,'main',dispatch)
    monkeypatch.setattr(sys,'argv',['phase4_campaign.py','register','--phase','phase4b-s1'])
    namespace=runpy.run_path('scripts/phase4_campaign.py',run_name='__main__')
    assert namespace['__name__']=='__main__'
    assert json.loads(capsys.readouterr().out)['phase']=='phase4b-s1'


@pytest.mark.parametrize('stage',['S1','S2t','S2v','S3'])
def test_r1_stage_deadline_comes_only_from_manifest(stage):
    from scripts import phase4b_campaign as driver
    manifest={'max_stage_seconds':{'S1':14400,'S2t':75900,'S2v':79350,'S3':67825}}
    assert driver.stage_ceiling(manifest,stage)==manifest['max_stage_seconds'][stage]
    with pytest.raises(ValueError,match='registered'):
        driver.stage_ceiling({'max_stage_seconds':{}},stage)


def test_r1_stage_paths_have_no_previous_literal_deadlines():
    import ast
    import inspect
    from scripts import phase4b_campaign as driver
    for function in (driver.run_stage,driver.main):
        assert not any(isinstance(node,ast.Constant) and node.value in (3600,10800)
                       for node in ast.walk(ast.parse(inspect.getsource(function))))


def runtime_fixture(tmp_path):
    from scripts.phase4_policy import PhaseLayout
    from scripts.phase4_campaign import derive_ceilings
    layout=PhaseLayout('phase4b',root=tmp_path,approvals_base=tmp_path/'approvals')
    limits=derive_ceilings(benchmark_fixture())
    manifest={'phase':'phase4b','stages':['S2t','S2v','S3'], 'max_stage_seconds':limits['max_stage_seconds'],
        'runtime_ceilings':limits,'arms':{f'BTCUSDT:{bt}:full':values for bt,values in limits['arms'].items()}}
    return layout,manifest,'a'*64


@pytest.mark.parametrize('stage',['S2t','S3'])
def test_r2_live_breach_audits_before_process_exit(tmp_path,stage):
    import subprocess
    import sys
    layout,manifest,digest=runtime_fixture(tmp_path)
    from scripts import phase4_policy as policy
    fixture=tmp_path/'runtime.json';fixture.write_bytes(policy.canonical_bytes(manifest))
    script=r'''
import json,sys,time
from pathlib import Path
from scripts import phase4b_campaign as driver,optimize_orderflow as opt,phase4_resources as resources
from scripts.phase4_policy import PhaseLayout
root,stage=sys.argv[1:]
layout=PhaseLayout('phase4b',root=Path(root),approvals_base=Path(root)/'approvals')
manifest=json.loads((Path(root)/'runtime.json').read_text());digest='a'*64
resources.current_memory=lambda:{'rss_bytes':100,'peak_rss_bytes':100,'private_bytes':100,
    'peak_private_bytes':100,'private_is_fallback':False}
opt.PHASE4_HEARTBEAT_SECONDS=.02
driver.stage_progress(layout,digest,stage,arm='BTCUSDT:fixed_tick_imbalance:full',trial=7,marker='neighbour 17/46')
def audit(stop):
    driver.record_termination(layout,manifest,digest,stage,stop={**stop,'trial':7,
        'reason':'max_stage_seconds' if stage=='S3' else 'max_trial_seconds'})
with driver.file_timing(layout,digest,stage,.03,audit=audit):time.sleep(2)
raise AssertionError('watchdog did not exit')
'''
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path),stage],capture_output=True,text=True,timeout=10)
    assert result.returncode==70,result.stderr
    from scripts import phase4b_campaign as driver
    audit=driver.recover_termination_audit(layout,digest,stage)
    assert audit['kind']=='timeout_confirmed'  # rev 13: observed in-process evidence at/above its ceiling.
    assert audit['stage']==stage and audit['trial']==7
    assert audit['arm']=='BTCUSDT:fixed_tick_imbalance:full'
    assert audit['duration_seconds']>.03 and audit['ceiling_seconds']==.03
    assert audit['last_progress_marker']=='neighbour 17/46' and audit['utc'].endswith('+00:00')
    assert audit['ceiling_inputs']==manifest['runtime_ceilings']['inputs']
    events=[json.loads(line) for line in (layout.output/'campaign-log.jsonl').read_text().splitlines()]
    assert any(event.get('event')=='termination_audit' for event in events)


@pytest.mark.parametrize('stage',['S2t','S3'])
def test_r2_failed_append_recovers_and_consumes_stage_for_every_approval(tmp_path,monkeypatch,stage):
    from scripts import phase4b_campaign as driver,phase4_campaign as campaign
    layout,manifest,digest=runtime_fixture(tmp_path)
    driver.stage_progress(layout,digest,stage,arm='BTCUSDT:fixed_tick_imbalance:full',trial=12,marker='trial 12')
    append=campaign.append_log
    monkeypatch.setattr(campaign,'append_log',lambda *a:(_ for _ in ()).throw(OSError('injected append failure')))
    stop={'reason':'max_stage_seconds','measured_seconds':80000.,'limit_seconds':67825}
    with pytest.raises(OSError,match='append failure'):driver.record_termination(layout,manifest,digest,stage,stop=stop)
    assert list((layout.checkpoints/digest/'terminations'/stage).glob('*.pending.json'))
    monkeypatch.setattr(campaign,'append_log',append)
    record=driver.recover_termination_audit(layout,digest,stage)
    assert record['kind']=='timeout_confirmed'
    assert record['trial']==12 and record['last_progress_marker']=='trial 12'
    assert driver.recover_termination_audit(layout,digest,stage)==record
    events=[json.loads(line) for line in (layout.output/'campaign-log.jsonl').read_text().splitlines()]
    assert len(events)==1
    monkeypatch.setattr(driver,'read_manifest',lambda layout:(manifest,digest))
    monkeypatch.setattr(driver,'require_phase4b_approval',lambda *a,**kw:pytest.fail('consumed stage reached approval'))
    with pytest.raises(ValueError,match='consumed|breach|new registration'):
        driver.run_stage(layout,stage,'different-owner-approval',operation=lambda *a:pytest.fail('resumed'))


def test_r1_external_timeout_uses_registered_s3_limit_and_audits(tmp_path,monkeypatch):
    from scripts import phase4b_campaign as driver,phase4_approvals as approvals,measure_rss,phase4_resources
    layout,manifest,digest=runtime_fixture(tmp_path)
    layout.approval_root.mkdir(parents=True);(layout.approval_root/('b'*64+'.json')).write_text('{}')
    monkeypatch.setattr(driver,'PhaseLayout',lambda phase:layout)
    monkeypatch.setattr(driver,'read_manifest',lambda layout:(manifest,digest))
    monkeypatch.setattr(driver,'require_phase4b_approval',lambda *a,**kw:{})
    monkeypatch.setattr(approvals,'verify_phase4b_approval',lambda *a,**kw:{})
    monkeypatch.setattr(driver.shutil,'which',lambda name:'/synthetic/timeout')
    from contextlib import contextmanager
    @contextmanager
    def locks(layout,manifest):yield {}
    monkeypatch.setattr(driver,'stage_locks',locks)
    clock={'now':1000.}
    monkeypatch.setattr(driver,'_now',lambda:clock['now'])
    calls=[]
    def measured(argv,**kw):
        calls.append(argv);clock['now']+=67826.  # Observed elapsed from launch L.
        return {'returncode':124,'wall_seconds':67826.}
    monkeypatch.setattr(measure_rss,'run_measured',measured)
    monkeypatch.setattr(phase4_resources,'enforce_memory',lambda *a:None)
    with pytest.raises(SystemExit):driver.main(['run','--phase','phase4b','--stage','S3','--owner-approved','c'*64])
    assert calls[0][1]=='--foreground' and calls[0][4]=='67825'
    record=driver.recover_termination_audit(layout,digest,'S3')
    assert record['kind']=='timeout_confirmed' and record['observed_exit_code']==124


def test_r1_live_stage_watchdog_uses_registered_s3_limit(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from scripts import phase4b_campaign as driver,phase4_policy as policy
    layout,manifest,digest=runtime_fixture(tmp_path)
    manifest['snapshots']=[]
    monkeypatch.setattr(driver,'read_manifest',lambda layout:(manifest,digest))
    monkeypatch.setattr(driver,'source_state',lambda:{'base':'test'})
    monkeypatch.setattr(driver,'require_phase4b_approval',lambda *a,**kw:{'sha256':'b'*64,'gates_sha256':'g'})
    monkeypatch.setattr(driver,'preflight',lambda *a:None)
    monkeypatch.setattr(driver,'protected_hashes',lambda *a:{})
    from scripts.phase4b_runtime import runtime_fingerprint
    manifest['runtime_fingerprint']=runtime_fingerprint()
    for previous in ('S2t','S2v'):
        path=layout.checkpoints/digest/f'{previous}.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(policy.canonical_bytes({'status':'complete'}))
    calls=[];watched=[]
    @contextmanager
    def timing(layout,digest,label,ceiling,**kw):
        calls.append(ceiling);yield
    @contextmanager
    def watchdog(layout,manifest,digest,stage):
        watched.append(driver.stage_ceiling(manifest,stage));yield
    monkeypatch.setattr(driver,'file_timing',timing)
    monkeypatch.setattr(driver,'stage_watchdog',watchdog)
    monkeypatch.setattr(driver,'execute_stage',lambda *a:{'looks':{}})
    monkeypatch.setenv(driver.LAUNCH_ENV,json.dumps({'utc':'u','monotonic':driver._now(),'attempt':'production'}))
    monkeypatch.setattr(driver,'require_inherited_locks',lambda *a:None)
    result=driver.run_stage(layout,'S3','b'*64,finalize=False)
    # rev 13: the stage ceiling runs from launch L in stage_watchdog; file_timing keeps heartbeat/memory only.
    assert result['status']=='pending_postflight' and watched==[67825] and calls==[None]


def test_r2_torn_audit_append_is_reconstructed_once(tmp_path):
    from scripts import phase4b_campaign as driver
    layout,manifest,digest=runtime_fixture(tmp_path)
    driver.record_termination(layout,manifest,digest,'S3',stop={'reason':'max_stage_seconds',
        'measured_seconds':67826.,'limit_seconds':67825})
    log=layout.output/'campaign-log.jsonl'
    log.write_bytes(b'{"event":"termination_aud')
    driver.recover_termination_audit(layout,digest,'S3')
    driver.recover_termination_audit(layout,digest,'S3')
    assert len([json.loads(line) for line in log.read_text().splitlines()])==1


def test_r2_invalid_or_modified_probe_is_refused():
    from scripts import phase4_campaign as campaign
    benchmark=benchmark_fixture();benchmark['lanes']['fixed_tick_imbalance']['worst_case_probe']['params']['ts_imbalance_ratio']=3.
    with pytest.raises(ValueError,match='probe'):campaign.derive_ceilings(benchmark)


def test_r1_hand_supplied_stage_limit_must_equal_formula(tmp_path,monkeypatch):
    from scripts import phase4_campaign as campaign,phase4_policy as policy
    from tests.test_phase4b_operations import fixture_inputs,intervals,sha
    layout,snapshots,comparators,origin=fixture_inputs(tmp_path,monkeypatch)
    monkeypatch.setattr(campaign,'source_state',lambda:{'base':'base','sha256':'source','dirty':False})
    s1=policy.PhaseLayout('phase4b-s1',root=tmp_path,approvals_base=tmp_path/'approvals')
    first=campaign.register(s1.manifest,snapshots,phase=s1.phase,layout=s1,comparators=comparators,phase4_origin=origin)
    from scripts.phase4b_runtime import runtime_fingerprint
    runtime=runtime_fingerprint()
    benchmark=benchmark_fixture();benchmark.update(campaign_sha256=sha(s1.manifest),runtime_fingerprint=runtime)
    path=layout.output/'s1/benchmark.json';path.write_bytes(policy.canonical_bytes(benchmark))
    path.with_suffix('.json.sha256').write_text(sha(path)+'\n')
    checkpoint=layout.checkpoints/sha(s1.manifest)/'S1.json';checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(policy.canonical_bytes({'status':'complete','stage':'S1','phase':s1.phase,
        'campaign_sha256':sha(s1.manifest),'source_state':first['source_state'],
        'runtime_fingerprint':runtime,'result':benchmark}))
    with pytest.raises(ValueError,match='limit'):
        campaign.register(layout.manifest,snapshots,phase=layout.phase,layout=layout,comparators=comparators,
            phase4_origin=origin,benchmark=path,holdout_intervals=intervals(),stage_limits=[('S3',1)])


def test_r2_approvals_bind_the_current_contract_and_preserve_rev11():
    from scripts import phase4_approvals as approvals
    from pathlib import Path
    expected='75b22452c0b168f43fd91704a7aee23a4cb1c84c637f4a8f775b87be13857b7f'  # rev 13
    assert approvals.PHASE4B_PLAN_SHA256==expected
    assert hashlib.sha256(approvals.PHASE4B_PLAN_PATH.read_bytes()).hexdigest()==expected
    assert hashlib.sha256(Path('docs/plans/phase4b-full-strategy-plan-rev12.md').read_bytes()).hexdigest()==(
        'dbb407fb611720e42d70ac6c051d82077350247ef83fb433f13e9d97d8849099')
    assert hashlib.sha256(Path('docs/plans/phase4b-full-strategy-plan-rev11.md').read_bytes()).hexdigest()==(
        '2c9cf6a80c912500552241a3cd4d223dbc5cc8e3847e0ef86c39862f6e457af7')
    assert hashlib.sha256(Path('docs/plans/phase4b-full-strategy-plan.md').read_bytes()).hexdigest()==(
        '89f64efe62090c271045d33b4ef9cbabb257f3ca5d782aa9032144d322f029c7')


def test_r2_concurrent_watchdogs_wait_for_complete_writeahead_record(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    from scripts import phase4b_campaign as driver,phase4_policy as policy
    layout,manifest,digest=runtime_fixture(tmp_path)
    original=driver.write_json;writing=threading.Event();release=threading.Event()
    def delayed(layout,path,value,*,exclusive=False):
        if str(path).endswith('.pending.json') and 'terminations' in str(path):
            path=layout.checked_write(path);path.parent.mkdir(parents=True,exist_ok=True)
            raw=policy.canonical_bytes(value)
            with path.open('xb') as handle:
                handle.write(raw[:1]);handle.flush();writing.set();release.wait(3)
                handle.write(raw[1:]);handle.flush()
            return path
        return original(layout,path,value,exclusive=exclusive)
    monkeypatch.setattr(driver,'write_json',delayed)
    stop={'reason':'max_stage_seconds','measured_seconds':67826.,'limit_seconds':67825}
    with ThreadPoolExecutor(2) as pool:
        first=pool.submit(driver.record_termination,layout,manifest,digest,'S3',stop=stop)
        assert writing.wait(2)
        second=pool.submit(driver.record_termination,layout,manifest,digest,'S3',stop=stop)
        try:
            time.sleep(.05)
            assert not second.done(),'second watchdog read an incomplete marker'
        finally:release.set()
        assert first.result(timeout=3)==second.result(timeout=3)


def test_r1_registration_json_forwards_every_hand_limit(tmp_path,monkeypatch):
    from scripts import phase4b_campaign as driver,phase4_campaign as campaign,phase4_policy as policy
    layout=policy.PhaseLayout('phase4b',root=tmp_path,approvals_base=tmp_path/'approvals')
    values={'snapshots':[],'comparators':{},'phase4_origin':{},'holdout_intervals':{},
        'trial_limits':[['BTCUSDT:fixed_tick_imbalance:full',1]],
        'study_limits':[['BTCUSDT:fixed_tick_imbalance:full',2]],'stage_limits':[['S3',3]]}
    path=tmp_path/'registration.json';path.write_text(json.dumps(values))
    monkeypatch.setattr(driver,'PhaseLayout',lambda phase:layout)
    monkeypatch.setattr(campaign,'register',lambda *a,**kw:kw)
    result=driver.main(['register','--phase','phase4b','--registration',str(path)])
    assert all(result[name]==values[name] for name in ('trial_limits','study_limits','stage_limits'))


def test_r2_reconciled_ceiling_stop_is_audited_before_admission_refuses(tmp_path,monkeypatch):
    from scripts import optimize_orderflow as opt,phase4b_campaign as driver
    layout,manifest,digest=runtime_fixture(tmp_path)
    stop={'reason':'max_trial_seconds','measured_seconds':601.,'limit_seconds':600,'trial':7,'reconciled':True}
    monkeypatch.setattr(opt,'read_study_attrs',lambda *a:{'identity':{'phase':'phase4b'},'phase4_runtime_stop':stop})
    args=SimpleNamespace(phase4_timeout_audit=lambda value:driver.record_termination(layout,manifest,digest,'S2t',stop=value))
    with pytest.raises(ValueError,match='runtime stop'):
        opt.phase4_db_admission(layout.studies_db,'synthetic',args,layout=layout)
    audit=driver.recover_termination_audit(layout,digest,'S2t')
    # rev 13 (R13-02): an estimated/reconciled overrun is audited but never confirmed as a timeout.
    assert audit['trial']==7 and audit['kind']=='external_or_unknown_termination'
    assert audit['charge_basis']=='upper_bound'
