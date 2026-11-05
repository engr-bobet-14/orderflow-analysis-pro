"""Fail-before-I/O admission and frozen candidate reservation proofs (T3 red first)."""
from pathlib import Path
from types import SimpleNamespace
import json
import multiprocessing as mp
import time

import pytest

from scripts import optimize_orderflow as opt, phase4_studies as p4
from scripts import phase4_policy as policy
from orderflow_system.config.settings import TradeSignalConfig, parse_trade_signals


def args(*extra):
    return opt.build_parser().parse_args(['--phase4', '--no-warm-start',
        '--study-db', 'output/optimize/phase4/studies.db',
        '--output', 'output/optimize/phase4/results.json', '--cache-dir', 'output/optimize/phase4/cache', *extra])


@pytest.mark.parametrize('extra,match', [
    (['--bar-type','time_1m','--end-date','2023-12-31'], 'time_1m'),
    (['--end-date','2024-01-01'], 'development'),
    (['--n-jobs','2'], 'n-jobs'),
    (['--evaluate-holdout'], 'holdout-look'),
    (['--trade-signals','absorption_levels'], 'trade-cache'),
    (['--trade-signals','absorption_levels','--trade-cache','output/optimize/trade_cache/x'], None),
])
def test_admission_before_data_work(extra, match, monkeypatch, tmp_path):
    a = args(*extra)
    if match is None:
        a.no_warm_start = False
        match = 'warm'
    before = set(tmp_path.rglob('*'))
    with pytest.raises(ValueError, match=match):
        p4.configure(a)
    assert set(tmp_path.rglob('*')) == before


@pytest.mark.parametrize('trigger', ['trade_signals','trade_cache','freeze_params','holdout_look','campaign_manifest',
                                    'study_db','output','cache_dir'])
def test_implied_activation(trigger):
    a = opt.build_parser().parse_args([])
    if trigger == 'trade_signals':
        a.trade_signals = 'absorption_levels'
    elif trigger in ('study_db','output','cache_dir'):
        setattr(a, trigger, Path('output/optimize/phase4/x'))
    else:
        setattr(a, trigger, 'x')
    assert p4.is_phase4(a)
    with pytest.raises(ValueError):
        p4.configure(a)


@pytest.mark.parametrize('destination', ['study_db', 'output', 'cache_dir', 'trade_cache'])
def test_write_destinations_refused_without_changes(destination, tmp_path):
    a = args()
    setattr(a, destination, Path('output/optimize/main/cache'))
    before = set(tmp_path.rglob('*'))
    with pytest.raises(ValueError, match='allowlist'):
        p4.configure(a)
    assert set(tmp_path.rglob('*')) == before


def test_symlink_allowlist_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, 'ROOT', tmp_path)
    (tmp_path/'output').mkdir()
    (tmp_path/'output/phase4').symlink_to(tmp_path/'protected', target_is_directory=True)
    with pytest.raises(ValueError, match='allowlist|output'):
        policy.checked_write(tmp_path/'output/phase4/result.json')


def test_spaces_valid_and_inactive_trade_dimensions_absent():
    bar_space = dict(opt.SEARCH_SPACE_BOUNDS)
    for mode in ('absorption_levels','exhaustion_bonus','exhaustion_required','framing_trades'):
        signals = parse_trade_signals(mode)
        space = p4.active_space(bar_space, signals, frozen=True)
        assert all(k.startswith('ts_') for k in space)
        assert len(space) == (5 if mode == 'absorption_levels' else 0 if mode == 'framing_trades' else 3)
        for k, (lo, hi, _) in space.items():
            TradeSignalConfig(**{k[3:]:lo})
            TradeSignalConfig(**{k[3:]:hi})
    assert p4.active_space(bar_space, TradeSignalConfig(), frozen=False) == bar_space


def test_holdout_reservation_reuses_or_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(p4, 'checked_write', lambda p: Path(p))
    path = tmp_path/'exposures.json'
    candidate = {'study_identity_hash':'a', 'trial':1, 'params_sha256':'b', 'replay_digest':'c', 'complete':3}
    calls = []
    def replay():
        calls.append(1)
        return {'net_expectancy_r':1.0}
    first = p4.reserved_look(path, 'tick:P2', candidate, replay)
    assert first == p4.reserved_look(path, 'tick:P2', candidate, replay)
    assert len(calls) == 1
    with pytest.raises(ValueError, match='candidate'):
        p4.reserved_look(path, 'tick:P2', {**candidate,'trial':2}, replay)
    with pytest.raises(ValueError, match='candidate'):
        p4.reserved_look(path, 'tick:P2', {**candidate,'complete':4}, replay)
    crash = tmp_path/'crash.json'
    with pytest.raises(RuntimeError):
        p4.reserved_look(crash, 'arm', candidate, lambda: (_ for _ in ()).throw(RuntimeError('crash')))
    assert p4.reserved_look(crash, 'arm', candidate, replay)['completed_after_interruption']


def _concurrent_look(path, marker):
    p4.checked_write = lambda p: Path(p)
    def replay():
        with Path(marker).open('a') as f:
            f.write('replay\n')
        time.sleep(.2)
        return {'ok':True}
    p4.reserved_look(Path(path), 'arm', {'candidate':'same'}, replay)


def test_holdout_lock_serializes_racing_attempts(tmp_path):
    ctx = mp.get_context('fork')
    children = [ctx.Process(target=_concurrent_look, args=(str(tmp_path/'e.json'), str(tmp_path/'m'))) for _ in range(2)]
    for child in children:
        child.start()
    for child in children:
        child.join(10)
        assert child.exitcode == 0
    assert (tmp_path/'m').read_text() == 'replay\n'


def test_legacy_guard_reads_identity_before_open(monkeypatch):
    a = opt.build_parser().parse_args([])
    a.symbol,a.bar_type = 'BTCUSDT','fixed_tick_imbalance'
    seen = []
    monkeypatch.setattr(opt,'_stored_identity',lambda db,name:seen.append(name) or {'phase':'phase4'})
    with pytest.raises(ValueError,match='legacy'):
        p4.legacy_study_guard(a,'protected')
    assert seen == ['protected']


def test_off_legacy_identity_unchanged(tmp_path):
    from test_optimize_orderflow import args_for
    a = args_for(tmp_path)
    a.trade_signals = 'off'
    a.phase4 = False
    first = opt.study_identity(a,{'fingerprint':'f'*64})
    del a.trade_signals
    assert opt.study_identity(a,{'fingerprint':'f'*64}) == first
    assert first[1]['trade_signals']=='disabled'
    assert 'phase' not in first[1]


def test_no_worker_budget_overshoot(monkeypatch):
    # No real trial/storage: admission counts COMPLETE only and stops exactly at the budget.
    from types import SimpleNamespace
    a = SimpleNamespace(n_trials=3,seed=42,n_jobs=1,phase4=True)
    study = SimpleNamespace(trials=[SimpleNamespace(state='FAIL')])
    monkeypatch.setattr(opt,'complete_trials',lambda s:[t for t in s.trials if t.state=='COMPLETE'])
    study.optimize = lambda objective,n_trials:study.trials.append(SimpleNamespace(state='COMPLETE'))
    opt.optimize_loop(study,None,None,a)
    assert len(study.trials)==4 and len(opt.complete_trials(study))==3


def test_symlink_into_unapproved_output_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(policy,'ROOT',tmp_path)
    (tmp_path/'output').mkdir()
    (tmp_path/'output/unapproved').mkdir()
    (tmp_path/'output/phase4').symlink_to(tmp_path/'output/unapproved',target_is_directory=True)
    with pytest.raises(ValueError,match='allowlist'):
        policy.checked_write(tmp_path/'output/phase4/file')


def test_holdout_dry_run_cannot_reserve_or_replay(monkeypatch):
    a = args('--dry-run','--bar-type','fixed_tick_imbalance','--holdout-look','s:0')
    a.symbol,a.bar_type = 'BTCUSDT','fixed_tick_imbalance'
    a.space = dict(opt.SEARCH_SPACE_BOUNDS)
    monkeypatch.setattr(opt,'CachedBars',lambda *a:[])
    monkeypatch.setattr(p4,'attach_cache',lambda *a:None)
    monkeypatch.setattr(p4,'load_frozen',lambda *a:None)
    monkeypatch.setattr(p4,'holdout_look',lambda *a:pytest.fail('dry run exposed holdout'))
    assert opt.optimize_study(a,None,{})['dry_run']


def test_completed_candidate_exact_budget_and_pending_refused(tmp_path):
    import optuna
    from test_warm_start import make_source_study,valid_params
    params = valid_params()
    db = tmp_path/'unit.db'
    study = make_source_study(db,'unit',[
        {'value':1.,'attrs':{'params':params,'replay_digest':'a'*64}},
        {'value':2.,'attrs':{'params':params,'replay_digest':'b'*64}}],
        study_attrs={'identity':{'phase':'phase4','trade_signals':'disabled'}})
    best,identity,reservation = p4.completed_candidate(db,'unit',1,2)
    assert best['number']==1 and reservation['complete']==2
    with pytest.raises(ValueError,match='exact COMPLETE'):
        p4.completed_candidate(db,'unit',1,3)
    with pytest.raises(ValueError,match='best feasible'):
        p4.completed_candidate(db,'unit',0,2)
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.WAITING))
    with pytest.raises(ValueError,match='RUNNING/WAITING'):
        p4.completed_candidate(db,'unit',1,2)


def test_p2_artifact_and_single_look_without_study(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from test_warm_start import valid_params
    from orderflow_system.data.models import BarType
    monkeypatch.setattr(p4,'ROOT',tmp_path)
    monkeypatch.setattr(policy,'ROOT',tmp_path)
    monkeypatch.setattr(p4,'checked_write',lambda p:Path(p))
    monkeypatch.setattr(opt,'_shared_lock_readonly',lambda *a:nullcontext())
    monkeypatch.setattr(opt,'open_study',lambda *a:pytest.fail('P2 created a study'))
    monkeypatch.setattr(opt,'split_timeline',lambda *a:(1,[]))
    monkeypatch.setattr(opt,'run_backtest',lambda *a,**kw:{'replay_digest':'c'*64,'raw_signal_digest':'d'*64})
    source = {'study_identity_hash':'a','trial':4,'params_sha256':'b','replay_digest':'e','complete':200}
    monkeypatch.setattr(p4,'completed_candidate',lambda *a:({}, {'trade_signals':'disabled'}, source))
    spec = tmp_path/'freeze.json'
    monkeypatch.setattr(opt,'read_study_attrs',lambda *a:{'identity':{'trade_signals':'disabled'}})
    spec.write_text(json.dumps({'study_db':str(tmp_path/'output/optimize/phase4/b0.db'),'study':'B0','trial':4,'budget':200}))
    a = SimpleNamespace(symbol='BTCUSDT',bar_type='fixed_tick_imbalance',trade_signals='framing_trades',
        trade_signal_config=parse_trade_signals('framing_trades'),trade_updates=[None],
        frozen_values=valid_params(),frozen_identity={**source,'source_study':'B0'},freeze_params=spec,
        trade_cache_identity={'manifest_sha256':'a','builder_fp':'b'},campaign_identity='campaign')
    artifact = p4.write_candidate(a,[SimpleNamespace(bar_type=BarType.TICK_IMBALANCE)])
    loaded = p4.read_candidate(artifact['candidate'])
    assert loaded['mode']=='framing_trades'
    path = tmp_path/'exposures.json'
    result = p4.reserved_look(path,'P2',{'artifact_sha256':artifact['candidate']},lambda:{'ok':True})
    assert p4.reserved_look(path,'P2',{'artifact_sha256':artifact['candidate']},lambda:pytest.fail('second replay'))==result
    with pytest.raises(ValueError,match='candidate'):
        p4.reserved_look(path,'P2',{'artifact_sha256':'other'},lambda:{})
    Path(artifact['path']).write_text('{}')
    with pytest.raises(ValueError,match='tampered'):
        p4.read_candidate(artifact['candidate'])


def test_main_implied_trade_mode_defaults_fail_before_any_data_or_write(monkeypatch):
    before = {str(p) for p in (opt.ROOT/'output/optimize').rglob('*')}
    for name in ('build_cache','require_source','open_study'):
        monkeypatch.setattr(opt,name,lambda *a,**k:pytest.fail('unsafe request reached I/O'))
    with pytest.raises(SystemExit):
        opt.main(['--trade-signals','absorption_levels','--trade-cache','output/optimize/trade_cache',
                  '--no-warm-start'])
    assert {str(p) for p in (opt.ROOT/'output/optimize').rglob('*')}==before


def test_minute_flags_off_refused_before_trade_root_discovery(monkeypatch):
    for name in ('build_cache','require_source'):
        monkeypatch.setattr(opt,name,lambda *a,**k:pytest.fail('minute mode reached source discovery'))
    with pytest.raises(SystemExit):
        opt.main(['--phase4','--trade-signals','off','--bar-type','time_1m','--end-date','2023-12-31'])


def test_p2_requires_frozen_b0_before_io():
    with pytest.raises(ValueError,match='freeze-params'):
        p4.configure(args('--trade-signals','framing_trades','--trade-cache','output/optimize/trade_cache'))


def test_stored_holdout_does_not_replay_frozen_baseline(monkeypatch):
    a = args('--bar-type','fixed_tick_imbalance','--holdout-look','s:0')
    a.symbol,a.bar_type = 'BTCUSDT','fixed_tick_imbalance'
    a.space = dict(opt.SEARCH_SPACE_BOUNDS)
    monkeypatch.setattr(opt,'CachedBars',lambda *a:[])
    monkeypatch.setattr(p4,'attach_cache',lambda *a:None)
    monkeypatch.setattr(p4,'load_frozen',lambda *a:pytest.fail('repeat replayed frozen B0'))
    monkeypatch.setattr(p4,'holdout_look',lambda *a:{'stored':True})
    assert opt.optimize_study(a,None,{})=={'stored':True}
