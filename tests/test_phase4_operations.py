"""Campaign/benchmark/probe code is tested without creating real studies (T5/T6/T8)."""
from pathlib import Path
import pytest
from scripts import phase4_campaign as campaign, phase4_benchmark as bench, phase4_d6_probe as probe
from scripts import search_space_audit as audit, walk_forward_selection as walk, optimize_orderflow as opt


def test_campaign_sealed_and_dry_run(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'checked_write',lambda p:Path(p))
    path = tmp_path/'campaign.json'
    sources=[]
    for i in range(3):
        db=tmp_path/f'legacy-{i}.db';db.write_bytes(b'legacy')
        sources.append({'path':str(db),'sha256':campaign.sha256_file(db)})
    manifest = campaign.register(path,sources)
    assert len(manifest['arms']) == 10
    assert sum(a['budget'] for a in manifest['arms'].values() if a['kind']=='study') == 1000
    assert campaign.read_campaign(path)[0] == manifest
    commands = campaign.commands(path,manifest)
    assert all('--no-warm-start' in command for command in commands)
    assert not (tmp_path/'studies.db').exists()
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='hash'):
        campaign.read_campaign(path)


def test_benchmark_extrapolation_and_boundary():
    rows=[]
    for trades,bars in ((100,10),(200,20),(400,30)):
        rows.append({'trade_count':trades,'verified_trade_count':trades,'bar_counts':{'fixed_tick_imbalance':bars},
            'build_seconds':2+.01*trades,'cache_bytes':150+10*bars,
            'disk_components':{'fixed_bytes':100,'bar_bytes':10*bars,'day_bytes':50,'finalised_days':1},
            'peak_rss_bytes':100,'replay_seconds':{'fixed_tick_imbalance':{m:bars*.1 if m=='off' else bars*.2 for m in bench.MODES}},
            'cache_open':{'fixed_tick_imbalance':{'hash_seconds':0.,'hash_bytes':100.,'digest_seconds':0.,'bars':bars,'other_seconds':0.}}})
    result=bench.extrapolate(rows,10000,{'fixed_tick_imbalance':1000},full_days=10)
    assert result['build_wall_seconds']==pytest.approx(102)
    assert result['disk_bytes']==pytest.approx(10600)
    assert result['trial_seconds']['fixed_tick_imbalance']['absorption_levels']==pytest.approx(160)
    with pytest.raises(ValueError,match='development'):
        bench.validate_days('2024-01-01')


def test_probe_requires_owner_approval_before_reads(monkeypatch):
    monkeypatch.setattr(opt,'read_study_trials',lambda *a:pytest.fail('read before approval'))
    with pytest.raises(ValueError,match='approval'):
        probe.run_probe(Path('missing.db'),Path('missing-cache'),None)


def test_audit_and_walk_forward_accept_trade_dims():
    trials = [{'number':i,'state':'COMPLETE','value':float(i),'feasible':True,'params':{'ts_max_through_bps':i/10}}
              for i in range(10)]
    assert 'ts_max_through_bps' in audit.boundary_hits(trials,{'ts_max_through_bps':(0.,4.,'float')})
    # Walk-forward uses fold metrics only; dimensionality is intentionally irrelevant.
    t = {'number':1,'params':trials[0]['params'],'sharpe':[1.,2.],'n_trades':[20,20],
         'blown':[False,False],'weight':[10,10]}
    assert walk.selection_walk_forward([t],11)[0]['selected_trial'] == 1


def test_probe_uses_read_only_database_opens(tmp_path,monkeypatch):
    import sqlite3
    from test_warm_start import make_source_study,valid_params
    from test_optimize_orderflow import args_for
    from test_pipeline import synth_bars
    from scripts import diagnose_trades as diagnose
    identity=opt.study_identity(args_for(tmp_path),{'fingerprint':'a'*64})[1]
    identity['strategy_fp']='f6820adb'+'0'*56
    db=tmp_path/'unit.db'
    make_source_study(db,'unit',[{'value':1.,'attrs':{'params':valid_params(),'replay_digest':'b'*64}}],
                      study_attrs={'identity':identity})
    monkeypatch.setattr(diagnose,'find_cached_bars',lambda *a:(None,{'symbol':'BTCUSDT','bar_type':'fixed_dollar_imbalance',
                                                               'start':'2023-01-01','end':'2023-02-01'}))
    class UnitBars(list):
        def __init__(self,*args):
            super().__init__(synth_bars(20))
    monkeypatch.setattr(opt,'CachedBars',UnitBars)
    monkeypatch.setattr(opt,'run_backtest',lambda *a,**kw:{'replay_digest':'b'*64})
    original=sqlite3.connect
    opened=[]
    def connect(path,*a,**kw):
        opened.append(str(path))
        assert str(path).endswith('?mode=ro') and kw.get('uri') is True
        return original(path,*a,**kw)
    monkeypatch.setattr(sqlite3,'connect',connect)
    monkeypatch.setattr(probe,'approve_probe',lambda *a:{'sha256':'unit-test-approval'})
    result=probe.run_probe(db,tmp_path/'cache','unit-test-only')
    assert result['all_match'] and len(result['trials'])==1
    assert opened


def test_probe_static_classification_covers_each_hunk():
    diff='diff --git a/a.py b/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n@@ -4 +4 @@\n-x\n+y\n'
    rows=probe.static_hunks(diff)
    assert len(rows)==2
    assert all(r['file']=='a.py' and r['classification']=='flags_off_reachable' for r in rows)
    assert all('conservative' in r['basis'] for r in rows)
