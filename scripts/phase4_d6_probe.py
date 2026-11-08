"""Owner-gated read-only replay probe for historical f6820adb studies; no compatibility edits."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import optimize_orderflow as opt, diagnose_trades as diagnose
from scripts.dataset_policy import ensure_dev_window
from scripts.phase4_policy import checked_write, source_state, verify_source
from scripts.measure_rss import maxrss_bytes


def static_hunks(diff):
    """Fail closed: classify every changed hunk as potentially flags-off reachable.

    This conservative overapproximation does not certify an unreachable branch.
    The inspector can refine the table after checking the guards and call sites.
    """
    rows,current = [],None
    for line in diff.splitlines():
        if line.startswith('+++ b/'):
            current = line[6:]
        elif line.startswith('@@ '):
            rows.append({'file':current,'hunk':line,'classification':'flags_off_reachable',
                         'basis':'conservative: potential import/definition/runtime reachability; inspect guards manually'})
    return rows


def approve_probe(owner_approved,campaign_manifest):
    from scripts.phase4_studies import read_campaign
    from scripts.phase4_approvals import verify_approval, consume_approval
    if not campaign_manifest:
        raise ValueError('D6 probe requires host campaign approval and --campaign-manifest')
    _,digest=read_campaign(campaign_manifest)
    approval=verify_approval('campaign',owner_approved,digest)
    consume_approval(approval)
    return approval


def run_probe(db,cache_dir,owner_approved,campaign_manifest=None):
    if not owner_approved:
        raise ValueError('D6 full replay requires later owner approval')
    approval=approve_probe(owner_approved,campaign_manifest)
    state = source_state()
    from orderflow_system.data.paired_dataset import sha256_file
    before = sha256_file(db)
    rows = []
    for name in opt.list_study_names(db):
        identity = opt.read_study_attrs(db,name).get('identity',{})
        if not identity.get('strategy_fp','').startswith('f6820adb'):
            continue
        path,meta = diagnose.find_cached_bars(cache_dir,identity['data_fp'])
        ensure_dev_window(date.fromisoformat(meta['start']),date.fromisoformat(meta['end']))
        if meta['bar_type']=='time_1m':
            raise ValueError('D6 probe reads existing info-bar caches only')
        bars = opt.CachedBars(path,meta['bar_type'])
        settings = diagnose.bind_replay_settings(identity)
        args = SimpleNamespace(**settings,symbol=meta['symbol'])
        n_search,_ = opt.split_timeline(bars,args)
        for trial in opt.read_study_trials(db,name):
            if trial['state']!='COMPLETE':
                continue
            params = {**opt.baseline_extra_params(meta['symbol']),**trial['attrs']['params']}
            result = opt.run_backtest(bars[:n_search],opt.build_instrument_config(params,meta['symbol']),
                                      params['price_proximity_pct'],bars[0].bar_type)
            expected = trial['attrs'].get('replay_digest')
            rows.append({'study':name,'trial':trial['number'],'expected':expected,
                         'actual':result['replay_digest'],'matches':expected==result['replay_digest']})
            import resource
            if maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)>4*1024**3:
                raise ValueError('D6 RSS exceeded 4 GiB')
    if sha256_file(db)!=before:
        raise ValueError('source DB changed during read-only probe')
    verify_source(state)
    diff = subprocess.check_output(['git','-C',str(ROOT),'diff','43f7c75','--','orderflow_system','scripts/optimize_orderflow.py'],text=True)
    return {'source_db_sha256':before,'source_state':state,'owner_approval_sha256':approval['sha256'],
            'trials':rows,'all_match':all(r['matches'] for r in rows),
            'static_diff':diff,'static_hunks':static_hunks(diff),
            'static_classification':'conservative: all hunks potentially flags-off reachable; manual refinement required',
            'recommendation':'keep compatibility list empty; owner decision after Phase 4 merges'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-db',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--owner-approved')
    parser.add_argument('--campaign-manifest',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'output/phase4/d6/report.json')
    cli = parser.parse_args(argv)
    out = checked_write(cli.out)
    report = run_probe(cli.study_db,cli.cache_dir,cli.owner_approved,cli.campaign_manifest)
    out.parent.mkdir(parents=True,exist_ok=True)
    opt.atomic_json(out,report)
    return report


if __name__=='__main__':
    main()
