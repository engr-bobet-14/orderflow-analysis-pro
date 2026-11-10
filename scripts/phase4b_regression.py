"""Read-only Phase 4b host proofs. Never run against real artifacts without owner authorization."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import phase4_studies as p4, optimize_orderflow as opt


def checkout_path(value, root=ROOT):
    """Historical absolute paths are rebased onto this checkout, never the original."""
    path=Path(value)
    if path.is_absolute():
        if path.is_relative_to(Path(root).resolve()):
            path=path.relative_to(Path(root).resolve())
        else:
            parts=path.parts
            worktrees=[i for i in range(len(parts)-2) if parts[i:i+2]==('.claude','worktrees')]
            if worktrees:
                path=Path(*parts[worktrees[-1]+3:])
            else:
                markers=[i for i,p in enumerate(parts) if p in ('output','scripts','orderflow_system','tests','docs','notebooks')]
                if not markers:raise ValueError('source path has no checkout-relative suffix')
                path=Path(*parts[markers[0]:])
    resolved=(Path(root)/path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):raise ValueError('source path escapes checkout')
    return resolved


def verify_protected(root, expected):
    for name,digest in expected.items():
        path=checkout_path(name,root)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('protected file changed: '+str(path))
    return {'checked':len(expected),'unchanged':True}


def verify_replays(legacy, comparators, replay_legacy, replay_comparator):
    for value in legacy:
        if replay_legacy(value)['replay_digest']!=value.replay_digest:raise ValueError('Stage A digest mismatch')
    for value in comparators:
        if replay_comparator(value)['replay_digest']!=value.replay_digest:raise ValueError('Stage B digest mismatch')
    return {'stageA':len(legacy),'stageB':len(comparators),'byte_identical':True}


def _prefix(bt):
    from scripts.trade_feature_cache import production_generation,generation_directory
    from scripts.phase4_benchmark import DATASET_ID
    from types import SimpleNamespace
    root=generation_directory(ROOT/'output/optimize/trade_cache',DATASET_ID)
    generation=production_generation(root);path=Path(generation['lanes'][bt])
    return p4.SearchPrefix.open(path/'bars.bin',bt,path,SimpleNamespace(
        holdout_fraction=.2,n_splits=5,embargo=.02),production=True,dataset_id=DATASET_ID,bar_type=bt)


def check_digests(stage_a,stage_b):
    from orderflow_system.data.models import BarType
    count_a=count_b=0
    modes=('off','absorption_levels','exhaustion_bonus','exhaustion_required','framing_trades',
           'absorption_levels,exhaustion_bonus,framing_trades')
    defaults={k:v for k,v in p4.full_strategy_defaults().items() if not k.startswith('ts_')}
    defaults['absorption_min_delta_pct']=opt.instrument_config('BTCUSDT').absorption.min_delta_pct
    for bt in BarType:
        rows=json.loads((stage_a/f'{bt.value}.json').read_bytes())
        if set(rows)!=set(modes):raise ValueError('Stage A original six configurations required')
        prefix=_prefix(bt.value)
        for mode in modes:
            signals=p4.signals_for(mode)
            replay=p4.LegacyReplay(defaults,signals,rows[mode]['replay_digest'])
            result=p4.replay_legacy(replay,prefix)
            if mode!='off' and result.get('raw_signal_digest')!=rows[mode].get('raw_signal_digest'):
                raise ValueError('Stage A raw signal digest mismatch')
            count_a+=1
    for bt in ('fixed_tick_imbalance','fixed_volume_imbalance'):
        frozen=json.loads((stage_b/f'{bt}.json').read_bytes())
        db=checkout_path(frozen['study_db'])
        trials=sorted((t for t in opt.read_study_trials(db,frozen['study'])
            if t['state']=='COMPLETE' and t['feasible'] and t['value'] is not None),
            key=lambda t:(-t['value'],t['number']))
        if len(trials)<4:raise ValueError('Stage B proof needs top3 and median feasible trials')
        best=trials[0]
        if (best['number']!=frozen['trial'] or best['attrs']['params']!=frozen['params']
                or best['attrs']['replay_digest']!=frozen['replay_digest']):
            raise ValueError('Stage B frozen comparator differs from stored best')
        prefix=_prefix(bt)
        selected={t['number']:t for t in [*trials[:3],trials[len(trials)//2]]}
        for trial in selected.values():
            p4.replay_comparator(p4.FrozenComparator(trial['attrs']['params'],trial['attrs']['replay_digest']),prefix)
            count_b+=1
    if (count_a,count_b)!=(36,8):raise ValueError('expected 36 Stage A and 8 unique Stage B proofs')
    return {'stageA':count_a,'stageB':count_b,'byte_identical':True}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stageA',type=Path,default=Path('output/phase4/ablation'))
    parser.add_argument('--stageB',type=Path,default=Path('output/phase4/frozen'))
    parser.add_argument('--check-digests',action='store_true')
    parser.add_argument('--protected',action='store_true')
    parser.add_argument('--baseline',type=Path,default=Path('output/phase4b/evidence/protected-before.json'),
        help='host hash inventory collected before build; includes 46 protected files, DBs, Phase 4 tree, COMPLETE and builders')
    cli=parser.parse_args(argv)
    try:
        if cli.check_digests:return check_digests(checkout_path(cli.stageA),checkout_path(cli.stageB))
        if cli.protected:
            expected=json.loads(checkout_path(cli.baseline).read_bytes())
            from scripts.trade_feature_cache import BUILDER_FILES
            required={*BUILDER_FILES,'output/optimize/studies.db','output/optimize/main/studies.db',
                'output/optimize/smoke/studies.db','output/optimize/phase4/studies.db',
                'output/phase4/campaign.json','output/phase4/campaign.json.sha256'}
            local={str(checkout_path(p).relative_to(ROOT)) for p in expected}
            if (not required<=local or len(local)<46
                    or not any(p.startswith('output/optimize/trade_cache/') and p.endswith('/COMPLETE.json') for p in local)):
                raise ValueError('protected inventory lacks required frozen inputs')
            return verify_protected(ROOT,expected)
        raise ValueError('choose --check-digests or --protected')
    except (OSError,ValueError,KeyError) as exc:parser.error(str(exc))


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
