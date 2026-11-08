"""Phase 4b runtime accounting (plan rev 13 C7): conservative charges never become a confirmed timeout.

The Phase 4 functions (reconcile_phase4_trials, reconciled_runtime_stop, phase4_runtime_stop) are reused
unchanged for accounting; only their stop *promotion* is withheld here, by never handing them limits.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import time

from scripts import optimize_orderflow as opt

UPPER_BOUND = 'upper_bound'
OBSERVED = 'observed'


def _accounting_only(args):
    # No limits: reconciled_runtime_stop then records nothing; first-party memory stops still apply.
    return SimpleNamespace(phase4=getattr(args, 'phase4', True))


def _upper_bound_charges(study):
    return [r for r in study.user_attrs.get('phase4_reconciliation_audit', [])
            if r.get('accounting') == 'conservative_upper_bound']


def reconcile(study, args):
    """Caller holds the study and worker locks. RUNNING/WAITING -> FAIL, charged conservatively."""
    pending = study.user_attrs.get('phase4_accounting_pending')
    if pending and any(v is not None for v in pending.get('limits', {}).values()):
        # The write-ahead marker carries the old limits; drop them so the estimate is not promoted.
        study.set_user_attr('phase4_accounting_pending',
                            {**pending, 'limits': {k: None for k in pending.get('limits', {})}})
    opt.reconcile_phase4_trials(study, _accounting_only(args))
    audit = study.user_attrs.get('phase4_reconciliation_audit', [])
    labelled = [{**r, 'charge_basis': UPPER_BOUND} if r.get('accounting') == 'conservative_upper_bound'
                and 'charge_basis' not in r else r for r in audit]
    if labelled != audit:
        study.set_user_attr('phase4_reconciliation_audit', labelled)


def prepare(db, name):
    """Before open_study admission (caller holds the study and worker locks): resolve a pending
    write-ahead marker as accounting only, so Phase 4's admission cannot promote its estimate."""
    from pathlib import Path
    if not Path(db).exists() or name not in opt.list_study_names(db):
        return False
    attrs = opt.read_study_attrs(db, name)
    if attrs.get('identity', {}).get('phase') != 'phase4b' or not attrs.get('phase4_accounting_pending'):
        return False  # Never mutates a non-4b study; open_study's admission still checks identity.

    import optuna
    study = optuna.load_study(storage='sqlite:///' + str(db), study_name=name)
    reconcile(study, SimpleNamespace(phase4=True))
    return True


def cumulative_runtime(study):
    reset = study.user_attrs.get('phase4_runtime_reset', {}).get('runtime_seconds', 0.)
    return opt.phase4_runtime_baseline(study) - reset


def admit(study, args):
    """After reconcile: an exhausted study allowance is a distinct, consuming, non-timeout kind."""
    limit = getattr(args, 'max_study_seconds', None)
    charged = cumulative_runtime(study)
    if limit is not None and charged >= limit:
        stop = {'reason': 'study_allowance_exhausted', 'measured_seconds': charged, 'limit_seconds': limit,
                'charge_basis': UPPER_BOUND if _upper_bound_charges(study) else OBSERVED,
                'boundaries': {'prior_charge_seconds': charged, 'current_seconds': 0.},
                'utc': datetime.now(timezone.utc).isoformat()}
        audit = getattr(args, 'phase4_termination_audit', None)
        if audit is not None:
            audit(stop)
        raise ValueError('study allowance exhausted by charged runtime; stage consumed (not a timeout)')
    return charged


def runtime_stop(study, args, elapsed, *, prior, begun):
    """Live loop check. Trial durations are observed terminal walls; the study clock is prior + current."""
    if study.user_attrs.get('phase4_runtime_stop') is not None:
        return True
    current = time.perf_counter() - begun
    stop = None
    trial_limit = getattr(args, 'max_trial_seconds', None)
    if trial_limit is not None:
        reset = study.user_attrs.get('phase4_runtime_reset', {}).get('last_trial_number', -1)
        walls = study.user_attrs.get('phase4_terminal_wall_seconds', {})
        for trial in study.trials:
            duration = walls.get(str(trial.number))
            if trial.number <= reset or duration is None:
                continue
            if duration >= trial_limit:
                stop = {'reason': 'max_trial_seconds', 'trial': trial.number, 'measured_seconds': duration,
                        'limit_seconds': trial_limit,
                        'boundaries': {'prior_charge_seconds': 0., 'current_seconds': duration,
                                       # Optuna stores naive local time (review I1).
                                       'trial_started_utc': trial.datetime_start.astimezone(timezone.utc).isoformat()
                                       if trial.datetime_start else None}}
                break
    study_limit = getattr(args, 'max_study_seconds', None)
    since_reset = elapsed - study.user_attrs.get('phase4_runtime_reset', {}).get('runtime_seconds', 0.)
    if stop is None and study_limit is not None and since_reset >= study_limit:
        stop = {'reason': 'max_study_seconds', 'measured_seconds': since_reset, 'limit_seconds': study_limit,
                'boundaries': {'prior_charge_seconds': prior, 'current_seconds': current,
                               'current_attempt_started_perf_counter': begun,
                               # The prior charge may include stale-bound estimates (plan rev 13 C7).
                               'prior_charge_basis': UPPER_BOUND if _upper_bound_charges(study) else OBSERVED}}
    if stop is None:
        return False
    stop.update(live=True, charge_basis=OBSERVED, utc=datetime.now(timezone.utc).isoformat())
    study.set_user_attr('phase4_runtime_stop', stop)
    audit = getattr(args, 'phase4_termination_audit', None)
    if audit is not None:
        audit(stop)
    return True
