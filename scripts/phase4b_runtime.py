"""Executable and configuration binding shared by calibration and campaign admission."""
from dataclasses import asdict
from enum import Enum
import hashlib
import importlib
from importlib import metadata
import inspect
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

from scripts.phase4_policy import ROOT,canonical_bytes,source_state


def runtime_fingerprint():
    # Load the same execution closure in registration and worker processes, including lazy imports, so the hashed
    # module set never depends on import history (S1 attempt 9056f902: the worker's gate check lazily imported
    # phase4b_proof, the post-flight child did not). tests/test_phase4b_fingerprint_determinism.py guards closure.
    for name in ('scripts.phase4_campaign','scripts.phase4b_campaign','scripts.phase4_report','scripts.phase4b_regression',
                 'scripts.phase4_benchmark','scripts.phase4_resources','scripts.trade_feature_cache',
                 'scripts.phase4b_proof','scripts.phase4b_accounting','scripts.search_space_audit'):
        importlib.import_module(name)
    from scripts import optimize_orderflow as opt,phase4_studies as p4
    from scripts.trade_feature_cache import BUILDER_FILES
    from orderflow_system.config.settings import TradeFlowConfig
    from orderflow_system.signals.aggregator import SignalAggregator
    modules={}
    for name,module in tuple(sys.modules.items()):
        if name.split('.')[0] not in ('orderflow_system','scripts') or not getattr(module,'__file__',None):
            continue
        path=Path(module.__file__).resolve()
        key=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        modules[key]=hashlib.sha256(path.read_bytes()).hexdigest()
    def plain(value):
        if isinstance(value,Enum):return value.value
        if isinstance(value,dict):return {key:plain(item) for key,item in value.items()}
        if isinstance(value,(list,tuple)):return [plain(item) for item in value]
        return value
    sampled=p4.full_strategy_defaults()
    configuration={
        'instrument_defaults':asdict(opt.instrument_config('BTCUSDT')),
        'strategy_defaults':{'sampled':sampled,'effective_params':p4.full_strategy_params(sampled),
            'aggregator':{name:param.default for name,param in inspect.signature(SignalAggregator).parameters.items()}},
        'trade_flow_defaults':asdict(TradeFlowConfig()),
        'effective_trade_flow':asdict(p4.effective_trade_config(SimpleNamespace(phase='phase4b'),sampled)),
        'search_space':p4.FULL_STRATEGY_SPACE}
    return {'source_state':source_state(),'module_sha256':modules,
        'frozen_builder_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in BUILDER_FILES},
        'configuration':canonical_bytes(plain(configuration)).decode(),
        'versions':{'python':platform.python_version(),**{name:metadata.version(name)
            for name in ('numpy','scipy','pandas','optuna','RiskLabAI')}}}


def require_runtime_fingerprint(expected):
    if not expected or expected!=runtime_fingerprint():
        raise ValueError('runtime fingerprint changed; stale calibration requires new S1 registration')
