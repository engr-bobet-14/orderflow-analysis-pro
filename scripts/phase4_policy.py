"""Phase 4 filesystem and source provenance guards; no implicit writes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parents[1]
WRITE_ROOTS = ('output/optimize/phase4', 'output/optimize/trade_cache', 'output/phase4', 'output/duckdb_tmp')
PROTECTED = ('output/optimize/studies.db', 'output/optimize/results.json', 'output/optimize/cache',
             'output/optimize/main', 'output/optimize/smoke', 'output/certificates', 'output/diagnose',
             'orderflow_system/config/paired_datasets.json')


@dataclass(frozen=True)
class PhaseLayout:
    """All campaign-owned paths; phase4 retains its original policy exactly."""
    phase: str
    root: Path = field(default_factory=lambda: ROOT)
    approvals_base: Path | None = None

    def __post_init__(self):
        if self.phase not in ('phase4', 'phase4b', 'phase4b-s1'):
            raise ValueError('unknown campaign phase')
        object.__setattr__(self, 'root', Path(self.root).resolve())

    @property
    def namespace(self):
        return 'phase4' if self.phase == 'phase4' else 'phase4b'

    @property
    def output(self):
        return self.root/'output'/self.namespace

    @property
    def optimize(self):
        return self.root/'output/optimize'/self.namespace

    @property
    def manifest(self):
        return self.output/('s1/campaign.json' if self.phase == 'phase4b-s1' else 'campaign.json')

    @property
    def studies_db(self):
        return self.optimize/'studies.db'

    @property
    def approval_root(self):
        import os
        import pwd
        base = self.approvals_base or Path(pwd.getpwuid(os.getuid()).pw_dir)/'.orderflow-approvals'
        return Path(base)/self.namespace

    @property
    def allocation_root(self):
        return self.approval_root.parent/'holdout-allocation'

    @property
    def exposure(self):
        return self.output/'holdout/exposures.json'

    @property
    def checkpoints(self):
        return self.output/'checkpoints'

    @property
    def snapshots(self):
        return self.output/'reports/snapshots'

    @property
    def write_roots(self):
        return WRITE_ROOTS if self.phase == 'phase4' else ('output/phase4b', 'output/optimize/phase4b', 'output/duckdb_tmp')

    def checked_write(self, path):
        path = Path(path).expanduser()
        resolved = (path if path.is_absolute() else self.root/path).resolve()
        protected = PROTECTED + (() if self.phase == 'phase4' else
            ('output/phase4', 'output/optimize/phase4', 'output/optimize/trade_cache'))
        if (not resolved.is_relative_to(self.root/'output')
                or not any(resolved.is_relative_to(self.root/p) for p in self.write_roots)
                or any(resolved.is_relative_to(self.root/p) for p in protected)
                or resolved.is_relative_to('/Volumes')):
            raise ValueError(f'{self.phase} write destination outside allowlist: {resolved}')
        return resolved

    def db_path(self, path):
        result = self.checked_write(path)
        if (self.phase == 'phase4' and not result.is_relative_to(self.optimize)
                or self.phase != 'phase4' and result != self.studies_db):
            raise ValueError('database differs from the phase layout')
        return result


def checked_write(path):
    resolved = Path(path).expanduser().resolve()
    if (not any(resolved.is_relative_to(ROOT/p) for p in WRITE_ROOTS)
            or resolved.is_relative_to('/Volumes')
            or any(resolved == ROOT/p or resolved.is_relative_to(ROOT/p) for p in PROTECTED)):
        raise ValueError(f'Phase 4 write destination outside allowlist: {resolved}')
    # A symlinked allowlist root must not grant access to an external/protected tree.
    if not resolved.is_relative_to(ROOT/'output'):
        raise ValueError(f'Phase 4 write destination outside checkout output: {resolved}')
    return resolved


def source_state(base='HEAD'):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args])
    revision = git('rev-parse', base).decode().strip()
    diff = git('diff', '--binary', revision)
    untracked = []
    for raw in sorted(git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0')):
        if raw:
            path = raw.decode()
            untracked.append((path, hashlib.sha256((ROOT/path).read_bytes()).hexdigest()))
    value = hashlib.sha256(diff + json.dumps(untracked, separators=(',', ':')).encode()).hexdigest()
    return {'base': revision, 'sha256': value, 'dirty': bool(diff or untracked)}


def verify_source(before):
    if source_state(before['base']) != before:
        raise ValueError('source_state changed during run')


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def phase4_db_path(path):
    try:
        resolved=Path(path).expanduser().resolve()
    except TypeError as exc:
        raise ValueError('Phase 4 database path is required') from exc
    if not resolved.is_relative_to(ROOT/'output/optimize/phase4'):
        raise ValueError('database must resolve inside output/optimize/phase4 (Phase 4)')
    return resolved


def protected_state():
    """46-file owner baseline plus the expanded protected trees; all reads are read-only."""
    from orderflow_system.data.paired_dataset import sha256_file
    baseline = ROOT/'output/phase4/build/protected-before.sha256'
    if not baseline.is_file():
        raise ValueError('protected-before.sha256 baseline required before real runs')
    entries = [line.split(maxsplit=1)[1].strip() for line in baseline.read_text().splitlines() if line.strip()]
    expected = {line.split(maxsplit=1)[1].strip():line.split(maxsplit=1)[0]
                for line in baseline.read_text().splitlines() if line.strip()}
    files = {Path(p) for p in entries}
    registry = next((p for p in files if p.as_posix().endswith('/orderflow_system/config/paired_datasets.json')),None)
    if registry is None:
        raise ValueError('protected baseline lacks dataset registry')
    main_root = registry.parents[2]
    for base in {ROOT,main_root}:
        for rel in ('output/optimize/main','output/optimize/smoke','output/certificates','output/diagnose'):
            files.update(p for p in (base/rel).rglob('*') if p.is_file())
    actual = {str(p):sha256_file(p) for p in sorted(files)}
    if any(actual[p] != digest for p,digest in expected.items()):
        raise ValueError('protected set differs from owner baseline')
    return actual


def bounded_preflight():
    import shutil
    if not Path('/Volumes/Market-Data').is_dir():
        raise ValueError('Market-Data is not mounted')
    if shutil.disk_usage(ROOT).free < 100*1024**3:
        raise ValueError('preflight requires 100 GiB free disk')
    return protected_state()


def assert_no_conflicts(*, strict=True):
    import os
    import psutil
    scripts = {'optimize_orderflow.py','replay_trades.py','trade_feature_cache.py','phase4_benchmark.py',
               'phase4_campaign.py','phase4b_campaign.py','phase4b_regression.py','validate_oos.py','diagnose_trades.py'}
    try:
        ignored = {os.getpid(),*[p.pid for p in psutil.Process().parents()]}
        for process in psutil.process_iter(['pid','cmdline']):
            command = process.info['cmdline'] or []
            if (process.pid not in ignored and command and 'python' in Path(command[0]).name
                    and any(Path(token).name in scripts for token in command[1:])):
                raise ValueError(f'conflicting process {process.pid}: {command}')
    except (OSError,psutil.Error) as exc:
        if strict:raise ValueError(f'production preflight process enumeration unavailable: {exc}') from exc
        return {'status':'unavailable','error':str(exc)}
    return {'status':'clear','scripts':sorted(scripts)}


def mount_ready():
    path=Path('/Volumes/Market-Data')
    return path.is_dir() and path.is_mount()


def paired_readable(ds):
    import os
    from orderflow_system.data.paired_dataset import pinned_root
    return (ds.directory.is_dir() and os.access(ds.directory,os.R_OK|os.X_OK)
            and os.access(ds.directory/'dataset_manifest.json',os.R_OK)
            and ds.root_hash==pinned_root(ds.dataset_id))


def destination_filesystem(path):
    path=Path(path).resolve()
    while not path.exists():path=path.parent
    return path


def destination_free_bytes(path):
    import shutil
    return shutil.disk_usage(destination_filesystem(path)).free


def production_preflight(ds,destination,state,estimated_cache_bytes):
    """Read only. Every failure precedes approval consumption and destination writes."""
    if not mount_ready():raise ValueError('production preflight: Market-Data mount missing')
    if not paired_readable(ds):raise ValueError('production preflight: registered paired root unreadable or unregistered')
    required=max(100*1024**3,2*estimated_cache_bytes)
    free=destination_free_bytes(destination)
    if free<required:raise ValueError(f'production preflight: insufficient disk: {free} < {required}')
    processes=assert_no_conflicts(strict=True)
    protected=protected_state()
    baseline=owner_protected_state()
    if protected!=baseline:raise ValueError('production preflight: protected set differs from owner baseline')
    if source_state(state['base'])!=state:raise ValueError('production preflight: source_state changed')
    from scripts.phase4_resources import require_memory_plan
    memory=require_memory_plan([{'mapped_files':{}}])
    return {'mode':'production','mount':'/Volumes/Market-Data','mount_ready':True,
        'paired_root':str(ds.directory),'paired_readable':True,
        'destination_filesystem':str(destination_filesystem(destination)),
        'free_disk_bytes':free,'required_disk_bytes':required,'estimated_cache_bytes':estimated_cache_bytes,
        'processes':processes,'protected_state':protected,'source_state':state,'memory':memory}


def owner_protected_state():
    path=ROOT/'output/phase4/build/protected-extended.json'
    if not path.is_file():raise ValueError('production preflight: extended owner protected baseline missing')
    return json.loads(path.read_text())
