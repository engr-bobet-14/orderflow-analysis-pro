"""S1 preflight/post-flight verifies Phase 4's sealed protected baseline at its exact original paths.

Found after S1 registration: protected_hashes rebased every baseline path onto the 4b checkout, so the
27 /Volumes entries raised and the 15 information-driven-bars entries were checked at the wrong place.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts import phase4b_campaign as driver
from scripts import phase4_policy as policy


def host_or_skip(reason):
    # Proof runs set PHASE4B_REQUIRE_HOST=1 so a missing host input fails instead of skipping silently.
    if os.environ.get('PHASE4B_REQUIRE_HOST')=='1':
        pytest.fail('host input required: '+reason)
    pytest.skip(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def roots(tmp_path):
    base = tmp_path.resolve()
    result = tuple(base/name for name in ('Market-Data', 'orderflow', 'information-driven-bars'))
    files = {}
    for root, rel in zip(result, ('info-bars/BTCUSDT/a.parquet', 'output/optimize/studies.db',
                                  'output/info_bars_paired/hours.parquet')):
        path = root/rel
        path.parent.mkdir(parents=True)
        path.write_bytes(rel.encode())
        files[str(path)] = sha(rel.encode())
    return base, result, files


def baseline_for(base, entries):
    text = ''.join(f'{digest}  {name}\n' for name, digest in entries)
    path = base/'protected-before.sha256'
    path.write_text(text)
    return path, sha(text.encode())


def verify(path, roots, sealed, count):
    return driver.phase4_protected_files(path, roots, sealed_sha256=sealed, count=count)


def test_exact_original_paths_in_all_three_roots_are_verified_in_place(roots):
    base, authorized, files = roots
    path, sealed = baseline_for(base, files.items())
    assert verify(path, authorized, sealed, 3) == files


def test_baseline_must_be_the_sealed_phase4_baseline(roots):
    base, authorized, files = roots
    path, _ = baseline_for(base, files.items())
    with pytest.raises(ValueError, match='sealed'):
        verify(path, authorized, '0'*64, 3)


def test_entry_count_must_match_the_sealed_baseline(roots):
    base, authorized, files = roots
    path, sealed = baseline_for(base, files.items())
    with pytest.raises(ValueError, match='count'):
        verify(path, authorized, sealed, 46)


def test_changed_file_is_detected(roots):
    base, authorized, files = roots
    path, sealed = baseline_for(base, files.items())
    Path(next(iter(files))).write_bytes(b'changed')
    with pytest.raises(ValueError, match='protected file changed'):
        verify(path, authorized, sealed, 3)


def test_missing_file_is_refused(roots):
    base, authorized, files = roots
    path, sealed = baseline_for(base, files.items())
    os.unlink(next(iter(files)))
    with pytest.raises(FileNotFoundError):
        verify(path, authorized, sealed, 3)


@pytest.mark.parametrize('name', [
    '/elsewhere/output/optimize/studies.db',          # outside every authorized root
    'output/optimize/studies.db',                    # relative
])
def test_unknown_or_relative_path_is_refused(roots, name):
    base, authorized, files = roots
    path, sealed = baseline_for(base, [*files.items(), (name, '0'*64)])
    with pytest.raises(ValueError, match='authorized|exact'):
        verify(path, authorized, sealed, 4)


def test_traversal_out_of_an_authorized_root_is_refused(roots, tmp_path):
    base, authorized, files = roots
    outside = base/'outside.db'
    outside.write_bytes(b'x')
    name = str(authorized[1])+'/../outside.db'
    path, sealed = baseline_for(base, [*files.items(), (name, sha(b'x'))])
    with pytest.raises(ValueError, match='exact'):
        verify(path, authorized, sealed, 4)


def test_root_itself_and_non_normalized_paths_are_refused(roots):
    base, authorized, files = roots
    for name in (str(authorized[0]), str(authorized[1])+'//output/optimize/studies.db'):
        path, sealed = baseline_for(base, [*files.items(), (name, '0'*64)])
        with pytest.raises(ValueError, match='authorized|exact'):
            verify(path, authorized, sealed, 4)


def test_symlinked_file_is_refused(roots):
    base, authorized, files = roots
    target = base/'target.db'
    target.write_bytes(b'x')
    link = authorized[1]/'output/link.db'
    link.symlink_to(target)
    path, sealed = baseline_for(base, [*files.items(), (str(link), sha(b'x'))])
    with pytest.raises(ValueError, match='symlink'):
        verify(path, authorized, sealed, 4)


def test_symlinked_directory_component_is_refused(roots):
    base, authorized, files = roots
    real = base/'real'
    real.mkdir()
    (real/'f.db').write_bytes(b'x')
    (authorized[2]/'linked').symlink_to(real, target_is_directory=True)
    name = str(authorized[2]/'linked/f.db')
    path, sealed = baseline_for(base, [*files.items(), (name, sha(b'x'))])
    with pytest.raises(ValueError, match='symlink'):
        verify(path, authorized, sealed, 4)


def test_duplicate_entry_is_refused(roots):
    base, authorized, files = roots
    first = next(iter(files.items()))
    path, sealed = baseline_for(base, [*files.items(), first])
    with pytest.raises(ValueError, match='duplicate'):
        verify(path, authorized, sealed, 4)


def test_authorized_roots_are_market_data_main_checkout_and_sibling_bars_repo():
    market, main, bars = driver.protected_roots(driver.ROOT)
    assert market == Path('/Volumes/Market-Data')
    assert (main/'.git').is_dir()
    assert driver.ROOT == main or driver.ROOT.is_relative_to(main/'.claude/worktrees')
    assert bars == main.parent/'information-driven-bars'


def test_protected_hashes_uses_exact_paths_not_checkout_rebasing(roots, monkeypatch):
    from scripts import trade_feature_cache as tfc
    from scripts.phase4_benchmark import DATASET_ID
    base, authorized, files = roots
    checkout = base/'checkout'
    layout = policy.PhaseLayout('phase4b-s1', root=checkout, approvals_base=base/'approvals')
    baseline = checkout/'output/phase4/build/protected-before.sha256'
    baseline.parent.mkdir(parents=True)
    text = ''.join(f'{d}  {n}\n' for n, d in files.items())
    baseline.write_text(text)
    for name in tfc.BUILDER_FILES:
        p = checkout/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'frozen')
    complete = checkout/'output/optimize/trade_cache'/DATASET_ID/'COMPLETE.json'
    complete.parent.mkdir(parents=True)
    complete.write_bytes(b'{}')
    monkeypatch.setattr(driver, 'PHASE4_PROTECTED_BASELINE_SHA256', sha(text.encode()))
    monkeypatch.setattr(driver, 'PHASE4_PROTECTED_COUNT', 3)
    monkeypatch.setattr(driver, 'protected_roots', lambda root: authorized)
    snapshot = base/'snapshot.db'
    snapshot.write_bytes(b'snap')
    manifest = {'snapshots': [{'path': str(snapshot), 'sha256': sha(b'snap')}]}
    hashes = driver.protected_hashes(layout, manifest)
    assert {k: hashes[k] for k in files} == files
    assert str(complete) in hashes and hashes[str(snapshot)] == sha(b'snap')
    assert all(str((checkout/name).resolve()) in hashes for name in tfc.BUILDER_FILES)
    snapshot.write_bytes(b'changed')
    with pytest.raises(ValueError, match='registered historical snapshot changed'):
        driver.protected_hashes(layout, manifest)
    assert not any(Path(k).is_relative_to(checkout) and Path(k).name in ('a.parquet', 'hours.parquet')
                   for k in hashes)


def test_production_baseline_constants_match_the_staged_phase4_baseline():
    baseline = driver.ROOT/'output/phase4/build/protected-before.sha256'
    if not baseline.is_file():
        host_or_skip('Phase 4 protected baseline not staged in this checkout')
    assert sha(baseline.read_bytes()) == driver.PHASE4_PROTECTED_BASELINE_SHA256
    assert len([l for l in baseline.read_text().splitlines() if l.strip()]) == driver.PHASE4_PROTECTED_COUNT == 46


def test_actual_production_preflight_with_the_registered_s1_manifest():
    """Host regression: the real preflight and protected_hashes on the registered manifest (hash-only reads)."""
    layout = policy.PhaseLayout('phase4b-s1')
    if not policy.mount_ready() or not layout.manifest.is_file():
        host_or_skip('needs /Volumes/Market-Data and the registered S1 manifest')
    manifest = json.loads(layout.manifest.read_bytes())
    if manifest['source_state'] == policy.source_state():
        assert driver.read_manifest(layout)[0] == manifest      # full admission path: seal + source_state
    else:
        # Code changed since registration: the stage path must refuse it until S1 is re-registered.
        with pytest.raises(ValueError, match='source_state changed'):
            driver.read_manifest(layout)
    driver.preflight(layout, manifest)
    hashes = driver.protected_hashes(layout, manifest)
    baseline = layout.root/'output/phase4/build/protected-before.sha256'
    exact = {l.split(maxsplit=1)[1].strip(): l.split(maxsplit=1)[0]
             for l in baseline.read_text().splitlines() if l.strip()}
    assert {k: hashes[k] for k in exact} == exact
    market, main, bars = driver.protected_roots(layout.root)
    counts = [sum(Path(k).is_relative_to(r) for k in exact) for r in (market, bars)]
    assert counts == [27, 15]
    assert sum(Path(k).is_relative_to(main) and not Path(k).is_relative_to(main/'.claude') for k in exact) == 4
