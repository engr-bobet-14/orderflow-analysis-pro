"""Amendment A2: S1-only owner exception for the inspection gate R (never recorded as R = pass)."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts import phase4_approvals as approvals
from scripts.phase4_policy import PhaseLayout, canonical_bytes

STATE = {'base': 'b' * 40, 'sha256': 'c' * 64, 'dirty': False}
CAMPAIGN = 'd' * 64


def layout_for(tmp_path, phase):
    base = tmp_path / 'approvals';base.mkdir(mode=0o700, parents=True, exist_ok=True)
    layout = PhaseLayout(phase, root=tmp_path, approvals_base=base)
    layout.approval_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return layout


@pytest.fixture(autouse=True)
def pinned(tmp_path, monkeypatch):
    plan = tmp_path / 'plan.md';plan.write_bytes(b'plan rev 13')
    policy = tmp_path / 'a2.md';policy.write_bytes(b'amendment A2')
    a1 = tmp_path / 'a1.md';a1.write_bytes(b'amendment A1')
    monkeypatch.setattr(approvals, 'PHASE4B_A1_PATH', a1)
    monkeypatch.setattr(approvals, 'PHASE4B_PLAN_PATH', plan)
    monkeypatch.setattr(approvals, 'PHASE4B_PLAN_SHA256', hashlib.sha256(b'plan rev 13').hexdigest())
    monkeypatch.setattr(approvals, 'PHASE4B_S1_EXCEPTION_PATH', policy)
    monkeypatch.setattr(approvals, 'source_state', lambda: dict(STATE))


def proof_content(role, **changes):
    from scripts import phase4b_proof as proof
    if role == 'unit':
        value = {'role': 'unit', 'argv': ['python', *proof.proof_argv('unit')[1:]],
                 'passed': 1747, 'failed': 0, 'errors': 0, 'skipped': 0, 'require_host': True,
                 'returncode': 0, 'collected_count': 1747, 'summary': '1747 passed, 29 deselected in 2s'}
    elif role == 'real_data':
        collected = [f'tests/test_x.py::t{i}' for i in range(proof.EXPECTED_COLLECTED)]
        value = {'role': 'real_data', 'argv': ['python', '-m', 'pytest', '-q', *proof.real_data_selection()],
                 'collected': collected, 'horizon_utc': proof.HORIZON_UTC,
                 'amendment_sha256': hashlib.sha256(b'amendment A1').hexdigest(),
                 'passed': proof.EXPECTED_COLLECTED, 'failed': 0, 'errors': 0, 'skipped': 0, 'returncode': 0,
                 'summary': f'{proof.EXPECTED_COLLECTED} passed, 1801 deselected in 9s'}
    else:
        return role.encode()
    value.update(changes)
    return json.dumps(value).encode()


def artifact(tmp_path, role, content=None):
    path = tmp_path / 'proof' / f'{role}.json';path.parent.mkdir(exist_ok=True)
    path.write_bytes(content or proof_content(role))
    return {'role': role, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def exception_record(interim, **changes):
    record = {'kind': 'inspection_exception', 'phase': 'phase4b-s1', 'stage': 'S1', 'campaign_sha256': CAMPAIGN,
              'plan_sha256': approvals.PHASE4B_PLAN_SHA256, 'source_state': dict(STATE),
              'amendment_sha256': hashlib.sha256(b'amendment A2').hexdigest(),
              'interim_review_sha256': interim['sha256'],
              'owner_statement': 'S1 calibration may use the Superpowers interim review; R is not passed.',
              'created_utc': '2026-11-08T06:00:00Z'}
    record.update(changes)
    return record


def write_gates(layout, r, artifacts, **extra):
    gates = {'U': 'pass', 'AC1': 'pass', 'AC7': 'pass', 'R': r, 'plan_sha256': approvals.PHASE4B_PLAN_SHA256,
             'campaign_sha256': CAMPAIGN, 'source_state': dict(STATE), 'artifacts': artifacts, **extra}
    path = layout.approval_root / f'gates-{CAMPAIGN}.json';path.write_bytes(canonical_bytes(gates))
    os.chmod(path, 0o600)
    return gates


def exception_gates(tmp_path, layout, **record_changes):
    base = [artifact(tmp_path, r) for r in ('unit', 'digests', 'protected', 'real_data')]
    interim = artifact(tmp_path, 'interim_review')
    path = approvals.install_inspection_exception(layout, exception_record(interim, **record_changes))
    owner = {'role': 'owner_exception', 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return base, interim, owner


def test_a2_s1_exception_accepted_and_r_stays_not_pass(tmp_path):
    layout = layout_for(tmp_path, 'phase4b-s1')
    base, interim, owner = exception_gates(tmp_path, layout)
    gates = write_gates(layout, approvals.S1_EXCEPTION_R, [*base, interim, owner])
    assert approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S1')
    assert gates['R'] == 'owner_exception_interim_review' != 'pass'


@pytest.mark.parametrize('phase', ['phase4b'])
def test_a2_exception_refused_for_s2_and_s3_manifest(tmp_path, phase):
    layout = layout_for(tmp_path, phase)
    with pytest.raises(ValueError, match='phase4b-s1'):
        approvals.install_inspection_exception(layout, exception_record(artifact(tmp_path, 'interim_review'),
                                                                        phase=phase))
    s1 = layout_for(tmp_path / 's1', 'phase4b-s1')
    base, interim, owner = exception_gates(tmp_path, s1)
    write_gates(layout, approvals.S1_EXCEPTION_R, [*base, interim, owner])
    with pytest.raises(ValueError, match='inspection gates'):
        approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S1')


def test_a2_r_pass_path_is_unchanged(tmp_path):
    layout = layout_for(tmp_path, 'phase4b')
    roles = ('unit', 'digests', 'protected', 'real_data', 'inspection')
    write_gates(layout, 'pass', [artifact(tmp_path, r) for r in roles])
    assert approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S1')
    write_gates(layout, 'pass', [artifact(tmp_path, r) for r in roles[:4]] + [artifact(tmp_path, 'interim_review')])
    with pytest.raises(ValueError, match='incomplete'):
        approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S1')


def test_a2_r_pass_cannot_be_backed_by_the_interim_exception(tmp_path):
    layout = layout_for(tmp_path, 'phase4b-s1')
    base, interim, owner = exception_gates(tmp_path, layout)
    write_gates(layout, 'pass', [*base, interim, owner])
    with pytest.raises(ValueError, match='incomplete'):
        approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S1')


@pytest.mark.parametrize('value', ['interim', 'waived', '', None])
def test_a2_any_other_r_value_is_refused(tmp_path, value):
    layout = layout_for(tmp_path, 'phase4b-s1')
    base, interim, owner = exception_gates(tmp_path, layout)
    write_gates(layout, value, [*base, interim, owner])
    with pytest.raises(ValueError, match='inspection gates'):
        approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S1')


@pytest.mark.parametrize('field,value', [
    ('campaign_sha256', 'e' * 64), ('source_state', {**STATE, 'base': 'f' * 40}), ('stage', 'S2t'),
    ('plan_sha256', '0' * 64), ('amendment_sha256', '1' * 64), ('interim_review_sha256', '2' * 64)])
def test_a2_exception_record_binding_mismatch_refused(tmp_path, field, value):
    layout = layout_for(tmp_path, 'phase4b-s1')
    base = [artifact(tmp_path, r) for r in ('unit', 'digests', 'protected', 'real_data')]
    interim = artifact(tmp_path, 'interim_review')
    record = exception_record(interim, **{field: value})
    if field == 'stage':
        with pytest.raises(ValueError, match='S1'):
            approvals.install_inspection_exception(layout, record)
        return
    path = approvals.install_inspection_exception(layout, record)
    owner = {'role': 'owner_exception', 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    write_gates(layout, approvals.S1_EXCEPTION_R, [*base, interim, owner])
    with pytest.raises(ValueError, match='exception'):
        approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S1')


def test_a2_exception_record_must_be_private_canonical_and_in_its_root(tmp_path):
    layout = layout_for(tmp_path, 'phase4b-s1')
    base, interim, owner = exception_gates(tmp_path, layout)
    path = Path(owner['path'])
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        approvals.install_inspection_exception(layout, json.loads(path.read_bytes()))
    os.chmod(path, 0o644)
    write_gates(layout, approvals.S1_EXCEPTION_R, [*base, interim, owner])
    with pytest.raises(ValueError, match='exception'):
        approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S1')
    os.chmod(path, 0o600)
    stray = tmp_path / 'proof' / path.name;stray.write_bytes(path.read_bytes());os.chmod(stray, 0o600)
    write_gates(layout, approvals.S1_EXCEPTION_R, [*base, interim, {**owner, 'path': str(stray)}])
    with pytest.raises(ValueError, match='exception'):
        approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S1')


def test_a2_policy_document_must_match_the_record(tmp_path, monkeypatch):
    layout = layout_for(tmp_path, 'phase4b-s1')
    base, interim, owner = exception_gates(tmp_path, layout)
    write_gates(layout, approvals.S1_EXCEPTION_R, [*base, interim, owner])
    approvals.PHASE4B_S1_EXCEPTION_PATH.write_bytes(b'edited amendment')
    with pytest.raises(ValueError, match='exception'):
        approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S1')


def test_a2_other_gates_still_required_under_the_exception(tmp_path):
    layout = layout_for(tmp_path, 'phase4b-s1')
    base, interim, owner = exception_gates(tmp_path, layout)
    for gate in ('U', 'AC1', 'AC7'):
        write_gates(layout, approvals.S1_EXCEPTION_R, [*base, interim, owner], **{gate: 'fail'})
        with pytest.raises(ValueError, match='inspection gates'):
            approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S1')
    write_gates(layout, approvals.S1_EXCEPTION_R, [*base, owner])
    with pytest.raises(ValueError, match='incomplete'):
        approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S1')


# ---- Review fixes (interim review of A1/A2).
def test_a2_exception_requires_stage_s1_explicitly(tmp_path):
    layout = layout_for(tmp_path, 'phase4b-s1')
    base, interim, owner = exception_gates(tmp_path, layout)
    write_gates(layout, approvals.S1_EXCEPTION_R, [*base, interim, owner])
    for stage in (None, 'S2t', 'S3'):
        with pytest.raises(ValueError, match='inspection gates'):
            approvals.verify_phase4b_gates(layout, CAMPAIGN, stage)


@pytest.mark.parametrize('role,changes,match', [
    ('real_data', {'argv': ['python', '-m', 'pytest', '-q', '-m', 'real_data']}, 'A1 command'),
    ('real_data', {'failed': 1}, 'failed'),
    ('real_data', {'skipped': 1, 'passed': 17}, 'failed'),
    ('real_data', {'horizon_utc': '2023-05-09T19:14:04.024Z'}, 'horizon'),
    ('real_data', {'amendment_sha256': '3' * 64}, 'amendment'),
    ('real_data', {'collected': ['tests/test_real_dataset.py::test_pipeline_on_real_bars[x]'] * 18}, 'collected'),
    ('unit', {'argv': ['python', '-m', 'pytest', '-q']}, 'not real_data'),
    ('unit', {'argv': ['python', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-m', 'not real_data',
                       '-m', 'real_data']}, 'not real_data'),
    ('unit', {'failed': 1}, 'unit proof'),
    ('unit', {'errors': 1}, 'unit proof'),
    # A1a: the gate itself requires zero skipped unit tests and the mandatory host-input flag.
    ('unit', {'skipped': 1}, 'skipped'),
    ('unit', {'skipped': None}, 'skipped'),
    ('unit', {'require_host': False}, 'PHASE4B_REQUIRE_HOST'),
    ('unit', {'require_host': 'true'}, 'PHASE4B_REQUIRE_HOST'),
    ('unit', {'require_host': None}, 'PHASE4B_REQUIRE_HOST'),
    # A1b: successful exit, every collected test passed, and well-formed integer results.
    ('unit', {'returncode': 1}, 'exit'),
    ('unit', {'returncode': None}, 'malformed'),
    ('unit', {'returncode': False}, 'malformed'),
    ('unit', {'passed': 1746, 'summary': '1746 passed in 2s'}, 'every collected'),
    ('unit', {'collected_count': 0, 'passed': 0, 'summary': '0 passed'}, 'every collected'),
    ('unit', {'collected_count': '1747'}, 'malformed'),
    ('unit', {'passed': 1747.0}, 'malformed'),
    ('unit', {'skipped': False}, 'malformed'),
    ('unit', {'skipped': 0.0}, 'malformed'),
    ('unit', {'failed': '0'}, 'malformed'),
    ('unit', {'summary': 'Interrupted: 1 error during collection'}, 'summary'),
    ('unit', {'summary': None}, 'summary'),
    ('real_data', {'returncode': 2}, 'exit'),
    ('real_data', {'returncode': None}, 'malformed'),
    ('real_data', {'summary': 'no tests ran'}, 'summary')])
def test_a1_gates_check_proof_artifact_content(tmp_path, role, changes, match):
    layout = layout_for(tmp_path, 'phase4b')
    roles = ('unit', 'digests', 'protected', 'real_data', 'inspection')
    artifacts = [artifact(tmp_path, r, proof_content(r, **changes) if r == role else None) for r in roles]
    write_gates(layout, 'pass', artifacts)
    with pytest.raises(ValueError, match=match):
        approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S2t')


@pytest.mark.parametrize('role,missing', [('unit', 'returncode'), ('unit', 'collected_count'), ('unit', 'passed'),
                                          ('unit', 'summary'), ('real_data', 'returncode')])
def test_a1b_missing_results_are_refused(tmp_path, role, missing):
    layout = layout_for(tmp_path, 'phase4b')
    value = json.loads(proof_content(role))
    del value[missing]
    roles = ('unit', 'digests', 'protected', 'real_data', 'inspection')
    artifacts = [artifact(tmp_path, r, json.dumps(value).encode() if r == role else None) for r in roles]
    write_gates(layout, 'pass', artifacts)
    with pytest.raises(ValueError, match='malformed|summary'):
        approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S2t')


def test_a1b_failed_unit_exit_is_refused_on_the_a2_s1_exception_path(tmp_path):
    layout = layout_for(tmp_path, 'phase4b-s1')
    base, interim, owner = exception_gates(tmp_path, layout)
    base[0] = artifact(tmp_path, 'unit', proof_content('unit', returncode=1))
    write_gates(layout, approvals.S1_EXCEPTION_R, [*base, interim, owner])
    with pytest.raises(ValueError, match='exit'):
        approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S1')


@pytest.mark.parametrize('missing', ['skipped', 'require_host'])
def test_a1a_unit_artifact_without_skip_count_or_host_flag_is_refused(tmp_path, missing):
    # Artifacts written before A1a (no skip count / host flag) must not satisfy the gate.
    layout = layout_for(tmp_path, 'phase4b')
    value = json.loads(proof_content('unit'))
    del value[missing]
    roles = ('unit', 'digests', 'protected', 'real_data', 'inspection')
    artifacts = [artifact(tmp_path, r, json.dumps(value).encode() if r == 'unit' else None) for r in roles]
    write_gates(layout, 'pass', artifacts)
    with pytest.raises(ValueError, match='skipped|PHASE4B_REQUIRE_HOST'):
        approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S2t')


def test_a2_inspection_basis_is_returned_for_the_stage_record(tmp_path):
    layout = layout_for(tmp_path, 'phase4b-s1')
    base, interim, owner = exception_gates(tmp_path, layout)
    write_gates(layout, approvals.S1_EXCEPTION_R, [*base, interim, owner])
    basis = approvals.inspection_basis(layout, CAMPAIGN)
    assert basis == {'R': 'owner_exception_interim_review', 'owner_exception_sha256': owner['sha256'],
                     'interim_review_sha256': interim['sha256']}


def test_a2_exceptions_directory_must_be_private(tmp_path):
    layout = layout_for(tmp_path, 'phase4b-s1')
    base, interim, owner = exception_gates(tmp_path, layout)
    os.chmod(approvals.exception_root(layout), 0o755)
    write_gates(layout, approvals.S1_EXCEPTION_R, [*base, interim, owner])
    with pytest.raises(ValueError, match='exception'):
        approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S1')


@pytest.mark.parametrize('created', ['2026-11-08 06:00', '2026-11-08T06:00:00+08:00', 'yesterday'])
def test_a2_exception_created_utc_must_be_utc(tmp_path, created):
    layout = layout_for(tmp_path, 'phase4b-s1')
    with pytest.raises(ValueError, match='UTC'):
        approvals.install_inspection_exception(
            layout, exception_record(artifact(tmp_path, 'interim_review'), created_utc=created))


def test_a2_stage_record_and_log_show_the_inspection_basis(tmp_path, monkeypatch):
    from scripts import phase4b_campaign as driver, phase4_campaign as campaign, phase4_policy as policy
    from tests.test_phase4b_operations import fixture_inputs, benchmark_lane
    layout, snapshots, comparators, origin = fixture_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, 'source_state', lambda: {'base': 'base', 'sha256': 'source', 'dirty': False})
    monkeypatch.setattr(driver, 'source_state', campaign.source_state)
    basis = {'R': 'owner_exception_interim_review', 'owner_exception_sha256': 'e' * 64,
             'interim_review_sha256': 'f' * 64}
    monkeypatch.setattr(driver, 'require_phase4b_approval',
                        lambda *a, **kw: {'sha256': 'approval', 'gates_sha256': 'gates', 'inspection_basis': basis})
    monkeypatch.setattr(driver, 'preflight', lambda *a: None)
    s1 = policy.PhaseLayout('phase4b-s1', root=tmp_path, approvals_base=tmp_path / 'approvals')
    campaign.register(s1.manifest, snapshots, phase='phase4b-s1', layout=s1, comparators=comparators,
                      phase4_origin=origin)
    driver.run_stage(s1, 'S1', 'approval1',
                     operation=lambda *a: {'lanes': {bt: benchmark_lane() for bt in comparators}})
    lines = [json.loads(line) for line in (s1.output / 'campaign-log.jsonl').read_text().splitlines()]
    started = [line for line in lines if line.get('status') == 'started']
    assert started and started[0]['inspection_basis'] == basis


def test_a2_basis_comes_from_the_verified_bytes_without_a_second_read(tmp_path, monkeypatch):
    layout = layout_for(tmp_path, 'phase4b-s1')
    base, interim, owner = exception_gates(tmp_path, layout)
    write_gates(layout, approvals.S1_EXCEPTION_R, [*base, interim, owner])
    def second_read(*a):
        raise AssertionError('gates file read twice')
    monkeypatch.setattr(approvals, 'inspection_basis', second_read)
    sha, basis = approvals.verify_phase4b_gates(layout, CAMPAIGN, 'S1', with_basis=True)
    assert basis == {'R': 'owner_exception_interim_review', 'owner_exception_sha256': owner['sha256'],
                     'interim_review_sha256': interim['sha256']}
