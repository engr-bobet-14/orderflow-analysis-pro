"""Read host-owned approvals and log their use; never create approval records."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import stat
import sys

from scripts.phase4_policy import ROOT, canonical_bytes, source_state

PLAN_SHA256 = '30bde18198de534eaad6b5d47cff0fdbe89f706fe6350093c25b31726ddee258'
PLAN_PATH = ROOT/'docs/plans/phase4-optuna-trade-signals-plan.md'
GATES = ('U','FH1','FH2','FH3','FH4','R')


def approval_root():
    return Path(pwd.getpwuid(os.getuid()).pw_dir)/'.orderflow-approvals/phase4'


def trusted_path(path, *, directory=False):
    if path.is_symlink() or not path.exists():
        raise ValueError('host approval record/directory missing or symlinked')
    info=path.stat()
    if info.st_uid!=os.getuid() or info.st_mode & 0o022:
        raise ValueError('approval paths must be owned by current uid and not group/world-writable')
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError('invalid approval path type')


def consume_approval(approval):
    """Full builds are single-use; campaign admissions append one audit line per use."""
    root=approval_root()
    trusted_path(root.parent,directory=True)
    trusted_path(root,directory=True)
    trusted_path(root/(approval['sha256']+'.json'))
    path=root/(approval['sha256']+'.consumed')
    single=approval['record']['scope']=='full_history_tfc'
    if path.exists() or path.is_symlink():
        trusted_path(path)
    flags=os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW|(os.O_EXCL if single else os.O_APPEND)
    try:
        fd=os.open(path,flags,0o600)
    except FileExistsError as exc:
        raise ValueError('full-history approval already consumed') from exc
    with os.fdopen(fd,'ab') as handle:
        handle.write(canonical_bytes({'approval_sha256':approval['sha256'],
            'command_argv':approval['record']['command_argv'],'pid':os.getpid(),
            'used_utc':datetime.now(timezone.utc).isoformat()}))
        handle.flush()
        os.fsync(handle.fileno())


def command_argv(argv=None):
    """The selector is excluded to avoid embedding a record's hash in its own bytes."""
    values = list(argv if argv is not None else getattr(sys,'orig_argv',[sys.executable,*sys.argv]))
    result=[]
    i=0
    while i < len(values):
        if values[i]=='--owner-approved':
            if i+1 == len(values):
                raise ValueError('approval selector requires a record hash')
            i+=2
        elif values[i].startswith('--owner-approved='):
            i+=1
        else:
            result.append(values[i]);i+=1
    return result


def _state(record,campaign_sha256=None):
    if hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()!=PLAN_SHA256 or record.get('plan_sha256')!=PLAN_SHA256:
        raise ValueError('approval plan hash mismatch')
    current=source_state()
    if record.get('source_state')!=current or record.get('base_commit')!=current['base']:
        raise ValueError('approval source state/base commit mismatch')
    if campaign_sha256 is not None and record.get('campaign_sha256')!=campaign_sha256:
        raise ValueError('approval campaign hash mismatch')
    try:
        created=datetime.fromisoformat(record['created_utc'].replace('Z','+00:00'))
        if created.tzinfo is None or created.utcoffset().total_seconds()!=0:
            raise ValueError('not UTC')
    except (KeyError,TypeError,ValueError,AttributeError) as exc:
        raise ValueError('approval created_utc must be a UTC timestamp') from exc


def verify_approval(scope,selector,campaign_sha256=None,argv=None):
    if not isinstance(selector,str) or not re.fullmatch('[0-9a-f]{64}',selector):
        raise ValueError('host approval requires a content-addressed record hash')
    root=approval_root()
    path=root/(selector+'.json')
    trusted_path(root.parent,directory=True)
    trusted_path(root,directory=True)
    trusted_path(path)
    raw=path.read_bytes()
    try:
        record=json.loads(raw)
    except (ValueError,UnicodeError) as exc:
        raise ValueError('invalid host approval record') from exc
    if not isinstance(record,dict) or canonical_bytes(record)!=raw or hashlib.sha256(raw).hexdigest()!=selector:
        raise ValueError('approval name/content hash mismatch')
    fields={'scope','plan_sha256','command_argv','base_commit','source_state','created_utc'}
    if scope=='campaign':
        fields.add('campaign_sha256')
    if set(record)!=fields or record.get('scope')!=scope or scope not in ('full_history_tfc','campaign'):
        raise ValueError('approval scope/fields mismatch')
    _state(record,campaign_sha256)
    if record['command_argv']!=command_argv(argv):
        raise ValueError('approval command argv mismatch')
    if scope=='full_history_tfc' and ((root/(selector+'.consumed')).exists() or (root/(selector+'.consumed')).is_symlink()):
        raise ValueError('full-history approval already consumed')
    return {'sha256':selector,'record':record}


def require_campaign_approval(selector,campaign_sha256,argv=None):
    approval=verify_approval('campaign',selector,campaign_sha256,argv)
    path=approval_root()/f'gates-{campaign_sha256}.json'
    trusted_path(path)
    raw=path.read_bytes()
    try:
        gates=json.loads(raw)
    except (ValueError,UnicodeError) as exc:
        raise ValueError('invalid host campaign gates record') from exc
    if not isinstance(gates,dict):
        raise ValueError('invalid host campaign gates record')
    _state(gates,campaign_sha256)
    if any(gates.get(key)!='pass' for key in GATES):
        raise ValueError('campaign requires host U, FH1-FH4 and inspection gates')
    consume_approval(approval)
    return {**approval,'gates_sha256':hashlib.sha256(raw).hexdigest()}


def approval_for_command(campaign_sha256,argv):
    """A driver may use only a matching host record for each child, never mint one."""
    matches=[]
    for path in approval_root().glob('*.json'):
        if not re.fullmatch('[0-9a-f]{64}',path.stem):
            continue
        try:
            verify_approval('campaign',path.stem,campaign_sha256,argv)
        except ValueError:
            continue
        matches.append(path.stem)
    if len(matches)!=1:
        raise ValueError('exactly one host approval record is required for this child command')
    return matches[0]
