"""Phase 4 private/RSS accounting and admission budgets; no data or study writes."""
from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
import re
import subprocess
import sys

GiB=1024**3
PRIVATE_BUDGET=4*GiB
TOTAL_BUDGET=24*GiB
MIN_AVAILABLE=8*GiB


def process_memory(pid=None):
    """macOS rusage_info_v4, including the kernel's lifetime private-footprint peak."""
    pid=os.getpid() if pid is None else pid
    if sys.platform=='darwin':
        # Layout from the installed macOS SDK sys/resource.h, RUSAGE_INFO_V4.
        names=('user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size '
               'phys_footprint proc_start_abstime proc_exit_abstime child_user_time child_system_time '
               'child_pkg_idle_wkups child_interrupt_wkups child_pageins child_elapsed_abstime '
               'diskio_bytesread diskio_byteswritten cpu_time_qos_default cpu_time_qos_maintenance '
               'cpu_time_qos_background cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated '
               'cpu_time_qos_user_interactive billed_system_time serviced_system_time logical_writes '
               'lifetime_max_phys_footprint instructions cycles billed_energy serviced_energy '
               'interval_max_phys_footprint runnable_time').split()
        class Usage(ctypes.Structure):
            _fields_=[('uuid',ctypes.c_ubyte*16)]+[(name,ctypes.c_uint64) for name in names]
        try:
            lib=ctypes.CDLL(ctypes.util.find_library('proc') or '/usr/lib/libproc.dylib',use_errno=True)
            call=lib.proc_pid_rusage
            call.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];call.restype=ctypes.c_int
            value=Usage()
            if call(pid,4,ctypes.byref(value))!=0:
                raise OSError(ctypes.get_errno(),'proc_pid_rusage unavailable')
            return {'rss_bytes':value.resident_size,'private_bytes':value.phys_footprint,
                    'peak_private_bytes':max(value.phys_footprint,value.lifetime_max_phys_footprint),
                    'private_metric':'proc_pid_rusage RUSAGE_INFO_V4 phys_footprint','private_is_fallback':False}
        except (OSError,AttributeError):
            pass
    import psutil
    process=psutil.Process(pid)
    rss=process.memory_info().rss
    if sys.platform!='darwin':
        try:
            private=process.memory_full_info().uss
            return {'rss_bytes':rss,'private_bytes':private,'peak_private_bytes':private,
                    'private_metric':'psutil USS (sampled)','private_is_fallback':False}
        except (psutil.Error,AttributeError):
            pass
    # Missing private measurement fails closed; RSS remains a separate strict bound.
    return {'rss_bytes':rss,'private_bytes':None,'peak_private_bytes':None,
            'private_metric':'RSS only; private measurement unavailable','private_is_fallback':True}


def memory_bounds(sample,mapped_bytes=0):
    rss=sample.get('peak_rss_bytes',sample.get('rss_bytes',0))
    private=sample.get('peak_private_bytes',sample.get('private_bytes'))
    private_ok=private is not None and not sample.get('private_is_fallback',False) and private<=PRIVATE_BUDGET
    return {'pass':bool(rss<=PRIVATE_BUDGET and private_ok),'rss_bytes':rss,'private_bytes':private,
            'mapped_cache_bytes':mapped_bytes,'mapped_rss_credit_bytes':0,'rss_limit_bytes':PRIVATE_BUDGET,
            'private_limit_bytes':PRIVATE_BUDGET,
            'private_measurement_available':private is not None}


def enforce_memory(sample,mapped_bytes=0):
    result=memory_bounds(sample,mapped_bytes)
    if not result['pass']:
        raise ValueError(f'Phase 4 process memory bounds exceeded or unavailable: {result}')
    return result


def current_memory():
    import resource
    result=process_memory()
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['peak_rss_bytes']=int(peak if sys.platform=='darwin' else peak*1024)
    return result


def enforce_current(mapped_bytes=0):
    return enforce_memory(current_memory(),mapped_bytes)


def mapped_cache_files(directory):
    """Read-only TFC arrays and bars, counted once per mapped file, rounded to pages."""
    import json
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_bytes())
    page=os.sysconf('SC_PAGE_SIZE')
    return {str((directory/name).resolve()):((directory/name).stat().st_size+page-1)//page*page
            for name in manifest['files']}


def parse_vm_stat(text):
    match=re.search(r'page size of (\d+) bytes',text)
    if not match:raise ValueError('vm_stat page size missing')
    values=[]
    for label in ('free','inactive','speculative'):
        field=re.search(r'Pages '+label+r':\s*(\d+)\.',text)
        if not field:raise ValueError(f'vm_stat Pages {label} missing')
        values.append(int(field.group(1)))
    return int(match.group(1))*sum(values)


def available_memory():
    if sys.platform=='darwin':
        text=subprocess.check_output(['/usr/bin/vm_stat'],text=True,stderr=subprocess.PIPE)
        return parse_vm_stat(text),'vm_stat free+inactive+speculative'
    import psutil
    return psutil.virtual_memory().available,'psutil available (non-macOS fallback)'


def memory_plan(plans,available_bytes=None):
    mapped={}
    for plan in plans:
        for path,size in plan.get('mapped_files',{}).items():
            key=str(Path(path).resolve());mapped[key]=max(mapped.get(key,0),size)
    private=sum(p.get('private_budget_bytes',PRIVATE_BUDGET) for p in plans)
    error=None;source='provided'
    if available_bytes is None:
        try:available_bytes,source=available_memory()
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            error=str(exc)
    planned=private+sum(mapped.values())
    return {'pass':bool(planned<=TOTAL_BUDGET and available_bytes is not None and available_bytes>=MIN_AVAILABLE),
        'processes':len(plans),'planned_private_bytes':private,'unique_mapped_bytes':sum(mapped.values()),
        'planned_bytes':planned,'total_limit_bytes':TOTAL_BUDGET,'available_bytes':available_bytes,
        'minimum_available_bytes':MIN_AVAILABLE,'available_source':source,'error':error,'mapped_files':mapped}


class MemoryAdmissionError(ValueError):
    def __init__(self,report):
        self.report=report
        super().__init__(f'Phase 4 total-memory admission refused: {report}')


def require_memory_plan(plans):
    result=memory_plan(plans)
    if not result['pass']:raise MemoryAdmissionError(result)
    return result
