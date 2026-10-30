"""Execution ledger shared by replay diagnostics, notebooks and validation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from orderflow_system.data.models import ExecutionSnapshot

if TYPE_CHECKING:
    from orderflow_system.pipeline import BarResult


@dataclass(frozen=True)
class LedgerRecord:
    action: str
    execution: ExecutionSnapshot

    @property
    def bar_seq(self) -> int:
        return self.execution.bar_seq


class TradeLedger:
    """Record actual fills in action order, retaining tied-time bars separately.

    Prices and trade state come exclusively from immutable execution snapshots;
    a later replacement entry cannot change a previously recorded close.
    """

    def __init__(self):
        self.records: list[LedgerRecord] = []
        self.entries: list[LedgerRecord] = []
        self.closes: list[LedgerRecord] = []
        self.by_seq: dict[int, list[LedgerRecord]] = defaultdict(list)

    def process_result(self, result: BarResult) -> list[LedgerRecord]:
        records = []
        for aggregate in result.aggregated:
            if aggregate.action not in ("enter", "stop", "target", "exit"):
                continue
            record = LedgerRecord(aggregate.action, aggregate.execution)
            self.records.append(record)
            self.by_seq[record.bar_seq].append(record)
            (self.entries if aggregate.action == "enter" else self.closes).append(record)
            records.append(record)
        return records
