from __future__ import annotations
from typing import Iterable

from .bank import WeightBank
from .version_coordinator import VersionCoordinator

class StreamLoader:
    def __init__(self, module_map, expected_tensors: int, timeout_s: float = 120.0):
        self.bank = WeightBank(module_map)
        self.vc = VersionCoordinator(expected_tensors=expected_tensors, timeout_s=timeout_s)

    def consume(self, records: Iterable):
        for rec in records:
            self.bank.install_fp(rec.fqn, rec.payload, rec.qmeta)
            completed = self.vc.note_arrival(rec.fqn, rec.version_id)
            if completed is not None:
                self.bank.make_live()
                self.vc.reset(completed)
                yield completed
