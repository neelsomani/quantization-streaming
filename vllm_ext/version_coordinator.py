from __future__ import annotations
from collections import defaultdict
from typing import Optional

class VersionCoordinator:
    """Tracks arrivals and triggers swap when all expected tensors for a version arrive.
    Single process demo, so no distributed barrier here.
    """
    def __init__(self, expected_tensors: int, timeout_s: float = 120.0):
        self.expected = expected_tensors
        self.timeout_s = timeout_s
        self.bitmap = defaultdict(set)  # version -> set of fqns

    def note_arrival(self, fqn: str, version_id: int) -> Optional[int]:
        s = self.bitmap[version_id]
        s.add(fqn)
        if len(s) >= self.expected:
            return version_id
        return None

    def reset(self, version_id: int) -> None:
        if version_id in self.bitmap:
            del self.bitmap[version_id]
