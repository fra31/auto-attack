import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Set
import warnings

import torch


@dataclass
class EvaluationState:
    path: Optional[Path] = None
    _ran_attacks: Set[str] = field(default_factory=set)
    _robust_flags: Optional[torch.Tensor] = None
    _last_saved: datetime = datetime(1, 1, 1)
    _SAVE_TIMEOUT: int = 60
    _clean_accuracy: float = float("nan")

    def to_disk(self, force: bool = False) -> None:
        seconds_since_last_save = (datetime.now() -
                                   self._last_saved).total_seconds()
        if self.path is None or (seconds_since_last_save < self._SAVE_TIMEOUT
                                 and not force):
            return
        d = asdict(self)
        if self.robust_flags is not None:
            d["_robust_flags"] = d["_robust_flags"].cpu().tolist()
        d["_ran_attacks"] = list(self._ran_attacks)
        with self.path.open("w", ) as f:
            json.dump(d, f, default=str)

    @classmethod
    def from_disk(cls, path: Path) -> "EvaluationState":
        with path.open("r") as f:
            d = json.load(f)
        d["_robust_flags"] = torch.tensor(d["_robust_flags"], dtype=torch.bool)
        if path != Path(d["path"]):
            warnings.warn(
                UserWarning(
                    "The given path is different from the one found in the state file."
                ))
        d["_last_saved"] = datetime.fromisoformat(d["_last_saved"])
        return cls(**d)

    @property
    def robust_flags(self) -> Optional[torch.Tensor]:
        return self._robust_flags

    @robust_flags.setter
    def robust_flags(self, robust_flags: torch.Tensor) -> None:
        self._robust_flags = robust_flags
        self.to_disk(force=True)

    @property
    def ran_attacks(self) -> Set[str]:
        return self._ran_attacks

    def add_ran_attack(self, attack: str) -> None:
        self._ran_attacks.add(attack)
        self.to_disk()

    @property
    def clean_accuracy(self) -> float:
        return self._clean_accuracy

    @clean_accuracy.setter
    def clean_accuracy(self, accuracy) -> None:
        self._clean_accuracy = accuracy
        self.to_disk(force=True)
