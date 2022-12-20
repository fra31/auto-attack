import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Set
import warnings

import torch


@dataclass
class EvaluationState:
    _attacks_to_run: Set[str]
    path: Optional[Path] = None
    _run_attacks: Set[str] = field(default_factory=set)
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
        self._last_saved = datetime.now()
        d = asdict(self)
        if self.robust_flags is not None:
            d["_robust_flags"] = d["_robust_flags"].cpu().tolist()
        d["_run_attacks"] = list(self._run_attacks)
        with self.path.open("w", ) as f:
            json.dump(d, f, default=str)

    @classmethod
    def from_disk(cls, path: Path) -> "EvaluationState":
        with path.open("r") as f:
            d = json.load(f)
        d["_robust_flags"] = torch.tensor(d["_robust_flags"], dtype=torch.bool)
        d["path"] = Path(d["path"])
        if path != d["path"]:
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
    def run_attacks(self) -> Set[str]:
        return self._run_attacks

    def add_run_attack(self, attack: str) -> None:
        self._run_attacks.add(attack)
        self.to_disk()
        
    @property
    def attacks_to_run(self) -> Set[str]:
        return self._attacks_to_run
    
    @attacks_to_run.setter
    def attacks_to_run(self, _: Set[str]) -> None:
        raise ValueError("attacks_to_run cannot be set outside of the constructor")

    @property
    def clean_accuracy(self) -> float:
        return self._clean_accuracy

    @clean_accuracy.setter
    def clean_accuracy(self, accuracy) -> None:
        self._clean_accuracy = accuracy
        self.to_disk(force=True)

    @property
    def robust_accuracy(self) -> float:
        if self.robust_flags is None:
            raise ValueError("robust_flags is not set yet. Start the attack first.")
        if self.attacks_to_run - self.run_attacks:
            warnings.warn("You are checking `robust_accuracy` before all the attacks"
                          " have been run.")
        return self.robust_flags.float().mean().item()