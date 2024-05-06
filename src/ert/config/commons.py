from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Sequence

import numpy as np
import numpy.typing as npt


@dataclass(eq=False)
class Refcase:
    start_date: datetime
    keys: List[str]
    dates: Sequence[datetime]
    values: npt.NDArray[Any]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Refcase):
            return False
        return bool(
            self.start_date == other.start_date
            and self.keys == other.keys
            and self.dates == other.dates
            and np.all(self.values == other.values)
        )

    @property
    def all_dates(self) -> List[datetime]:
        return [self.start_date] + list(self.dates)
