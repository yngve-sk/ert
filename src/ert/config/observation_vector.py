from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Union

from .general_observation import GenObservation
from .response_properties import ResponseTypes
from .summary_observation import SummaryObservation

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class ObsVector:
    observation_type: ResponseTypes
    observation_name: str
    response_name: str
    observations: Dict[Union[int, datetime], Union[GenObservation, SummaryObservation]]

    def __len__(self) -> int:
        return len(self.observations)

    def __iter__(self) -> Iterable[Union[SummaryObservation, GenObservation]]:
        """Iterate over active report steps; return node"""
        return iter(self.observations.values())
