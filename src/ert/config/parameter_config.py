from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from ert.config._option_dict import option_dict
from ert.config.commons import CustomDict

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble


def parse_config(
    config: List[str], max_positionals: int
) -> Tuple[List[str], Dict[str, str]]:
    """
    This function is responsible for taking a config line and splitting it
    into positional arguments and named arguments in cases were the number
    of positional arguments vary.
    """
    offset = next(
        (i for i, val in enumerate(config) if len(val.split(":")) == 2), max_positionals
    )
    kwargs = option_dict(config, offset)
    args = config[:offset]
    return args, kwargs


@dataclasses.dataclass
class ParameterConfig(ABC):
    name: str
    forward_init: bool
    update: bool

    def sample_or_load(
        self,
        real_nr: int,
        random_seed: int,
        ensemble_size: int,
    ) -> xr.Dataset:
        return self.read_from_runpath(Path(), real_nr)

    @abstractmethod
    def __len__(self) -> int:
        """Number of parameters"""

    @abstractmethod
    def read_from_runpath(
        self,
        run_path: Path,
        real_nr: int,
    ) -> xr.Dataset:
        """
        This function is responsible for converting the parameter
        from the forward model to the internal ert format
        """

    @abstractmethod
    def write_to_runpath(
        self, run_path: Path, real_nr: int, ensemble: Ensemble
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        This function is responsible for converting the parameter
        from the internal ert format to the format the forward model
        expects
        """

    @abstractmethod
    def save_parameters(
        self,
        ensemble: Ensemble,
        group: str,
        realization: int,
        data: npt.NDArray[np.float_],
    ) -> None:
        """
        Save the parameter in internal storage for the given ensemble
        """

    @abstractmethod
    def load_parameters(
        self, ensemble: Ensemble, group: str, realizations: npt.NDArray[np.int_]
    ) -> Union[npt.NDArray[np.float_], xr.DataArray]:
        """
        Load the parameter from internal storage for the given ensemble
        """

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self, dict_factory=CustomDict)
        data["_ert_kind"] = self.__class__.__name__
        return data

    def save_experiment_data(  # noqa: B027
        self,
        experiment_path: Path,
    ) -> None:
        pass
