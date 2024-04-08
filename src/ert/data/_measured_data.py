"""
Read-only API to fetch responses (a.k.a measurements) and
matching observations from internal ERT-storage.
The main goal is to facilitate data-analysis using scipy and similar tools,
instead of having to implement analysis-functionality into ERT using C/C++.
The API is typically meant used as part of workflows.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble


class ResponseError(Exception):
    pass


class MeasuredData:
    def __init__(
        self,
        ensemble: Ensemble,
        keys: Optional[List[str]] = None,
        index_lists: Optional[List[List[Union[int, datetime]]]] = None,
    ):
        if keys is None:
            keys = sorted(list(ensemble.experiment.observation_keys))
        if not keys:
            raise ObservationError("No observation keys provided")
        if index_lists is not None and len(index_lists) != len(keys):
            raise ValueError("index list must be same length as observations keys")

        self._set_data(ensemble.get_measured_data(keys).to_long_dataframe().T)
        self._set_data(self.filter_on_column_index(keys, index_lists))

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def _set_data(self, data: pd.DataFrame) -> None:
        expected_keys = {"OBS", "STD"}
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"Invalid type: {type(data)}, should be type: {pd.DataFrame}"
            )
        if not expected_keys.issubset(data.index):
            missing = expected_keys - set(data.index)
            raise ValueError(
                f"{expected_keys} should be present in DataFrame index, \
                missing: {missing}"
            )
        self._data = data

    def remove_failed_realizations(self) -> None:
        """Removes rows with no simulated data, leaving observations and
        standard deviations as-is."""
        pre_index = self.data.index
        post_index = list(self.data.dropna(axis=0, how="all").index)
        drop_index = set(pre_index) - set(post_index + ["STD", "OBS"])
        self._set_data(self.data.drop(index=drop_index))

    def get_simulated_data(self) -> pd.DataFrame:
        """Dimension of data is (number of responses x number of realizations)."""
        return self.data[~self.data.index.isin(["OBS", "STD"])]

    def remove_inactive_observations(self) -> None:
        """Removes columns with one or more NaN or inf values."""
        filtered_dataset = self.data.replace([np.inf, -np.inf], np.nan).dropna(
            axis="columns", how="any"
        )
        if filtered_dataset.empty:
            raise ValueError(
                "This operation results in an empty dataset "
                "(could be due to one or more failed realizations)"
            )
        self._set_data(filtered_dataset)

    def is_empty(self) -> bool:
        return bool(self.data.empty)

    def filter_ensemble_std(self, std_cutoff: float) -> None:
        """
        Filters on ensemble variation versus a user defined standard
        deviation cutoff. If there is not enough variation in the measurements
        the data point is removed.
        """
        ens_std = self.get_simulated_data().std()
        std_filter = ens_std <= std_cutoff
        self._set_data(self.data.drop(columns=std_filter[std_filter].index))

    def filter_ensemble_mean_obs(self, alpha: float) -> None:
        """
        Filters on distance between the observed data and the ensemble mean
        based on variation and a user defined alpha.
        """
        ens_mean = self.get_simulated_data().mean()
        ens_std = self.get_simulated_data().std()
        obs_values = self.data.loc["OBS"]
        obs_std = self.data.loc["STD"]

        mean_filter = abs(obs_values - ens_mean) > alpha * (ens_std + obs_std)

        self._set_data(self.data.drop(columns=mean_filter[mean_filter].index))

    def filter_on_column_index(
        self,
        obs_keys: List[str],
        index_lists: Optional[List[List[Union[int, datetime]]]] = None,
    ) -> pd.DataFrame:
        if index_lists is None or all(index_list is None for index_list in index_lists):
            return self.data
        names = self.data.columns.get_level_values(0)
        data_index = self.data.columns.get_level_values("key_index")
        cond = self._create_condition(names, data_index, obs_keys, index_lists)
        return self.data.iloc[:, cond]

    @staticmethod
    def _create_condition(
        names: pd.Index,
        data_index: pd.Index,
        obs_keys: List[str],
        index_lists: List[List[Union[int, datetime]]],
    ) -> "npt.NDArray[np.bool_]":
        conditions = []
        for obs_key, index_list in zip(obs_keys, index_lists):
            if index_list is not None:
                index_cond = [data_index == index for index in index_list]
                index_cond = np.logical_or.reduce(index_cond)
                conditions.append(np.logical_and(index_cond, (names == obs_key)))
        return np.logical_or.reduce(conditions)


class ObservationError(Exception):
    pass
