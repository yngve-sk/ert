import sys
import typing
from collections import defaultdict
from datetime import datetime
from typing import (
    Any,
    DefaultDict,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    get_args,
)

from pydantic import BaseModel
from qtpy.QtGui import QColor
from typing_extensions import TypedDict

from _ert.events import (
    EESnapshot,
    EESnapshotUpdate,
    EnsembleCancelled,
    EnsembleEvent,
    EnsembleFailed,
    EnsembleStarted,
    EnsembleSucceeded,
    Event,
    FMEvent,
    ForwardModelStepFailure,
    ForwardModelStepRunning,
    ForwardModelStepStart,
    ForwardModelStepSuccess,
    RealizationEvent,
    RealizationFailed,
    RealizationPending,
    RealizationRunning,
    RealizationSuccess,
    RealizationTimeout,
    RealizationUnknown,
    RealizationWaiting,
)
from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator import state

if sys.version_info < (3, 11):
    from backports.datetime_fromisoformat import MonkeyPatch  # type: ignore

    MonkeyPatch.patch_fromisoformat()


class UnsupportedOperationException(ValueError):
    pass


_FM_TYPE_EVENT_TO_STATUS = {
    RealizationWaiting: state.REALIZATION_STATE_WAITING,
    RealizationPending: state.REALIZATION_STATE_PENDING,
    RealizationRunning: state.REALIZATION_STATE_RUNNING,
    RealizationFailed: state.REALIZATION_STATE_FAILED,
    RealizationSuccess: state.REALIZATION_STATE_FINISHED,
    RealizationUnknown: state.REALIZATION_STATE_UNKNOWN,
    RealizationTimeout: state.REALIZATION_STATE_FAILED,
    ForwardModelStepStart: state.FORWARD_MODEL_STATE_START,
    ForwardModelStepRunning: state.FORWARD_MODEL_STATE_RUNNING,
    ForwardModelStepSuccess: state.FORWARD_MODEL_STATE_FINISHED,
    ForwardModelStepFailure: state.FORWARD_MODEL_STATE_FAILURE,
}

_ENSEMBLE_TYPE_EVENT_TO_STATUS = {
    EnsembleStarted: state.ENSEMBLE_STATE_STARTED,
    EnsembleSucceeded: state.ENSEMBLE_STATE_STOPPED,
    EnsembleCancelled: state.ENSEMBLE_STATE_CANCELLED,
    EnsembleFailed: state.ENSEMBLE_STATE_FAILED,
}


def convert_iso8601_to_datetime(
    timestamp: Union[datetime, str],
) -> datetime:
    if isinstance(timestamp, datetime):
        return timestamp

    return datetime.fromisoformat(timestamp)


RealId = str
FmStepId = str


class SnapshotMetadata(TypedDict, total=False):
    aggr_job_status_colors: DefaultDict[RealId, Dict[FmStepId, QColor]]
    real_status_colors: Dict[RealId, QColor]
    sorted_real_ids: List[RealId]
    sorted_forward_model_ids: DefaultDict[RealId, List[FmStepId]]


def _filter_nones(some_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in some_dict.items() if value is not None}


class Snapshot:
    """The snapshot class is how we communicate the state of the ensemble between ensemble_evaluator and monitors.
    We start with an empty snapshot and as realizations progress, we send smaller snapshots only
    containing the changes which are then merged into the initial snapshot. In case a connection
    is dropped, we can send the entire snapshot.
    """

    def __init__(self) -> None:
        self._realization_states: Dict[
            str,
            Dict[str, Union[bool, datetime, str, Dict[str, "ForwardModel"]]],
        ] = defaultdict(dict)
        """A shallow dictionary of realization states. The key is a string with
        realization number, pointing to a dict with keys active (bool),
        start_time (datetime), end_time (datetime), callback_status_message (str) and status (str).
        callback_status_message is the error message from the data internalization process at the end of the forward model,
        and status is the error status from the forward model steps"""

        self._forward_model_states: DefaultDict[Tuple[str, str], ForwardModel] = (
            defaultdict(ForwardModel)  # type: ignore
        )
        """A shallow dictionary of forward_model states. The key is a tuple of two
        strings with realization id and forward_model id, pointing to a ForwardModel."""

        self._ensemble_state: Optional[str] = None
        # TODO not sure about possible values at this point, as GUI hijacks this one as
        # well
        self._metadata = SnapshotMetadata(
            aggr_job_status_colors={},
            real_status_colors={},
            sorted_real_ids=[],
            sorted_forward_model_ids={},
        )

    @classmethod
    def from_nested_dict(cls, data: Mapping[Any, Any]) -> "Snapshot":
        snapshot = Snapshot()
        if data is None:
            return snapshot
        if "metadata" in data:
            snapshot._metadata = data["metadata"]
        if "status" in data:
            snapshot._ensemble_state = data["status"]
        for real_id, realization_data in data.get("reals", {}).items():
            snapshot._realization_states[real_id] = _filter_nones(
                {
                    "status": realization_data.get("status"),
                    "active": realization_data.get("active"),
                    "start_time": realization_data.get("start_time"),
                    "end_time": realization_data.get("end_time"),
                    "callback_status_message": realization_data.get(
                        "callback_status_message"
                    ),
                }
            )
            for forward_model_id, job in realization_data.get(
                "forward_models", {}
            ).items():
                forward_model_idx = (real_id, forward_model_id)
                snapshot._forward_model_states[forward_model_idx] = job

        return snapshot

    def merge_snapshot(self, other_snapshot: "Snapshot") -> "Snapshot":
        self._metadata.update(other_snapshot._metadata)
        if other_snapshot._ensemble_state is not None:
            self._ensemble_state = other_snapshot._ensemble_state
        for real_id, other_real_data in other_snapshot._realization_states.items():
            self._realization_states[real_id].update(other_real_data)
        for (
            forward_model_id,
            other_fm_data,
        ) in other_snapshot._forward_model_states.items():
            self._forward_model_states[forward_model_id].update(other_fm_data)
        return self

    def merge_metadata(self, metadata: SnapshotMetadata) -> None:
        self._metadata.update(metadata)

    def to_dict(self) -> Dict[str, Any]:
        """used to send snapshot updates - for thread safety, this method should not
        access the _snapshot property"""
        _dict: Dict[str, Any] = {}
        if self._metadata:
            _dict["metadata"] = self._metadata
        if self._ensemble_state:
            _dict["status"] = self._ensemble_state
        if self._realization_states:
            _dict["reals"] = dict(self._realization_states)

        for (real_id, fm_id), fm_values_dict in self._forward_model_states.items():
            if "reals" not in _dict:
                _dict["reals"] = {}
            if real_id not in _dict["reals"]:
                _dict["reals"][real_id] = {}
            if "forward_models" not in _dict["reals"][real_id]:
                _dict["reals"][real_id]["forward_models"] = {}

            _dict["reals"][real_id]["forward_models"][fm_id] = fm_values_dict

        return _dict

    @property
    def status(self) -> Optional[str]:
        return self._ensemble_state

    @property
    def metadata(self) -> SnapshotMetadata:
        return self._metadata

    def get_all_forward_models(
        self,
    ) -> Mapping[Tuple[str, str], "ForwardModel"]:
        return self._forward_model_states.copy()

    def get_forward_model_status_for_all_reals(
        self,
    ) -> Mapping[Tuple[str, str], str]:
        return {
            idx: forward_model_state["status"]
            for idx, forward_model_state in self._forward_model_states.items()
            if "status" in forward_model_state
            and forward_model_state["status"] is not None
        }

    @property
    def reals(self) -> Mapping[str, "RealizationSnapshot"]:
        return {
            real_id: RealizationSnapshot(**real_data)
            for real_id, real_data in self._realization_states.items()
        }

    def get_forward_models_for_real(self, real_id: str) -> Dict[str, "ForwardModel"]:
        return {
            fm_idx[1]: forward_model_data.copy()
            for fm_idx, forward_model_data in self._forward_model_states.items()
            if fm_idx[0] == real_id
        }

    def get_real(self, real_id: str) -> "RealizationSnapshot":
        return RealizationSnapshot(**self._realization_states[real_id])

    def get_job(self, real_id: str, forward_model_id: str) -> "ForwardModel":
        return self._forward_model_states[(real_id, forward_model_id)].copy()

    def get_successful_realizations(self) -> typing.List[int]:
        return [
            int(real_idx)
            for real_idx, real_data in self._realization_states.items()
            if real_data.get(ids.STATUS, "") == state.REALIZATION_STATE_FINISHED
        ]

    def aggregate_real_states(self) -> typing.Dict[str, int]:
        states: Dict[str, int] = defaultdict(int)
        for real in self._realization_states.values():
            status = real["status"]
            assert isinstance(status, str)
            states[status] += 1
        return states

    def data(self) -> Mapping[str, Any]:
        # The gui uses this
        return self.to_dict()

    def update_realization(
        self,
        real_id: str,
        status: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        callback_status_message: Optional[str] = None,
    ) -> "Snapshot":
        self._realization_states[real_id].update(
            _filter_nones(
                {
                    "status": status,
                    "start_time": start_time,
                    "end_time": end_time,
                    "callback_status_message": callback_status_message,
                }
            )
        )
        return self

    def update_from_event(
        self, event: Event, source_snapshot: Optional["Snapshot"] = None
    ) -> "Snapshot":
        e_type = type(event)
        timestamp = event.time

        if source_snapshot is None:
            source_snapshot = Snapshot()
        if e_type in get_args(RealizationEvent):
            event = cast(RealizationEvent, event)
            status = _FM_TYPE_EVENT_TO_STATUS[type(event)]
            start_time = None
            end_time = None
            callback_status_message = None

            if e_type is RealizationRunning:
                start_time = convert_iso8601_to_datetime(timestamp)
            elif e_type in {
                RealizationSuccess,
                RealizationFailed,
                RealizationTimeout,
            }:
                end_time = convert_iso8601_to_datetime(timestamp)
            if type(event) is RealizationFailed:
                callback_status_message = event.callback_status_message
            self.update_realization(
                event.real,
                status,
                start_time,
                end_time,
                callback_status_message,
            )

            if e_type is RealizationTimeout:
                for (
                    forward_model_id,
                    forward_model,
                ) in source_snapshot.get_forward_models_for_real(event.real).items():
                    if (
                        forward_model.get(ids.STATUS)
                        != state.FORWARD_MODEL_STATE_FINISHED
                    ):
                        fm_idx = (event.real, forward_model_id)
                        if fm_idx not in source_snapshot._forward_model_states:
                            self._forward_model_states[fm_idx] = ForwardModel()
                        self._forward_model_states[fm_idx].update(
                            ForwardModel(
                                status=state.FORWARD_MODEL_STATE_FAILURE,
                                end_time=end_time,
                                error="The run is cancelled due to "
                                "reaching MAX_RUNTIME",
                            )
                        )

        elif e_type in get_args(FMEvent):
            event = cast(FMEvent, event)
            status = _FM_TYPE_EVENT_TO_STATUS[type(event)]
            start_time = None
            end_time = None
            error = None
            if e_type is ForwardModelStepStart:
                start_time = convert_iso8601_to_datetime(timestamp)
            elif e_type in {ForwardModelStepSuccess, ForwardModelStepFailure}:
                end_time = convert_iso8601_to_datetime(timestamp)
                if type(event) is ForwardModelStepFailure:
                    error = event.error_msg if event.error_msg else ""
                else:
                    # Make sure error msg from previous failed run is replaced
                    error = ""

            fm = ForwardModel(
                **_filter_nones(  # type: ignore
                    {
                        ids.STATUS: status,
                        ids.INDEX: event.fm_step,
                        ids.START_TIME: start_time,
                        ids.END_TIME: end_time,
                        ids.ERROR: error,
                    }
                )
            )

            if type(event) is ForwardModelStepRunning:
                fm[ids.CURRENT_MEMORY_USAGE] = event.current_memory_usage
                fm[ids.MAX_MEMORY_USAGE] = event.max_memory_usage
            if type(event) is ForwardModelStepStart:
                fm[ids.STDOUT] = event.std_out
                fm[ids.STDERR] = event.std_err

            self.update_forward_model(
                event.real,
                event.fm_step,
                fm,
            )

        elif e_type in get_args(EnsembleEvent):
            event = cast(EnsembleEvent, event)
            self._ensemble_state = _ENSEMBLE_TYPE_EVENT_TO_STATUS[type(event)]
        elif type(event) is EESnapshotUpdate:
            self.merge_snapshot(Snapshot.from_nested_dict(event.snapshot))
        elif type(event) is EESnapshot:
            return Snapshot.from_nested_dict(event.snapshot)
        else:
            raise ValueError(f"Unknown type: {e_type}")
        return self

    def update_forward_model(
        self,
        real_id: str,
        forward_model_id: str,
        forward_model: "ForwardModel",
    ) -> "Snapshot":
        self._forward_model_states[(real_id, forward_model_id)].update(forward_model)
        return self


class ForwardModel(TypedDict, total=False):
    status: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    index: Optional[str]
    current_memory_usage: Optional[str]
    max_memory_usage: Optional[str]
    name: Optional[str]
    error: Optional[str]
    stdout: Optional[str]
    stderr: Optional[str]


class RealizationSnapshot(BaseModel):
    status: Optional[str] = None
    active: Optional[bool] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    forward_models: Dict[str, ForwardModel] = {}
    callback_status_message: Optional[str] = None


class SnapshotDict(BaseModel):
    status: Optional[str] = state.ENSEMBLE_STATE_UNKNOWN
    reals: Dict[str, RealizationSnapshot] = {}
    metadata: Dict[str, Any] = {}


class SnapshotBuilder(BaseModel):
    forward_models: Dict[str, ForwardModel] = {}
    metadata: Dict[str, Any] = {}

    def build(
        self,
        real_ids: Sequence[str],
        status: Optional[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        callback_status_message: Optional[str] = None,
    ) -> Snapshot:
        top = SnapshotDict(status=status, metadata=self.metadata)
        for r_id in real_ids:
            top.reals[r_id] = RealizationSnapshot(
                active=True,
                forward_models=self.forward_models,
                start_time=start_time,
                end_time=end_time,
                status=status,
                callback_status_message=callback_status_message,
            )
        return Snapshot.from_nested_dict(top.model_dump())

    def add_forward_model(
        self,
        forward_model_id: str,
        index: str,
        name: Optional[str],
        status: Optional[str],
        current_memory_usage: Optional[str] = None,
        max_memory_usage: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ) -> "SnapshotBuilder":
        self.forward_models[forward_model_id] = ForwardModel(
            **_filter_nones(  # type: ignore
                {
                    ids.STATUS: status,
                    ids.INDEX: index,
                    ids.START_TIME: start_time,
                    ids.END_TIME: end_time,
                    ids.NAME: name,
                    ids.STDOUT: stdout,
                    ids.STDERR: stderr,
                    ids.CURRENT_MEMORY_USAGE: current_memory_usage,
                    ids.MAX_MEMORY_USAGE: max_memory_usage,
                }
            )
        )
        return self
