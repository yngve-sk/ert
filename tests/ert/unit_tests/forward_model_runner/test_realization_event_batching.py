from datetime import datetime

from _ert.events import (
    ForwardModelStepChecksum,
    ForwardModelStepRunning,
    ForwardModelStepStart,
    ForwardModelStepSuccess,
    Id,
)
from _ert.forward_model_runner.reporting.event import EventBatcher


def test_that_event_max_memory_is_merged_correctly():
    batcher = EventBatcher()

    batcher.add_event(
        ForwardModelStepStart(
            fm_step="0", time=datetime.now(), ensemble="ens", real="real"
        )
    )
    batcher.add_event(
        ForwardModelStepRunning(
            fm_step="0",
            time=datetime.now(),
            ensemble="ens",
            real="real",
            cpu_seconds=1,
            current_memory_usage=10,
            max_memory_usage=10,
        )
    )
    batcher.add_event(
        ForwardModelStepRunning(
            fm_step="0",
            time=datetime.now(),
            ensemble="ens",
            real="real",
            cpu_seconds=1,
            current_memory_usage=10,
            max_memory_usage=4,
        )
    )
    batcher.add_event(
        ForwardModelStepRunning(
            fm_step="0",
            time=datetime.now(),
            ensemble="ens",
            real="real",
            cpu_seconds=2,
            current_memory_usage=6,
            max_memory_usage=15,
        )
    )
    batcher.add_event(
        ForwardModelStepRunning(
            fm_step="0",
            time=datetime.now(),
            ensemble="ens",
            real="real",
            cpu_seconds=3,
            current_memory_usage=2,
            max_memory_usage=5,
        )
    )
    batcher.add_event(
        ForwardModelStepSuccess(
            fm_step="0",
            time=datetime.now(),
            ensemble="ens",
            real="real",
            current_memory_usage=0,
        )
    )

    flushed = batcher.flush_pending_events()

    first, second, third = flushed
    assert first.event_type == Id.FORWARD_MODEL_STEP_START
    assert second.event_type == Id.FORWARD_MODEL_STEP_RUNNING
    assert second.current_memory_usage == 2
    assert second.max_memory_usage == 15
    assert third.event_type == Id.FORWARD_MODEL_STEP_SUCCESS
    assert third.current_memory_usage == 0

    flushed2 = batcher.flush_pending_events()
    assert flushed2 == []


def test_checksum_overrides_previous():
    batcher = EventBatcher()

    batcher.add_event(
        ForwardModelStepChecksum(
            time=datetime.now(),
            ensemble="ens",
            real="real",
            checksums={"foo": {"old": 1}},
        )
    )
    batcher.add_event(
        ForwardModelStepChecksum(
            time=datetime.now(),
            ensemble="ens",
            real="real",
            checksums={"foo": {"new": 2}},
        )
    )

    flushed = batcher.flush_pending_events()
    assert len(flushed) == 1
    assert isinstance(flushed[0], ForwardModelStepChecksum)
    assert flushed[0].checksums == {"foo": {"new": 2}}


def test_only_last_checksum_sent():
    batcher = EventBatcher()

    batcher.add_event(
        ForwardModelStepStart(
            time=datetime.now(), ensemble="ens", real="real", fm_step="1"
        )
    )
    batcher.add_event(
        ForwardModelStepChecksum(
            time=datetime.now(), ensemble="ens", real="real", checksums={"x": {"a": 1}}
        )
    )
    batcher.add_event(
        ForwardModelStepChecksum(
            time=datetime.now(), ensemble="ens", real="real", checksums={"x": {"b": 2}}
        )
    )

    flushed = batcher.flush_pending_events()
    assert flushed[0].event_type == Id.FORWARD_MODEL_STEP_CHECKSUM
    assert flushed[0].checksums == {"x": {"b": 2}}
