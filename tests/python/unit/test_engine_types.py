from moe_infinity.engine.transfer_types import (
    TransferPriority,
    TransferRequest,
    TransferType,
)
from moe_infinity.engine.types import (
    Request,
    SamplingParams,
    SchedulerOutput,
    Sequence,
    SequenceStatus,
)


def test_request_default_status_waiting():
    req = Request(
        request_id="r1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
    )
    assert req.status == SequenceStatus.WAITING


def test_request_valid_transition_waiting_to_running():
    req = Request(
        request_id="r1",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(),
    )
    req.transition_to(SequenceStatus.RUNNING)
    assert req.status == SequenceStatus.RUNNING


def test_request_invalid_transition_finished_to_running_raises():
    req = Request(
        request_id="r1",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(),
        status=SequenceStatus.FINISHED_STOPPED,
    )
    try:
        req.transition_to(SequenceStatus.RUNNING)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_scheduler_output_creation():
    seq = Sequence(seq_id="s1", request_id="r1", token_ids=[1, 2])
    output = SchedulerOutput(
        scheduled_seqs=[seq],
        preempted_seqs=[],
        swapped_in_seqs=[],
        num_batched_tokens=2,
    )
    assert output.scheduled_seqs[0].seq_id == "s1"
    assert output.num_batched_tokens == 2


def test_transfer_priority_values():
    assert TransferPriority.URGENT.value == 0
    assert TransferPriority.NORMAL.value == 10


def test_transfer_request_creation():
    transfer = TransferRequest(
        transfer_id="t1",
        transfer_type=TransferType.KV_SWAP_IN,
        priority=TransferPriority.HIGH,
        source_device="cpu",
        target_device="cuda:0",
        device_id=0,
    )
    assert transfer.transfer_id == "t1"
    assert transfer.priority == TransferPriority.HIGH
    assert transfer.device_id == 0
