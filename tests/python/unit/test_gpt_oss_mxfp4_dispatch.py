import pytest
import torch


@pytest.mark.gpu
def test_native_mxfp4_gate_up_dequant_is_exact():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from moe_infinity._v4_fp4 import mxfp4_dequant
    from moe_infinity.kernel.mxfp4_gemm import mxfp4_dequantize

    torch.manual_seed(137)
    blocks = torch.randint(
        0, 256, (5760, 1440), dtype=torch.uint8, device="cuda"
    )
    scales = torch.randint(
        120, 135, (5760, 90), dtype=torch.uint8, device="cuda"
    )
    expected = mxfp4_dequantize(
        blocks, scales, dtype=torch.bfloat16, block_size=32
    )
    actual = mxfp4_dequant(blocks, scales)

    relative_error = (
        (actual.float() - expected.float()).abs()
        / expected.float().abs().clamp_min(1e-12)
    ).max()
    assert actual.shape == (5760, 2880)
    assert actual.dtype == torch.bfloat16
    assert relative_error.item() == 0.0


@pytest.mark.gpu
def test_dequantized_option_a_matches_resident_expert_forward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        from moe_infinity._v4_fp4 import mxfp4_dequant
    except Exception:
        pytest.skip("native MXFP4 dequant extension not built")

    from moe_infinity.kernel.mxfp4_gemm import fused_mxfp4_gemm

    torch.manual_seed(137)
    tokens, hidden, intermediate = 3, 64, 32
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    gate_blocks = torch.randint(
        0,
        256,
        (2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    gate_scales = torch.randint(
        120,
        135,
        (2 * intermediate, hidden // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    down_blocks = torch.randint(
        0,
        256,
        (hidden, intermediate // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    down_scales = torch.randint(
        120,
        135,
        (hidden, intermediate // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    gate_bias = torch.randn(
        2 * intermediate, dtype=torch.bfloat16, device="cuda"
    )
    down_bias = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")

    resident_gate_up = fused_mxfp4_gemm(x, gate_blocks, gate_scales, gate_bias)
    resident_gate, resident_up = (
        resident_gate_up[:, ::2],
        resident_gate_up[:, 1::2],
    )
    resident_activated = (resident_up.clamp(-7, 7) + 1) * (
        resident_gate.clamp(max=7)
        * torch.sigmoid(resident_gate.clamp(max=7) * 1.702)
    )
    resident = fused_mxfp4_gemm(
        resident_activated.to(torch.bfloat16),
        down_blocks,
        down_scales,
        down_bias,
    )

    gate_weight = mxfp4_dequant(gate_blocks, gate_scales)
    down_weight = mxfp4_dequant(down_blocks, down_scales)
    option_a_gate_up = x @ gate_weight.t() + gate_bias
    option_a_gate, option_a_up = (
        option_a_gate_up[:, ::2],
        option_a_gate_up[:, 1::2],
    )
    option_a_activated = (option_a_up.clamp(-7, 7) + 1) * (
        option_a_gate.clamp(max=7)
        * torch.sigmoid(option_a_gate.clamp(max=7) * 1.702)
    )
    option_a = option_a_activated @ down_weight.t() + down_bias

    gate_weight_f = gate_weight.float()
    down_weight_f = down_weight.float()
    golden_gate_up = x.float() @ gate_weight_f.t() + gate_bias.float()
    golden_gate, golden_up = golden_gate_up[:, ::2], golden_gate_up[:, 1::2]
    golden_activated = (golden_up.clamp(-7, 7) + 1) * (
        golden_gate.clamp(max=7)
        * torch.sigmoid(golden_gate.clamp(max=7) * 1.702)
    )
    golden = golden_activated @ down_weight_f.t() + down_bias.float()

    # Bound bf16 rounding by the down-GEMM magnitude instead of comparing two
    # cancellation-sensitive bf16 paths directly.
    envelope = (
        8
        * (2**-8)
        * (
            golden_activated.abs() @ down_weight_f.abs().t()
            + down_bias.float().abs()
        )
        + 1e-2
    )
    assert ((option_a.float() - golden).abs() <= envelope).all()
    assert ((resident.float() - golden).abs() <= envelope).all()


def _native_dispatch(
    tmp_path,
    hidden_dtype,
    *,
    active_experts=(0,),
    gpu_routing=False,
    dispatch_fault=None,
    wait_for_result=True,
    capture_wait_error=False,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        from moe_infinity.runtime.model_offload import _load_prefetch_lib

        prefetch_lib = _load_prefetch_lib()
    except Exception as exc:
        pytest.skip(f"native Archer extension unavailable: {exc}")

    from moe_infinity.models.gpt_oss import SyncGptOssMLP

    class Config:
        hidden_size = 64
        intermediate_size = 32
        num_local_experts = 2
        num_experts_per_tok = 1

    torch.manual_seed(137)
    hidden = Config.hidden_size
    intermediate = Config.intermediate_size
    mlp = SyncGptOssMLP(Config())
    tensors = [
        torch.randint(
            0, 256, (2 * intermediate, hidden // 2), dtype=torch.uint8
        ),
        torch.randint(
            120, 135, (2 * intermediate, hidden // 32), dtype=torch.uint8
        ),
        torch.randn(2 * intermediate, dtype=torch.bfloat16),
        torch.randint(0, 256, (hidden, intermediate // 2), dtype=torch.uint8),
        torch.randint(
            120, 135, (hidden, intermediate // 32), dtype=torch.uint8
        ),
        torch.randn(hidden, dtype=torch.bfloat16),
    ]
    mlp.experts.gate_up_proj.requires_grad_(False)
    mlp.experts.down_proj.requires_grad_(False)
    mlp.experts.gate_up_proj.data = tensors[0].unsqueeze(0).repeat(2, 1, 1)
    mlp.experts.gate_up_proj_scales.data = (
        tensors[1].unsqueeze(0).repeat(2, 1, 1)
    )
    mlp.experts.gate_up_proj_bias.data = tensors[2].unsqueeze(0).repeat(2, 1)
    mlp.experts.down_proj.data = tensors[3].unsqueeze(0).repeat(2, 1, 1)
    mlp.experts.down_proj_scales.data = tensors[4].unsqueeze(0).repeat(2, 1, 1)
    mlp.experts.down_proj_bias.data = tensors[5].unsqueeze(0).repeat(2, 1)

    hidden_states = torch.randn(3, hidden, dtype=hidden_dtype, device="cuda:0")
    expected = mlp._expert_forward_mxfp4(hidden_states, 0)

    engine = prefetch_lib.prefetch_handle(f"{tmp_path}/", 0.5)
    expert_tensor_ids = [list(range(6)), list(range(6, 12))]
    for tensor_id, tensor in zip(expert_tensor_ids[0], tensors):
        engine.offload(tensor, tensor_id)
    for tensor_id, tensor in zip(expert_tensor_ids[1], tensors):
        engine.offload(tensor, tensor_id)
    dense_tensor_ids = list(range(12, 12 + 2 * torch.cuda.device_count()))
    for tensor_id in dense_tensor_ids:
        engine.offload(torch.zeros(1, dtype=torch.bfloat16), tensor_id)
    split = len(dense_tensor_ids) // 2
    dense_before = [
        (f"model.dense.{index}", [[tensor_id]])
        for index, tensor_id in enumerate(dense_tensor_ids[:split])
    ]
    dense_after = [
        (f"model.dense.{index + split}", [[tensor_id]])
        for index, tensor_id in enumerate(dense_tensor_ids[split:])
    ]
    engine.set_topology(
        dense_before
        + [("model.layers.0.mlp.experts", expert_tensor_ids)]
        + dense_after
    )

    dispatcher = prefetch_lib.expert_dispatcher(2, 1, 0, 6, 1)
    dispatcher.register_expert(0, 0, expert_tensor_ids[0], "")
    dispatcher.register_expert(0, 1, expert_tensor_ids[1], "")
    torch.cuda.set_device(hidden_states.device)
    router_mask = torch.zeros((3, 2), dtype=torch.bool, device="cuda:0")
    router_weights = torch.zeros((3, 2), dtype=torch.bfloat16, device="cuda:0")
    for expert_id in active_experts:
        router_mask[:, expert_id] = True
        router_weights[:, expert_id] = 1.0 / max(len(active_experts), 1)
    dispatcher.set_inputs(hidden_states, router_mask, router_weights)
    if gpu_routing:
        if dispatch_fault is not None:
            dispatcher._set_dispatch_fault_for_test(dispatch_fault)
        dispatcher.dispatch_experts(0)
    else:
        dispatcher.set_expected_queue(len(active_experts))
        for expert_id in sorted(active_experts):
            dispatcher.enqueue_expert(
                0, expert_id, expert_id % torch.cuda.device_count(), False
            )
        dispatcher.notify_fetch_start()
    actual = None
    if wait_for_result:
        try:
            actual = dispatcher.wait_expert()
        except RuntimeError as error:
            if not capture_wait_error:
                raise
            actual = error
    return actual, expected, hidden_states, tensors, dispatcher


@pytest.mark.gpu
def test_native_dispatcher_matches_resident_expert_forward(tmp_path):
    actual, resident, hidden_states, tensors, _ = _native_dispatch(
        tmp_path, torch.bfloat16
    )
    from moe_infinity._v4_fp4 import mxfp4_dequant

    gate_weight = mxfp4_dequant(tensors[0].cuda(), tensors[1].cuda()).float()
    down_weight = mxfp4_dequant(tensors[3].cuda(), tensors[4].cuda()).float()
    golden_gate_up = (
        hidden_states.float() @ gate_weight.t() + tensors[2].cuda().float()
    )
    golden_gate, golden_up = golden_gate_up[:, ::2], golden_gate_up[:, 1::2]
    golden_activated = (golden_up.clamp(-7, 7) + 1) * (
        golden_gate.clamp(max=7)
        * torch.sigmoid(golden_gate.clamp(max=7) * 1.702)
    )
    golden = golden_activated @ down_weight.t() + tensors[5].cuda().float()
    envelope = (
        8
        * (2**-8)
        * (
            golden_activated.abs() @ down_weight.abs().t()
            + tensors[5].cuda().float().abs()
        )
        + 1e-2
    )

    assert ((actual.to(golden.device) - golden).abs() <= envelope).all()
    assert ((resident.float() - golden).abs() <= envelope).all()


@pytest.mark.gpu
def test_native_dispatcher_rejects_mismatched_hidden_dtype(tmp_path):
    error, _, _, _, _ = _native_dispatch(
        tmp_path, torch.float32, capture_wait_error=True
    )

    assert isinstance(error, RuntimeError)
    assert "hidden_states dtype must match expert input dtype" in str(error)


@pytest.mark.gpu
def test_native_gpu_routing_reports_sorted_active_experts(tmp_path):
    _, _, _, _, dispatcher = _native_dispatch(
        tmp_path,
        torch.bfloat16,
        active_experts=(1, 0),
        gpu_routing=True,
    )
    assert dispatcher.take_last_active_experts() == [0, 1]
    stats = dispatcher.get_routing_stats()
    assert stats["route_batches"] == 1
    assert stats["route_failures"] == 0
    assert stats["last_active_experts"] == 2


@pytest.mark.parametrize("fault", ["callback", "worker"])
@pytest.mark.gpu
def test_native_routing_failure_rethrows_without_deadlock(tmp_path, fault):
    with pytest.raises(RuntimeError, match=f"injected {fault} routing failure"):
        _native_dispatch(
            tmp_path / fault,
            torch.bfloat16,
            active_experts=(0,),
            gpu_routing=True,
            dispatch_fault=fault,
        )


@pytest.mark.gpu
def test_synchronous_submission_failure_closes_route_state(tmp_path):
    with pytest.raises(RuntimeError, match="injected submission failure"):
        _native_dispatch(
            tmp_path,
            torch.bfloat16,
            active_experts=(0,),
            gpu_routing=True,
            dispatch_fault="submission",
        )


@pytest.mark.gpu
def test_dispatcher_destruction_drains_inflight_route_callback(tmp_path):
    import gc

    _, _, _, _, dispatcher = _native_dispatch(
        tmp_path,
        torch.bfloat16,
        active_experts=(0,),
        gpu_routing=True,
        wait_for_result=False,
    )
    del dispatcher
    gc.collect()


@pytest.mark.gpu
def test_native_wait_returns_stream_ordered_exact_output(tmp_path):
    actual, expected, _, _, _ = _native_dispatch(
        tmp_path,
        torch.bfloat16,
        active_experts=(0, 1),
        gpu_routing=True,
    )
    consumer = actual.float().square().sum()

    torch.testing.assert_close(
        consumer, expected.float().square().sum(), rtol=1e-2, atol=1e-2
    )


@pytest.mark.gpu
def test_native_gpu_routing_empty_mask_returns_zero_without_deadlock(tmp_path):
    actual, _, _, _, dispatcher = _native_dispatch(
        tmp_path,
        torch.bfloat16,
        active_experts=(),
        gpu_routing=True,
    )
    assert dispatcher.take_last_active_experts() == []
    assert torch.count_nonzero(actual).item() == 0


@pytest.mark.gpu
def test_native_wait_orders_output_on_non_default_caller_stream(tmp_path):
    caller_stream = torch.cuda.Stream()
    with torch.cuda.stream(caller_stream):
        actual, expected, _, _, _ = _native_dispatch(
            tmp_path,
            torch.bfloat16,
            active_experts=(0, 1),
            gpu_routing=True,
        )
        dependent = actual.float().square().sum()
        done = torch.cuda.Event()
        done.record(caller_stream)

    done.synchronize()
    torch.testing.assert_close(
        dependent, expected.float().square().sum(), rtol=1e-2, atol=1e-2
    )


@pytest.mark.parametrize(
    "fault",
    [
        "fetch",
        "exec",
        "output",
        "completion_event",
        "retirement_launch",
    ],
)
@pytest.mark.gpu
def test_worker_and_handoff_failures_rethrow_from_wait(tmp_path, fault):
    with pytest.raises(RuntimeError, match=f"injected {fault} failure"):
        _native_dispatch(
            tmp_path / fault,
            torch.bfloat16,
            active_experts=(0,),
            gpu_routing=True,
            dispatch_fault=fault,
        )


@pytest.mark.gpu
def test_failure_cleanup_allows_dispatcher_destruction(tmp_path):
    import gc

    with pytest.raises(RuntimeError, match="injected completion_event failure"):
        _native_dispatch(
            tmp_path,
            torch.bfloat16,
            active_experts=(0,),
            gpu_routing=True,
            dispatch_fault="completion_event",
        )
    gc.collect()


@pytest.mark.gpu
def test_failed_generation_closes_state_and_next_generation_runs(tmp_path):
    error, _, hidden, _, dispatcher = _native_dispatch(
        tmp_path,
        torch.bfloat16,
        active_experts=(0,),
        gpu_routing=True,
        dispatch_fault="exec",
        capture_wait_error=True,
    )
    assert isinstance(error, RuntimeError)
    stats = dispatcher.get_routing_stats()
    assert stats["pending"] == 0
    assert stats["route_pending"] == 0
    failed_generation = stats["current_generation"]

    mask = torch.tensor(
        [[True, False], [True, False], [True, False]],
        dtype=torch.bool,
        device=hidden.device,
    )
    dispatcher.set_inputs(hidden, mask, mask.to(torch.bfloat16))
    dispatcher.dispatch_experts(0)
    dispatcher._fail_dispatch_for_test(
        failed_generation, "injected stale generation failure"
    )
    recovered = dispatcher.wait_expert()
    assert recovered.shape == hidden.shape
    recovered_stats = dispatcher.get_routing_stats()
    assert recovered_stats["stale_failures_quarantined"] == 1
    assert recovered_stats["pending"] == 0
    assert recovered_stats["route_pending"] == 0


@pytest.mark.gpu
def test_retirement_launch_failure_retires_recorded_event_without_next_dispatch(
    tmp_path,
):
    import time

    error, _, _, _, dispatcher = _native_dispatch(
        tmp_path,
        torch.bfloat16,
        active_experts=(0,),
        gpu_routing=True,
        dispatch_fault="retirement_launch",
        capture_wait_error=True,
    )
    assert isinstance(error, RuntimeError)
    assert "injected retirement_launch failure" in str(error)

    # This completes the caller-stream waits and their host handoff. Do not
    # start another dispatch: retirement must be autonomous.
    torch.cuda.synchronize()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        stats = dispatcher.get_routing_stats()
        if stats["completion_events_outstanding"] == 0:
            break
        time.sleep(0.01)
    assert stats["completion_events_retired"] >= 1
    assert stats["completion_events_outstanding"] == 0


@pytest.mark.parametrize(
    "active_experts",
    [(0,), (1,), (0, 1), ()],
)
@pytest.mark.gpu
def test_native_and_eager_dispatch_are_output_identical(
    tmp_path, active_experts
):
    eager_output, _, _, _, _ = _native_dispatch(
        tmp_path / "eager",
        torch.bfloat16,
        active_experts=active_experts,
        gpu_routing=False,
    )
    native_output, _, _, _, native = _native_dispatch(
        tmp_path / "native",
        torch.bfloat16,
        active_experts=active_experts,
        gpu_routing=True,
    )
    assert native.take_last_active_experts() == sorted(active_experts)
    torch.testing.assert_close(
        native_output, eager_output, rtol=1e-2, atol=1e-2
    )


@pytest.mark.multi_gpu
def test_native_routing_preserves_round_robin_gpu_ownership(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    actual, expected, _, _, dispatcher = _native_dispatch(
        tmp_path,
        torch.bfloat16,
        active_experts=(0, 1),
        gpu_routing=True,
    )
    assert dispatcher.take_last_active_experts() == [0, 1]
    # Real MXFP4 expert outputs span a large dynamic range, so bound the
    # cross-device float32 accumulation by BF16 magnitude rather than a flat
    # absolute tolerance.
    actual_f = actual.float()
    expected_f = expected.float()
    envelope = 8 * (2**-8) * expected_f.abs() + 1e-2
    assert ((actual_f - expected_f).abs() <= envelope).all()
