from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _function_body(path, start, end):
    source = (ROOT / path).read_text(encoding="utf-8")
    return source[source.index(start) : source.index(end, source.index(start))]


def test_expert_forward_has_no_stream_synchronize():
    body = _function_body(
        "core/parallel/expert_module.cpp",
        "torch::Tensor MoEMLP::forward",
        "void MoEMLP::ForwardHelper",
    )
    assert "cudaStreamSynchronize" not in body


def test_exec_stream_guard_precedes_input_tensor_work():
    body = _function_body(
        "core/parallel/expert_dispatcher.cpp",
        "void ExpertDispatcher::GPUExecFunc",
        "bool ExpertDispatcher::OutputFunc",
    )
    guard_at = body.index("c10::cuda::CUDAStreamGuard guard(torch_stream)")
    token_mask_at = body.index("router_mask_.index")
    input_copy_at = body.index("hidden_states_.index")
    assert guard_at < token_mask_at < input_copy_at


def test_blocking_cuda_helpers_are_explicit_and_not_used_by_dispatch():
    utilities = (ROOT / "core/utils/cuda_utils.h").read_text(encoding="utf-8")
    dispatcher = (ROOT / "core/parallel/expert_dispatcher.cpp").read_text(
        encoding="utf-8"
    )
    module = (ROOT / "core/parallel/expert_module.cpp").read_text(
        encoding="utf-8"
    )
    assert "elapsed_millis_blocking" in utilities
    assert "BlockingCudaCopy" not in dispatcher + module
    assert "GpuTimer" not in dispatcher + module


def test_all_dispatch_workers_use_common_failure_closure():
    path = "core/parallel/expert_dispatcher.cpp"
    fetch = _function_body(
        path,
        "void ExpertDispatcher::GPUFetchFunc",
        "void ExpertDispatcher::GPUExecFunc",
    )
    execute = _function_body(
        path,
        "void ExpertDispatcher::GPUExecFunc",
        "bool ExpertDispatcher::OutputFunc",
    )
    output = _function_body(
        path,
        "bool ExpertDispatcher::OutputFunc",
        "std::vector<ExpertDispatcher::CallResult>",
    )
    for body in (fetch, execute, output):
        assert "catch (...)" in body
        assert "FailDispatch(args.generation, std::current_exception()" in body
        assert "pending_.fetch_sub" not in body
    assert 'DLOG_WARN("GPUExecFunc: expert forward failed' not in execute


def test_route_error_is_exception_ptr_only():
    header = (ROOT / "core/parallel/expert_dispatcher.h").read_text(
        encoding="utf-8"
    )
    source = (ROOT / "core/parallel/expert_dispatcher.cpp").read_text(
        encoding="utf-8"
    )
    assert "std::exception_ptr route_error_;" in header
    assert "route_error_ = std::string" not in source
    assert "std::rethrow_exception(route_error)" in source


def test_route_pending_submission_is_raii_guarded():
    body = _function_body(
        "core/parallel/expert_dispatcher.cpp",
        "void ExpertDispatcher::DispatchExperts",
        "void ExpertDispatcher::FailDispatch",
    )
    pending_at = body.index("route_pending_.compare_exchange_strong")
    guard_at = body.index(
        "DispatchSubmissionGuard submission(this, generation)"
    )
    first_throwing_work = body.index("TORCH_CHECK(completion_events_.empty()")
    assert pending_at < guard_at < first_throwing_work


def test_fail_dispatch_is_generation_explicit_and_wait_never_recurses_lock():
    header = (ROOT / "core/parallel/expert_dispatcher.h").read_text(
        encoding="utf-8"
    )
    wait = _function_body(
        "core/parallel/expert_dispatcher.cpp",
        "torch::Tensor ExpertDispatcher::WaitHiddenStates",
        "void ExpertDispatcher::DispatchExperts",
    )
    assert "FailDispatch(const std::uint64_t failing_generation" in header
    assert "FailDispatch(generation, error)" in wait
    detach_end = wait.index("  }\n\n  const int device")
    fail_at = wait.index("FailDispatch(generation, error)")
    assert detach_end < fail_at
    assert "completion_events_.clear()" not in wait


def test_completion_retirement_is_autonomous():
    source = (ROOT / "core/parallel/expert_dispatcher.cpp").read_text(
        encoding="utf-8"
    )
    dispatch = _function_body(
        "core/parallel/expert_dispatcher.cpp",
        "void ExpertDispatcher::DispatchExperts",
        "void ExpertDispatcher::FailDispatch",
    )
    assert "CompletionWaitsConsumedCallback" in source
    assert "CompletionRetirementFunc" in source
    assert "QueueUnwaitedEventsForQuery" in source
    assert "completion_events_.clear()" not in dispatch
