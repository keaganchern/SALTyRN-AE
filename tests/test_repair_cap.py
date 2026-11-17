"""--max-repairs caps all LLM repairs of one kernel together (compile, execution and verification), so a kernel
makes at most 1 + N LLM calls; without it, the compile count restarts after each verification repair."""
from types import SimpleNamespace

import pytest

from src.workflow import pipeline
from src.workflow.config import Config
from src.workflow.status import StatusTracker


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, prompt, system=None, temperature=0.1, thinking=None, **kw):
        self.calls += 1
        return "```c\nvoid test_rvv(void) {}\n```"


def translate(tmp_path, monkeypatch, builds, verdict, max_repairs):
    """builds(i) says whether the i-th build passes; Spike passes whenever the build does."""
    kernels = tmp_path / "kernels"
    (kernels / "source").mkdir(parents=True)
    src = kernels / "source" / "f32-vadd.c"
    src.write_text("void xnn_f32_vadd(void) {}\n")
    n = iter(range(10**6))

    def compile_and_run(name, cfg, kernel_file):
        ok = builds(next(n))
        return (SimpleNamespace(success=ok, error="error: boom", elapsed=0.0),
                SimpleNamespace(success=True, passed=True, output="PASS", error="", elapsed=0.0) if ok else None)

    monkeypatch.setattr(pipeline, "_compile_and_run", compile_and_run)
    monkeypatch.setattr(pipeline, "_run_formal_verification", lambda *a, **kw: {
        "verdict": verdict, "max_verified_batch": 0, "counterexample": {"batch": 3, "fail_index": 0}})
    llm = FakeLLM()
    cfg = Config(kernels_dir=kernels, vlen=512, max_repairs=max_repairs)
    tracker = StatusTracker(kernels / "status.json")
    result = pipeline.translate_kernel(src, cfg, llm, {}, tracker)
    return llm.calls, result, tracker.get("f32-vadd")


@pytest.mark.parametrize("cap", [1, 3, 10])
def test_cap_bounds_compile_repairs(tmp_path, monkeypatch, cap):
    calls, result, st = translate(tmp_path, monkeypatch, lambda i: False, "ALL_PASSED", cap)
    assert calls == 1 + min(cap, 5)            # the per-kind cap (5) still applies under a larger total cap
    assert not result.success and st["attempts"] == calls and st["error"].startswith("compile_error")


def test_cap_counts_every_kind_of_repair(tmp_path, monkeypatch):
    # Every other build fails and every proof finds a counterexample: compile and verification repairs alternate.
    alternate = lambda i: i % 2 == 1
    uncapped, _, _ = translate(tmp_path / "a", monkeypatch, alternate, "COUNTEREXAMPLE", 0)
    assert uncapped == 1 + 6 + 5               # 6 compile repairs: their count restarts after each verification repair
    capped, result, st = translate(tmp_path / "b", monkeypatch, alternate, "COUNTEREXAMPLE", 10)
    assert capped == 11 and st["attempts"] == 11 and not result.success


def test_cap_does_not_stop_a_kernel_that_verifies(tmp_path, monkeypatch):
    calls, result, st = translate(tmp_path, monkeypatch, lambda i: i >= 2, "PARTIAL", 2)
    # 2 compile repairs, then the build passes; the proof's PARTIAL with max_batch 0 is accepted on the Spike pass
    assert calls == 3 and result.success and st["compiled"]


class ScriptedLLM(FakeLLM):
    """Replies from a script; a non-kernel reply ("", a fragment) is what the AE run saw from thinking repairs."""
    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)

    def chat(self, prompt, system=None, temperature=0.1, thinking=None, **kw):
        self.calls += 1
        return self.replies.pop(0) if self.replies else "```c\nvoid test_rvv(void) {}\n```"


def test_non_kernel_reply_keeps_the_previous_code(tmp_path, monkeypatch):
    kernels = tmp_path / "kernels"
    (kernels / "source").mkdir(parents=True)
    src = kernels / "source" / "f32-vadd.c"
    src.write_text("void xnn_f32_vadd(void) {}\n")
    seen, builds = [], iter(range(10**6))

    def compile_and_run(name, cfg, kernel_file):
        seen.append(kernel_file.read_text())
        ok = next(builds) >= 2              # the first two builds fail
        return (SimpleNamespace(success=ok, error="error: boom", elapsed=0.0),
                SimpleNamespace(success=True, passed=True, output="PASS", error="", elapsed=0.0) if ok else None)
    monkeypatch.setattr(pipeline, "_compile_and_run", compile_and_run)
    monkeypatch.setattr(pipeline, "_run_formal_verification", lambda *a, **kw: {
        "verdict": "PARTIAL", "max_verified_batch": 3, "counterexample": {}})
    good = "```c\nvoid test_rvv(int x) { (void)x; }\n```"
    llm = ScriptedLLM([good, "", "```c\nvoid test_rvv(int y) { (void)y; }\n```"])   # translation, empty, repair
    cfg = Config(kernels_dir=kernels, vlen=512, max_repairs=10)
    result = pipeline.translate_kernel(src, cfg, llm, {}, StatusTracker(kernels / "status.json"))
    assert result.success and llm.calls == 3
    assert seen[1] == seen[0] and "test_rvv(int x)" in seen[1]     # the empty reply did not replace the kernel
    assert "test_rvv(int y)" in seen[2]


def test_unchanged_code_does_not_rerun_the_proof(tmp_path, monkeypatch):
    kernels = tmp_path / "kernels"
    (kernels / "source").mkdir(parents=True)
    src = kernels / "source" / "f32-vadd.c"
    src.write_text("void xnn_f32_vadd(void) {}\n")
    proofs = []
    monkeypatch.setattr(pipeline, "_compile_and_run", lambda name, cfg, f: (
        SimpleNamespace(success=True, error="", elapsed=0.0),
        SimpleNamespace(success=True, passed=True, output="PASS", error="", elapsed=0.0)))
    monkeypatch.setattr(pipeline, "_run_formal_verification", lambda *a, **kw: proofs.append(1) or {
        "verdict": "COUNTEREXAMPLE", "max_verified_batch": 0, "counterexample": {"batch": 1, "fail_index": 0}})
    llm = ScriptedLLM(["```c\nvoid test_rvv(int x) { (void)x; }\n```", "nothing useful"])
    cfg = Config(kernels_dir=kernels, vlen=512, max_repairs=2)
    pipeline.translate_kernel(src, cfg, llm, {}, StatusTracker(kernels / "status.json"))
    assert len(proofs) == 2 and llm.calls == 3   # the non-kernel repair reused the first proof
