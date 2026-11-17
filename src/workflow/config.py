"""Pipeline configuration and result dataclasses."""

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # src/workflow/ -> SALT/


@dataclass
class Config:
    # Translation
    source: str = "Neon"           # source language (maps to prompts/<source>/ and kernels/source/)
    target: str = "RVV"            # target language (maps to kernels/target/)
    model: str = "gemini-3.8-flash"   # translation + repair (the paper used gemini-3.1-pro-preview-customtools)
    temperature: float = 0.1
    repair_temperature: float = 0.3
    thinking: Optional[str] = None  # Gemini thinking level (low/medium/high) or None

    # Verification
    verification_model: str = "gemini-3.8-flash"   # LLM for harness inference (buffer sizes, params); the paper used gemini-3.1-pro-preview. Empty = use 'model'

    # Target hardware: VLEN drives BOTH the translator's LMUL choice (prompt) and
    # the verifier's vsetvl semantics. 256 = legacy default (bare target name);
    # any other value suffixes the target as kernels/target/<name>-<vlen>.c.
    vlen: int = 256

    # Zephyr / Spike: zephyr-chipyard-sw in third_party/zephyr; Chipyard (for Spike) from $CHIPYARD_PATH,
    # with Spike skipped when it is unset.
    zephyr_base: str = field(default_factory=lambda: str(PROJECT_ROOT / "third_party" / "zephyr"))
    chipyard_path: str = field(default_factory=lambda: os.environ.get("CHIPYARD_PATH", ""))


    # Pipeline
    max_compile_retries: int = 5
    max_verification_retries: int = 5
    # Cap on all LLM repairs of one kernel together (compile, execution and verification), so a kernel
    # makes at most 1 + max_repairs LLM calls. 0 = no total cap: the compile/execution count restarts
    # after each verification repair, so up to 1 + C + V*(1 + C) calls (36 at the defaults).
    max_repairs: int = 0
    # Single-kernel mode: run the Zephyr build + Spike in build/slots/slot_<n> and the verification build
    # in build/slots/slot_<n>/verification, so concurrent single-kernel runs do not share build trees.
    # -1 = slot 0 and the shared verification build directory (one run at a time).
    build_slot: int = -1
    verification_backend: str = "auto"       # portfolio: cvc5 harness, route per-query (BV→bitwuzla, FP→cvc5); falls back to cvc5 if bitwuzla absent
    verification_timeout: int = 600          # TOTAL wall-clock budget (s) for the batch sweep; default 10 min, 0 = no budget
    verification_batch: int = 0              # if >0, run ONLY this batch size; 0 = sweep
    verification_input_range: tuple = None   # (lo, hi) finite F32 band for FP-multiply families; None = full domain
    verification_symbolic_params: bool = False  # prove for ALL valid params (default: concrete edge configs)
    # NaN rule of the proof: "strict" (NaN must match NaN; the paper's pipeline) or "sweep" (the rule of the
    # fully symbolic sweep: XNNPACK's, any output where the reference is NaN, for the multidimensional kernels
    # and f32-vmax/vmin, strict for the rest), so repairs are not spent on a difference the verdict accepts.
    verification_nan_rule: str = "strict"
    kernels_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "kernels")
    dry_run: bool = False
    skip_existing: bool = False
    jobs: int = 1                            # --batch: kernels to translate+verify concurrently (1 = sequential)
    rules_only: bool = False
    skip_translation: bool = False
    skip_spike: bool = False

    # Autocomp optimization (post-verify)
    optimize: bool = False                          # gate: run autocomp after a verified translation
    optimize_prob_id: int = -1                      # autocomp prob_id (single-kernel mode)
    optimize_map_file: Path = field(               # batch-mode kernel→prob_id map
        default_factory=lambda: PROJECT_ROOT / "kernels" / "optimize_map.json"
    )
    autocomp_models: list[str] = field(default_factory=lambda: [
        "anthropic::claude-opus-4-5-20251101",
    ])
    autocomp_iterations: int = 8
    autocomp_beam_size: int = 4
    autocomp_num_plan_candidates: int = 4
    autocomp_num_code_candidates: int = 2
    autocomp_vlen: int = 512
    autocomp_dlen: int = 256

    @property
    def source_dir(self) -> Path:
        return self.kernels_dir / "source"

    @property
    def target_dir(self) -> Path:
        return self.kernels_dir / "target"

    @property
    def status_file(self) -> Path:
        return self.kernels_dir / "status.json"

    @classmethod
    def from_env_and_args(cls, args: argparse.Namespace) -> "Config":
        cfg = cls()
        # Translation
        cfg.source = args.source
        cfg.target = args.target
        cfg.model = args.model
        cfg.temperature = args.temperature
        cfg.repair_temperature = args.repair_temperature
        cfg.thinking = args.thinking
        # Build paths
        cfg.zephyr_base = args.zephyr_base or os.environ.get("ZEPHYR_BASE", cfg.zephyr_base)
        cfg.chipyard_path = args.chipyard_path or os.environ.get("CHIPYARD_PATH", cfg.chipyard_path)
        # Pipeline
        cfg.max_compile_retries = args.max_compile_retries
        cfg.max_verification_retries = args.max_verification_retries
        cfg.max_repairs = max(0, getattr(args, "max_repairs", 0))
        cfg.build_slot = getattr(args, "build_slot", -1)
        cfg.verification_backend = getattr(args, "backend", "auto")  # portfolio by default
        cfg.verification_timeout = args.verification_timeout
        cfg.vlen = getattr(args, "vlen", 256)
        cfg.verification_batch = getattr(args, "verify_batch", 0)
        _ir = getattr(args, "input_range", "")
        if _ir:
            lo, hi = _ir.split(",")
            cfg.verification_input_range = (float(lo), float(hi))
        cfg.dry_run = args.dry_run
        cfg.skip_existing = args.skip_existing
        cfg.jobs = max(1, getattr(args, "jobs", 1))
        cfg.verification_symbolic_params = getattr(args, "symbolic_params", False)
        cfg.verification_nan_rule = getattr(args, "nan_rule", "strict")
        cfg.rules_only = args.rules_only
        cfg.skip_translation = args.skip_translation
        cfg.skip_spike = args.skip_spike
        # Autocomp opt-in
        cfg.autocomp_vlen = getattr(args, "autocomp_vlen", 512)
        cfg.autocomp_dlen = getattr(args, "autocomp_dlen", 256)
        cfg.optimize = getattr(args, "optimize", False)
        cfg.optimize_prob_id = getattr(args, "prob_id", -1)
        if getattr(args, "optimize_map", None):
            cfg.optimize_map_file = Path(args.optimize_map)
        if getattr(args, "autocomp_iterations", None) is not None:
            cfg.autocomp_iterations = args.autocomp_iterations
        if getattr(args, "autocomp_beam_size", None) is not None:
            cfg.autocomp_beam_size = args.autocomp_beam_size
        if getattr(args, "autocomp_model", None):
            cfg.autocomp_models = list(args.autocomp_model)
        if args.kernels_dir:
            cfg.kernels_dir = Path(args.kernels_dir)
        return cfg

    def create_llm_client(self):
        """Create an LLMClient for translation from this config."""
        from .llm import LLMClient
        return LLMClient.from_model_string(self.model)

    def create_verification_llm_client(self):
        """Create an LLMClient for verification tasks (buffer inference, invariants).
        Falls back to the translation model if verification_model is not set."""
        from .llm import LLMClient
        model = self.verification_model or self.model
        return LLMClient.from_model_string(model)


@dataclass
class CompileResult:
    """Result of a Zephyr/Spike compile-and-run attempt."""
    success: bool
    error_message: str = ""
    output: str = ""


@dataclass
class TranslationResult:
    """Result of translating a single kernel."""
    kernel_name: str
    success: bool
    attempts: int = 0
    output_file: str = ""
    error: str = ""
