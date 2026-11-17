"""Derive a multidimensional kernel's shape domain from its source.

XNNPACK kernels state their size contract in `assert`s: `assert(nc != 0)` marks
a size, `assert(kc % sizeof(float) == 0)` gives its unit (a byte count stepped
in floats), `assert(mr <= 4)` bounds it, `assert(nr == 2)` fixes a tile
parameter. This module reads those facts from the reference and the candidate,
then combines them with the fixture's ABI facts (derived strides, layouts) into
a growth plan: every free size grows from one unit upward, every other integer
argument is derived, fixed, or one of a few listed variants.

A plan covers every integer argument or raises, so no argument is silently left
at an arbitrary value.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from .frontend import Signature

# LP64 sizes of the types that appear in XNNPACK unit expressions.
_SIZEOF = {
    "float": 4, "double": 8, "void*": 8, "void *": 8, "size_t": 8,
    "int8_t": 1, "uint8_t": 1, "int16_t": 2, "uint16_t": 2,
    "int32_t": 4, "uint32_t": 4, "int64_t": 8, "uint64_t": 8,
    "xnn_float16": 2, "char": 1,
}
# Integer arguments that are pitches or pointer adjustments, never grown sizes.
_NOT_A_SIZE = re.compile(r"stride|increment|offset|extra|padding|_start$|_end$|^n[rk]$|^[ks]r$")
# Size vocabulary for sizes the source never asserts or compares (maxpool's
# kernel_elements is only used as `(int) kernel_elements - 9`).
_SIZE_WORDS = re.compile(r"height|width|channels|elements|pixels|rows|batch|block|^[mnk][cr]$|^g$|^ks$")


@dataclass
class Axis:
    name: str
    step: int            # argument value = index * step
    hi: int | None       # largest legal index (None: unbounded)


@dataclass
class ShapePlan:
    axes: list[Axis]
    variants: dict[str, list]            # finite sets enumerated at every shape
    derived: dict[str, str]              # C expressions over the axes (fixture strides)
    preconditions: list[str]             # reference entry asserts over integer arguments
    notes: list[str] = field(default_factory=list)
    candidate_checks: list[str] = field(default_factory=list)   # candidate-only entry asserts

    def frontier(self, level: int):
        """Shapes whose largest axis index is exactly `level`, each paired with
        every variant combination. Levels 1..N together cover the whole box
        [1..N]^k (clipped at each axis bound) with no gaps or repeats."""
        if level < 1:
            raise ValueError("level must be positive")
        caps = [min(level, a.hi) if a.hi else level for a in self.axes]
        if all(c < level for c in caps):
            return
        names, vals = list(self.variants), [self.variants[n] for n in self.variants]
        for idx in itertools.product(*(range(1, c + 1) for c in caps)):
            if max(idx) != level:
                continue
            shape = {a.name: i * a.step for a, i in zip(self.axes, idx)}
            for combo in itertools.product(*vals):
                yield shape | dict(zip(names, combo))

    def exhausted(self, level: int) -> bool:
        """True once every axis is bounded and below `level` (finite domain done)."""
        return all(a.hi is not None and a.hi < level for a in self.axes)

    def describe(self) -> dict:
        return {"axes": [{"name": a.name, "step": a.step, "max_index": a.hi} for a in self.axes],
                "variants": self.variants, "derived": self.derived,
                "preconditions": self.preconditions, "candidate_checks": self.candidate_checks,
                "notes": self.notes}


def _entry_asserts(src: str, ints: list[str]) -> list[str]:
    """Bodies of the `assert(...)` calls at function entry, before the first loop
    or assignment to an integer argument. Later asserts describe a modified
    argument (qu8-rdsum's remainder block asserts `channels <= 31` after
    `channels -= 32`), not the caller's contract."""
    barrier = re.compile(r"\b(for|while|do)\b|(?:\+\+|--)\s*\b(?:%s)\b|\b(?:%s)\b\s*(?:[-+*/%%&|^]?=(?!=)|\+\+|--)"
                         % ("|".join(ints) or "$^", "|".join(ints) or "$^"))
    start = src.find("{", re.search(r"\btest_\w+\s*\(", src).end()) if re.search(r"\btest_\w+\s*\(", src) else 0
    out = []
    for m in re.finditer(r"\bassert\s*\(", src[start:]):
        depth, i = 1, m.end()
        text = src[start:]
        if barrier.search(text, 0, m.start()):
            break
        while i < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        out.append(" ".join(text[m.end():i - 1].split()))
    return out


def _eval_unit(expr: str) -> int | None:
    """Evaluate a unit expression like `sizeof(float)` or `(1 * sizeof(void*))`."""
    def size(m):
        t = " ".join(m.group(1).split())
        return str(_SIZEOF[t]) if t in _SIZEOF else "None"
    e = re.sub(r"sizeof\s*\(\s*([\w\s\*]+?)\s*\)", size, expr)
    if not re.fullmatch(r"[\d\s\*\+\(\)]+", e):
        return None
    try:
        v = eval(e, {"__builtins__": {}})   # digits and * + ( ) only
    except Exception:
        return None
    return v if isinstance(v, int) and v > 0 else None


def _compared(name: str, body: str) -> bool:
    n = re.escape(name)
    return bool(re.search(rf"\b{n}\b\s*(-=|--|<=?|>=?|!=)|(--|<=?|>=?)\s*\b{n}\b", body))


def derive_plan(sig: Signature, sources: list[str], derived: dict[str, str],
                fixture_dims: dict[str, list]) -> ShapePlan:
    """Build the growth plan for one kernel pair.

    `sources` are the reference text, then the candidate's. The reference's entry
    asserts define the domain; candidate asserts it does not share are checked at
    every shape (a candidate that rejects a legal shape fails there). `derived` maps argument
    names to C expressions the fixture computes from the sizes (strides, tile
    constants, ranges). `fixture_dims` are the fixture's value lists: entries
    that are sizes become axes grown from one unit; the rest (pointer offsets,
    padding, harness-only modes such as a parameter profile) are variants
    enumerated at every shape.
    """
    derived = dict(derived)
    ints = [p.name for p in sig.params if p.is_int_scalar]
    variants = {d: list(v) for d, v in fixture_dims.items() if d not in ints}
    nonzero, unit, upper, fixed, pre = set(), {}, {}, {}, []
    body = "\n".join(sources)
    def over_ints(a):                                  # skip pointer and local-variable asserts
        idents = set(re.findall(r"\b[A-Za-z_]\w*\b", a)) - {"sizeof", "NULL", "void"} - set(_SIZEOF)
        return bool(idents) and idents <= set(ints)
    ref_asserts = [a for a in _entry_asserts(sources[0], ints) if over_ints(a)]
    candidate_checks = [a for s in sources[1:] for a in _entry_asserts(s, ints)
                        if over_ints(a) and a not in ref_asserts]
    for a in ref_asserts:
        pre.append(a)                                  # checked at every shape
        if m := re.fullmatch(r"(\w+) (!= 0|> 0|>= 1)", a):
            nonzero.add(m.group(1))
        elif m := re.fullmatch(r"(\w+) % (.+) == 0", a):
            if (u := _eval_unit(m.group(2))) is None:
                raise NotImplementedError(f"unit expression not understood: assert({a})")
            unit[m.group(1)] = max(unit.get(m.group(1), 1), u)
        elif m := re.fullmatch(r"(\w+) (<=|<) (\d+)", a):
            bound = int(m.group(3)) - (m.group(2) == "<")
            upper[m.group(1)] = min(upper.get(m.group(1), bound), bound)
        elif m := re.fullmatch(r"(\w+) == (\d+)", a):
            fixed[m.group(1)] = int(m.group(2))
    axes, notes = [], []
    for n in ints:
        if n in derived:
            continue
        if n in fixed:
            derived[n] = str(fixed[n])
            notes.append(f"{n} = {fixed[n]} (asserted)")
            continue
        is_size = (n in nonzero or (not _NOT_A_SIZE.search(n)
                                    and (_compared(n, body) or _SIZE_WORDS.search(n))))
        if not is_size:
            if n not in fixture_dims:
                raise NotImplementedError(f"integer argument {n!r} is not a size and the fixture "
                                          "neither derives it nor lists values for it")
            variants[n] = list(fixture_dims[n])
            notes.append(f"{n}: variant {variants[n]}")
            continue
        step = unit.get(n, 1)
        hi = upper[n] // step if n in upper else None
        axes.append(Axis(n, step, hi))
        why = ("asserted nonzero" if n in nonzero else
               "used as a loop bound" if _compared(n, body) else "size name")
        notes.append(f"{n}: size ({why}), step {step}" + (f", at most {upper[n]}" if n in upper else ""))
    if not axes:
        raise NotImplementedError("no size arguments found")
    for a in candidate_checks:
        notes.append(f"candidate-only precondition: assert({a})")
    return ShapePlan(axes, variants, derived, pre, notes, candidate_checks)


# ---- threshold probes ------------------------------------------------------
#
# A box [1..N]^k misses every path guarded by a threshold above N: the Phase-2
# gemm's `while (nc >= 16)` main loop never runs at nc <= 9. Both kernels state
# their thresholds; probing each at K, K+1, 2K and 2K+1 (the other axes at 1 and
# at their own first threshold, small bounded axes at every value) exercises the
# guarded path once, twice and with a remainder.

_UNIT = r"[\d\s\*]*(?:sizeof\s*\([\w\s\*]+\))?[\d\s\*]*"
_CMP = re.compile(rf"\b(\w+)\s*(>=|>|<=|<)\s*\(?\s*({_UNIT})\s*\)?(?![\w(])")                    # x op K
_CMP_R = re.compile(rf"(?<![\w)\]])\(?\s*({_UNIT})\s*\)?\s*(>=|>|<=|<)\s*(\w+)\b(?!\s*[\[(])")   # K op x
_BIT = re.compile(r"\b(\w+)\s*&\s*\(?\s*(\d+)\s*\)?")                                  # x & K
_MIRROR = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}


def _copies(name: str, body: str) -> dict:
    """Locals that track a size: x = name, x = name - C (maxpool's later passes),
    x = name / U (a byte count in elements). Maps x to (div, add): name = div * x + add."""
    out = {name: (1, 0)}
    decl = r"\b(?:size_t|int|unsigned|uint32_t|ptrdiff_t)\s+(\w+)\s*=\s*(?:\(\s*\w+\s*\)\s*)?"
    for m in re.finditer(decl + rf"\(?\s*{re.escape(name)}\s*(?:([-+])\s*(\d+)|/\s*({_UNIT}))?\s*\)?\s*;", body):
        if m.group(2):
            out[m.group(1)] = (1, int(m.group(3)) if m.group(2) == "-" else -int(m.group(3)))
        elif m.group(4):
            u = _eval_unit(m.group(4).strip())
            if u:
                out[m.group(1)] = (u, 0)
        else:
            out[m.group(1)] = (1, 0)
    return out


def _vlmax(src: str, var: str, vlen: int):
    """VLMAX of the vsetvl that sets `vl` from `var`, in elements, if any."""
    m = re.search(rf"\bvl\s*=\s*__riscv_vsetvl_e(\d+)m(f?)(\d+)\s*\(\s*{re.escape(var)}\s*\)", src)
    if not m:
        return None
    sew, lmul = int(m.group(1)), (1 / int(m.group(3)) if m.group(2) else int(m.group(3)))
    return int(vlen // sew * lmul)


def thresholds(plan: ShapePlan, sources: list[str], vlen: int = 512) -> dict:
    """Axis name -> sorted threshold indices (> 1) at which some branch changes:
    comparisons with a constant either way round (`c >= 4`, `4 < k`), bit tests
    (`nc & 8`), min/max with a constant (`min(r, 252)`), and comparisons with `vl`
    (one vector's worth, `w > vl`), on the size or a local that tracks it."""
    out = {}
    for a in plan.axes:
        ks = set()

        def add(div_add, op, v):
            div, off = div_add
            x = v if op in (">=", "<", "&") else v + 1      # the first x on the other side
            n = div * x + off                                # in the argument's units
            k = -(-n // a.step)
            if k > 1 and (a.hi is None or k <= a.hi):
                ks.add(k)

        for src in sources:
            names = _copies(a.name, src)
            for m in _CMP.finditer(src):
                if m.group(1) in names and m.group(3).strip():
                    v = _eval_unit(m.group(3).strip()) if m.group(3).strip() != "vl" else None
                    if v:
                        add(names[m.group(1)], m.group(2), v)
            for m in _CMP_R.finditer(src):
                if m.group(3) in names and m.group(1).strip():
                    v = _eval_unit(m.group(1).strip())
                    if v:
                        add(names[m.group(3)], _MIRROR[m.group(2)], v)
            for m in _BIT.finditer(src):
                k = int(m.group(2))
                if m.group(1) in names and k > 1 and k & (k - 1) == 0:
                    add(names[m.group(1)], "&", k)
            for m in re.finditer(r"\b(?:min|max)\s*\(\s*(\w+)\s*,\s*([\w\s\*\(\)]+?)\s*\)", src):
                v = _eval_unit(m.group(2)) if m.group(1) in names else None
                if v:
                    add(names[m.group(1)], ">=", v)
            for m in re.finditer(r"\b(\w+)\s*(>=|>)\s*vl\b|\bvl\s*(<=|<)\s*(\w+)\b", src):
                var = m.group(1) or m.group(4)
                if var in names:
                    vm = _vlmax(src, var, vlen)
                    if vm:
                        add(names[var], ">", vm)
        if ks:
            out[a.name] = sorted(ks)
    return out


def probe_shapes(plan: ShapePlan, sources: list[str], limit: int = 200, vlen: int = 512) -> list[dict]:
    th = thresholds(plan, sources, vlen)
    # Other axes at 1, their first threshold (a tail) and their last (a main loop):
    # a later pass with 5+ taps is exercised with full 4-channel vectors too.
    base = {a.name: sorted({1} | ({th[a.name][0], th[a.name][-1]} if a.name in th else set())) for a in plan.axes}
    for a in plan.axes:
        if a.hi is not None and a.hi <= 4:
            base[a.name] = list(range(1, a.hi + 1))
    variant = {n: v[0] for n, v in plan.variants.items()}
    tiered = []                      # (tier, size, shape): every threshold once (K, K+1) before twice (2K, 2K+1)
    for a in plan.axes:
        for k in th.get(a.name, []):
            for tier, v in ((0, k), (0, k + 1), (1, 2 * k), (1, 2 * k + 1)):
                if a.hi is not None and v > a.hi:
                    continue
                others = [b for b in plan.axes if b.name != a.name]
                for combo in itertools.product(*(base[b.name] for b in others)):
                    # argument values, like frontier()
                    s = {a.name: v * a.step, **{b.name: x * b.step for b, x in zip(others, combo)}, **variant}
                    tiered.append((tier, sum(s[b.name] // b.step for b in plan.axes), s))
    shapes = []
    for _, _, s in sorted(tiered, key=lambda t: (t[0], t[1])):
        if s not in shapes:
            shapes.append(s)
    return shapes[:limit]
