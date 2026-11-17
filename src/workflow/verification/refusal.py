"""What a repair prompt says when the verifier refused a candidate instead of finding a difference.

A refusal names what the verifier could not follow: a native use of an input value (a C test, conversion or library
call, found statically), a native access at run time (C arithmetic on an input value, a C store, a C read through a
cast pointer), or an intrinsic it has no model for. Each kind gets the advice that fits it, and the candidate's line
when it can be found (the static check reports the offending expression, an unmodeled intrinsic its name)."""
import re

_ADVICE = (
    (r"has no symbolic model|no model for",
     "The verifier has no model for that intrinsic: express that step with other RVV intrinsics."),
    (r"stored natively|C store",
     "Write outputs with vector stores (__riscv_vse*), not C assignments to output elements."),
    (r"native C access to a symbolic buffer",
     "Read inputs with vector loads (__riscv_vle*), not C reads through a pointer (a cast such as "
     "`*(const uint64_t*) p` included), and do not read outputs back in C."),
    (r"tail element",
     "Do not read vector elements past the vl of the operation that produced them (RVV leaves tail elements "
     "unspecified): use the same vl for the later operation, or the tail-undisturbed (_tu) variant to keep them."),
    (r"native float arithmetic",
     "Do the arithmetic on input values with RVV intrinsics: a scalar read from an input may only be passed to an "
     "intrinsic, not used in C arithmetic."),
    (r"extracted (?:integer|sum)",
     "Use the reduction's extracted sum only as `*output += sum` (once), not in other C code or another intrinsic."),
    (r"converted to an integer",
     "Convert with RVV conversion intrinsics (__riscv_vfcvt*), not C casts of input values."),
    (r"passed to \w+",
     "Use the RVV intrinsic for that operation (e.g. __riscv_vfabs, __riscv_vfmin) instead of a C library call on "
     "an input value."),
    (r"unrewritten param read",
     "Read parameters as fields (`params->scalar.min`), not through a cast pointer to the params struct."),
    (r"native read of a param field",
     "Read parameters as fields of the kernel's own params argument, in the kernel function itself "
     "(`params->scalar.slope`), not in a helper or through a copy of the params pointer."),
    (r"signature differs",
     "Keep the reference's exact signature: the same parameter types, including the `union`/`struct` of the params, "
     "and do not redefine XNNPACK's params types."),
    (r"test on a native value|C-level test|used as an index|address of a native value|struct member",
     "Do not branch on, index with or reinterpret input values in C: compute every case with vector intrinsics "
     "(compares into masks, __riscv_vmerge) instead."),
)
_GENERIC = ("Compute on input values with RVV intrinsics (a scalar read from an input may only be passed to an "
            "intrinsic) and write outputs with vector stores.")


def advice(reason: str) -> str:
    return next((a for pat, a in _ADVICE if re.search(pat, reason)), _GENERIC)


def source_line(src: str, reason: str):
    """The 1-based line of the candidate the refusal points at, or None: the static check's offending expression
    (tokens separated by spaces, matched with any spacing) or an unmodeled intrinsic's first call."""
    if re.search(r"no symbolic model|no model for", reason):
        for name in re.findall(r"__riscv_\w+|\bv[a-z0-9]+_[a-z0-9_]+", reason):
            if hit := re.search(rf"\b{re.escape(name)}\s*\(", src):
                return src.count("\n", 0, hit.start()) + 1
        return None
    m = re.search(r":\s*([^:]*)$", reason)
    toks = m.group(1).split() if m else []
    if len(toks) < 2:
        return None
    hit = re.search(r"\s*".join(re.escape(t) for t in toks), src)
    return src.count("\n", 0, hit.start()) + 1 if hit else None


def describe(reason: str, src: str = "") -> str:
    """The repair prompt's text for a refusal: what the verifier could not follow, where, and what to change."""
    line = source_line(src, reason) if src else None
    where = f" (line {line}: `{src.splitlines()[line - 1].strip()}`)" if line else ""
    return (f"The verifier refused this kernel instead of finding a difference: {reason}{where}.\n"
            "It runs the kernel natively and models only the RVV intrinsics. " + advice(reason))


def missing_intrinsics(compile_error: str) -> list:
    """RVV intrinsics a harness compile error says the verifier has no model (or no such overload) for."""
    pats = (r"undeclared identifier '(__riscv_\w+)'", r"'(__riscv_\w+)' was not declared",
            r"no matching function for call to '(?:salt_cvc5::)?(__riscv_\w+)", r"use of undeclared identifier '(__riscv_\w+)'")
    return sorted({m for p in pats for m in re.findall(p, compile_error)})
