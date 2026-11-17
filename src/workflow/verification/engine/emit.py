"""Generate a C++ equivalence harness from a NEON+RVV kernel pair.

Two paths: generate_harness (elementwise — bound sweep + params) and
generate_multidim (FixtureSpec-driven multi-dim families, e.g. transpose).
"""
from __future__ import annotations

import os
import re

from .frontend import (parse_signature, resolve_roles, parse_struct_leaves,
                       classify_param_struct, Signature, Roles)
# Curated valid param values / ranges / constraints (the merged param data module).
from ..param_configs import (FIELD_EDGES, EDGE_VALUES, PARAM_CONSTRAINTS,
                            CONCRETE_PARAM_CONFIGS, INT_CLAMP_EDGES, CLAMP_FIELD_SIDE,
                            get_param_ranges)

_FP_CTYPES = ("float", "double", "_Float16", "xnn_float16")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_XNN_MIN_H = os.path.join(_ROOT, "src", "verification_bw", "xnn_minimal.h")

# element type → (ElementKind, byte width)
_KIND = {
    "float": "F32", "double": "F64",
    "int8_t": "SINT", "int16_t": "SINT", "int32_t": "SINT", "int64_t": "SINT",
    "signed char": "SINT", "short": "SINT", "int": "SINT", "long": "SINT",
    "uint8_t": "UINT", "uint16_t": "UINT", "uint32_t": "UINT", "uint64_t": "UINT",
    "unsigned char": "UINT", "unsigned short": "UINT", "unsigned int": "UINT",
    "unsigned long": "UINT",
    "_Float16": "F16", "xnn_float16": "F16",
}
_BYTES = {
    "float": 4, "double": 8,
    "int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8,
    "signed char": 1, "short": 2, "int": 4, "long": 8,
    "uint8_t": 1, "uint16_t": 2, "uint32_t": 4, "uint64_t": 8,
    "unsigned char": 1, "unsigned short": 2, "unsigned int": 4, "unsigned long": 8,
    "_Float16": 2, "xnn_float16": 2,
}


def _kind(pointee: str) -> str:
    if pointee not in _KIND:
        raise NotImplementedError(f"unhandled element type {pointee!r}")
    return _KIND[pointee]


# Per-kernel output ULP tolerance: 0 (default) = bit-exact; >0 relaxes FP outputs
# to "agree within N ULP". Keys are name substrings; SALT_ULP_TOL=<n> overrides all.
_ULP_TOL = {
    "raddstoreexpminusmax": 4,
}


def _ulp_tol_for(name: str) -> int:
    env = os.environ.get("SALT_ULP_TOL")
    if env not in (None, ""):
        return int(env)
    for key, tol in _ULP_TOL.items():
        if key in name:
            return tol
    return 0


def _detect_byte_unit(src: str, bound: str):
    """Detect the XNNPACK `batch % sizeof(T)` byte-count idiom. Returns
    (is_byte_unit, element_type). `batch` is then a byte count whose unit is
    sizeof(T); otherwise it's an element count."""
    m = re.search(rf"\b{re.escape(bound)}\b\s*[%/]\s*sizeof\s*\(\s*([\w ]+?)\s*\)", src)
    if not m:  # loop-decrement form: `batch -= k * sizeof(T)`
        m = re.search(rf"\b{re.escape(bound)}\b[^;\n]*sizeof\s*\(\s*([\w ]+?)\s*\)", src)
    return (True, m.group(1).strip()) if m else (False, None)


def _field_values(path: str, ctype: str):
    """Valid concrete edge values for a struct leaf — by field name first (the
    semantic table), then by type. Returns C literals."""
    fname = path.split(".")[-1]
    side = CLAMP_FIELD_SIDE.get(fname)
    if side and re.fullmatch(r"(u?int(8|16|32|64)_t|(unsigned |signed )?(char|short|int))", ctype):
        sign = "unsigned" if ctype.startswith("u") else "signed"
        return [str(v) for v in INT_CLAMP_EDGES[sign][side]]
    raw = FIELD_EDGES.get(fname)
    if raw is None:
        raw = EDGE_VALUES.get(ctype)
    if not raw:
        return []
    if ctype == "float" or ctype in ("xnn_float16", "_Float16"):
        return [f"{float(v)}f" for v in raw]
    if ctype == "double":
        return [f"{float(v)}" for v in raw]
    return [str(int(v)) for v in raw]          # integer fields


def _gen_configs(leaves, cap: int = 24, struct_name: str = ""):
    """One-at-a-time configs around a base (each field's first value), so every
    field's values are exercised without a combinatorial blow-up. Cross-field
    validity holds by construction (output_min/max are fixed, multiplier≥1, …)."""
    if struct_name in CONCRETE_PARAM_CONFIGS:
        paths, combos = CONCRETE_PARAM_CONFIGS[struct_name]
        return paths, combos[:cap]
    fields = [(p, _field_values(p, t)) for p, t, _ in leaves if p != "_"]
    fields = [(p, vs) for p, vs in fields if vs]
    if not fields:
        return [], []
    paths = [p for p, _ in fields]
    base = [vs[0] for _, vs in fields]
    combos = [tuple(base)]
    for i, (_, vs) in enumerate(fields):
        for v in vs[1:]:
            c = list(base)
            c[i] = v
            combos.append(tuple(c))
    return paths, combos[:cap]


def _emit_config_table(st: str, paths, combos) -> str:
    set_cases, name_cases = [], []
    for i, combo in enumerate(combos):
        assigns = " ".join(f"p.{path} = {val};" for path, val in zip(paths, combo))
        set_cases.append(f"    case {i}: {assigns} break;")
        label = ",".join(f"{path.split('.')[-1]}={val.rstrip('f')}"
                         for path, val in zip(paths, combo))
        name_cases.append(f'    case {i}: return "{label}";')
    return (f"static void _set_params({st}& p, size_t cfg) {{\n"
            f"  switch (cfg) {{\n" + "\n".join(set_cases) + "\n  }\n}\n"
            f"static const char* _cfg_name(size_t cfg) {{\n"
            f"  switch (cfg) {{\n" + "\n".join(name_cases) +
            '\n  default: return "?";\n  }\n}\n')


def _strip_simd_includes(src: str) -> str:
    """Remove the kernel's own <arm_neon.h>/<riscv_vector.h> includes — salt.hpp
    supplies the symbolic shims, and the real headers aren't on the include path — and its standard C
    headers (<math.h>, ...), which the harness includes at file scope (not inside the kernel's namespace)."""
    return re.sub(r'^[ \t]*#[ \t]*include[ \t]*<(arm_neon|riscv_vector|math|float|stdint|stddef|stdbool|string|'
                  r'limits|assert|stdlib|inttypes)\.h>[ \t]*$', '', src, flags=re.M)


def _strip_kernel_redeclarations(src: str) -> str:
    """Remove what the harness already provides to the RVV kernel: file-scope
    `struct xnn_*` declarations/definitions (inside the RVV namespace they declare
    a distinct type, so `&params` no longer converts) and `#include "xnnpack/…"`
    lines (xnn_minimal.h supplies the helpers; the real headers aren't on the path)."""
    # A trailing comment on the line (`struct xnn_qs8_rsum_params; // forward declaration`) is part of it.
    eol = r'[ \t]*(?://[^\n]*|/\*[^\n]*?\*/[ \t]*)?\n'
    src = re.sub(r'^(?:struct|union)[ \t]+xnn_\w+[ \t]*\{[^\n]*\}[ \t]*;' + eol, '', src, flags=re.M)   # on one line
    src = re.sub(r'^(?:struct|union)[ \t]+xnn_\w+[ \t]*\{.*?^\};' + eol, '', src, flags=re.M | re.S)
    src = re.sub(r'^(?:struct|union)[ \t]+xnn_\w+[ \t]*;' + eol, '', src, flags=re.M)
    # The kernel's own definitions of XNNPACK's branch hints: the harness provides XNNPACK's (a redefinition without
    # the outer parentheses breaks `if XNN_UNLIKELY(c) stmt;`). They pass their condition through, so which
    # definition is used does not change what the kernel computes.
    src = re.sub(r'^[ \t]*#[ \t]*(?:define|undef)[ \t]+XNN_(?:LIKELY|UNLIKELY|UNPREDICTABLE)\b[^\n]*\n', '', src, flags=re.M)
    # Register pinning (`register vfloat32m2_t v asm("v8") = ...`) is a code-generation hint, and `register` is not
    # C++17 (the harness's language): the declaration without it declares the same variable.
    src = re.sub(r'\b(\w+)\s+(?:asm|__asm__)\s*\(\s*"[^"]*"\s*\)(?=\s*[=;,)])', r'\1', src)
    src = re.sub(r'^([ \t]*)register[ \t]+(?=(?:const[ \t]+|volatile[ \t]+)*[A-Za-z_])', r'\1', src, flags=re.M)
    # Alias declarations (`void test_neon(...) __attribute__((weak, alias("test_rvv")));`) only give a function
    # another name, and inside the harness's namespace the target's symbol is mangled, so they cannot be kept. If
    # the kernel calls one of those names itself, the harness no longer compiles (a refusal, never a wrong proof).
    attr = r'(?:[^;{}()]|\([^;{}()]*\))*'
    src = re.sub(r'^[^\n;{}#]*\([^;{}]*\)\s*__attribute__\s*\(\(' + attr + r'\balias\s*\(\s*"\w+"\s*\)' + attr +
                 r'\)\)\s*;[^\n]*(?:\n|\Z)', '', src, flags=re.M)
    return re.sub(r'^[ \t]*#[ \t]*include[ \t]*"xnnpack/[\w/]+\.h"[^\n]*$', '', src, flags=re.M)


def _neon_void_load_adapters(src: str) -> str:
    """Accept C void-pointer loads while retaining the typed symbolic shims."""
    loads = (("vld1q_s32", "int32x4_t", "int32_t"),
             ("vld1_s8", "int8x8_t", "int8_t"),
             ("vld1q_f32", "float32x4_t", "float"))
    return "\n".join(
        f"using salt::{fn};\ninline {ret} {fn}(const void* p) "
        f"{{ return salt::{fn}(static_cast<const {elem}*>(p)); }}"
        for fn, ret, elem in loads if re.search(rf"\b{fn}\s*\(", src))


def _split_body(src: str):
    """(head, body) split at the function's opening brace. Matches the actual
    `void test_*(` definition — robust to 'test_' mentions in comments."""
    m = re.search(r'\bvoid\s+test_\w*\s*\(', src)
    brace = src.find("{", m.end()) if m else -1
    return (src[:brace], src[brace:]) if brace != -1 else (src, "")


def _scalar_broadcast_inputs(src: str, input_names) -> set:
    """Inputs read as a scalar (`*name`) in the function *body* — a broadcast
    constant, not a vector loaded via an intrinsic. The signature's `T* name`
    decl is excluded by scanning only after the opening brace."""
    _, body = _split_body(src)
    return {n for n in input_names if re.search(rf'\*\s*{re.escape(n)}\b', body)}


_SCALAR_TYPE = re.compile(r'^(u?int(8|16|32|64)_t|uint|int|short|long|float|double|size_t|u?char)$')


def _split_top_commas(text: str) -> list:
    """`text` split at the commas outside parentheses, brackets and braces."""
    parts, depth, cur = [], 0, ""
    for ch in text:
        depth += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(ch, 0)
        if ch == "," and depth == 0:
            parts.append(cur); cur = ""
        else:
            cur += ch
    return parts + [cur]


_CAST_TYPE = re.compile(r'\(\s*(?:const\s+)?(u?int(?:8|16|32|64)_t|unsigned\s+int|unsigned|int|short|long|float|double|'
                        r'size_t|(?:unsigned\s+|signed\s+)?char)\s*\)')


def _cast_operand_end(l: str, k: int) -> int:
    """End of the unary expression a C cast applies to, starting at l[k] (or k if there is none)."""
    def group(j, o, c):
        d = 0
        while j < len(l):
            d += (l[j] == o) - (l[j] == c)
            j += 1
            if d == 0:
                return j
        return -1
    while k < len(l) and l[k] in " \t":
        k += 1
    while k < len(l) and l[k] in "-~!+" and l[k:k + 2] not in ("--", "++"):
        k += 1
        while k < len(l) and l[k] in " \t":
            k += 1
    if k < len(l) and l[k] == "(":
        e = group(k, "(", ")")
        return e if e > 0 else k
    m = re.match(r"[A-Za-z_]\w*", l[k:])
    if not m:
        return k
    e = k + m.end()
    if e < len(l) and l[e] == "<" and re.match(r"<\s*[\w:\s,*]+>", l[e:]):        # salt_as<int32_t>
        e = group(e, "<", ">")
    while e > 0 and e < len(l):
        if l[e] == "(":
            e = group(e, "(", ")")
        elif l[e] == "[":
            e = group(e, "[", "]")
        elif l[e] == "." or l[e:e + 2] == "->":
            m = re.match(r"(?:\.|->)\s*(?:template\s+)?[A-Za-z_]\w*(?:<[^<>()]*>)?", l[e:])
            if not m:
                break
            e += m.end()
        else:
            break
    return e if e > 0 else k


def _rewrite_symbolic_casts(l: str, symbolic) -> str:
    """`(T) operand` -> `salt_as<T>(operand)` where the operand holds one of the `symbolic` names or a salt_ value."""
    sym = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in sorted(symbolic, key=len, reverse=True)) +
                     r")\b|\bsalt_(?:as|param|scalar(?:_postinc)?)\b")
    for _ in range(16):                                   # nested casts, innermost (rightmost) first
        best = None
        for m in _CAST_TYPE.finditer(l):
            e = _cast_operand_end(l, m.end())
            operand = l[m.end():e].strip()
            if operand and sym.search(operand) and not re.fullmatch(r"\(\s*\)", operand):
                best = (m, e, operand)
        if not best:
            return l
        m, e, operand = best
        l = l[:m.start()] + "salt_as<" + re.sub(r"\s+", " ", m.group(1)) + ">(" + operand + ")" + l[e:]
    return l


def _rebind_tainted_locals(body: str, seeds=()) -> str:
    """A scalar-typed local initialized (transitively) from a symbolic scalar now
    holds a SymbolicScalar, so it becomes `auto x = salt_as<T>(init)`, which keeps
    C's conversion to the declared type. Iterates to a fixpoint
    over the taint set, which starts from `seeds` (names known to hold one). Vector
    types (int32x4_t, …) are left alone — their shims already carry symbolic lanes."""
    lines = body.split("\n")
    # A declaration continued over several lines is one statement: join it.
    joined, i = [], 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r'^\s*(const\s+)?(\w+)\s+\w+\s*=', ln)
        if m and _SCALAR_TYPE.match(m.group(2)) and ";" not in ln:
            j = i
            while j + 1 < len(lines) and ";" not in lines[j]:
                j += 1
            ln = " ".join([lines[i].rstrip()] + [l.strip() for l in lines[i + 1:j + 1]])
            i = j
        joined.append(ln)
        i += 1
    # `T a = x, b = y;` declares a and b as `T a = x; T b = y;` do (scalar declarators): one per line, so each is
    # rebound on its own (as one, `salt_as<T>(x, b = y)` would not even compile). A macro's trailing `\` goes on each.
    lines = []
    for ln in joined:
        m = re.match(r'^(\s*)((?:const\s+)?(\w+))\s+(\w+\s*=.*);(\s*(?://.*|\\)?)$', ln)
        parts = _split_top_commas(m.group(4)) if m and _SCALAR_TYPE.match(m.group(3)) else []
        if len(parts) > 1 and all(re.match(r'\s*\w+\s*=(?!=)', q) for q in parts):
            cont = " \\" if m.group(5).strip() == "\\" else ""
            for k, q in enumerate(parts):
                lines.append(f"{m.group(1)}{m.group(2)} {q.strip()};" + (m.group(5) if k == len(parts) - 1 else cont))
            continue
        lines.append(ln)
    tainted: set = set(seeds)
    changed = True
    while changed:
        changed = False
        for i, ln in enumerate(lines):
            m = re.match(r'^(\s*)(const\s+)?(\w+)\s+(\w+)\s*=\s*(.+?);(\s*(?://.*|\\)?)$', ln)   # a trailing // comment (or a macro's \) is kept
            if not m:
                continue
            indent, c, typ, name, init, tail = m.groups()
            if typ == "auto" or not _SCALAR_TYPE.match(typ):   # a name redeclared in another scope is rebound too
                continue
            if ("salt_scalar" in init or "salt_param" in init or
                    any(re.search(rf'\b{re.escape(t)}\b', init) for t in tainted)):
                # The declared type still converts the value, as C does (truncation included);
                # an init already cast to that type needs no second conversion.
                done = re.fullmatch(r'salt_scalar\([^()]*\)\.cast<' + re.escape(typ) + r'>\(\)', init.strip())
                lines[i] = (f'{indent}{c or ""}auto {name} = {init};{tail}' if done else
                            f'{indent}{c or ""}auto {name} = salt_as<{typ}>({init});{tail}')
                tainted.add(name)
                changed = True
        # A scalar declared without an initializer (`int16_t a0, a1;`) and later assigned a symbolic value holds
        # one from then on: it is declared as a SymbolicScalar of its type (holding 0 until its first assignment,
        # which C requires before any read), and each assignment to it converts to that type, as C's does.
        for i, ln in enumerate(lines):
            m = re.match(r'^(\s*)(\w+)\s+(\w+(?:\s*,\s*\w+)*)\s*;(\s*(?://.*)?)$', ln)
            if not m or not _SCALAR_TYPE.match(m.group(2)) or m.group(2) in ("return", "goto"):
                continue
            indent, typ, names_, tail = m.groups()
            names = [n.strip() for n in names_.split(",")]
            assign = lambda n: re.compile(rf'^(\s*){re.escape(n)}\s*=(?!=)\s*(.+?);(\s*(?://.*)?)$')
            hot = [n for n in names if any(
                (a := assign(n).match(l)) and ("salt_scalar" in a.group(2) or "salt_param" in a.group(2) or
                                               any(re.search(rf'\b{re.escape(t)}\b', a.group(2)) for t in tainted))
                for l in lines)]
            if not hot:
                continue
            lines[i] = f'{indent}' + "; ".join(f'SymbolicScalar<{typ}> {n}(({typ}) 0)' for n in names) + f';{tail}'
            for j, l in enumerate(lines):
                for n in names:
                    a = assign(n).match(l)
                    if a and "salt_as<" not in a.group(2):
                        lines[j] = f'{a.group(1)}{n} = salt_as<{typ}>({a.group(2)});{a.group(3)}'
            tainted.update(names)
            changed = True
    # A C cast of a symbolic value (`(int) output_min - 128`, `(int32_t)-salt_as<uint32_t>(shift)`, the outer cast
    # of `(uint32_t)(int32_t) zp`): the same conversion, salt_as<T>(operand) (a SymbolicScalar has no native
    # conversion, so the cast would not compile). The operand is the cast's own, as C parses it (unary operators,
    # then a name with its template arguments, calls, subscripts and members, or a parenthesized expression), so
    # the arithmetic around it is unchanged; salt_as of a native value is a static_cast. Innermost casts first.
    if tainted:
        lines = [_rewrite_symbolic_casts(l, tainted) for l in lines]
    return "\n".join(lines)


def _pointer_aliases(body: str, names) -> list:
    """`names` plus every local pointer initialized from one of them
    (`const int8_t* ak = a0;`), to a fixed point. The scalar-read rewrites must
    follow a copied pointer: reads through it would otherwise see the
    zero-filled backing array, not the symbolic input."""
    out = list(dict.fromkeys(names))
    decl = re.compile(r'\b(?:const\s+)?\w+\s*\*\s*(?:const\s+|restrict\s+)*(\w+)\s*=\s*([^;]+);')
    changed = True
    while changed:
        changed = False
        for m in decl.finditer(body):
            if m.group(1) not in out and any(re.search(rf'\b{re.escape(n)}\b', m.group(2)) for n in out):
                out.append(m.group(1))
                changed = True
    return out


def _native_reads(src: str, names) -> str | None:
    """`input via alias` for a remaining native read (`p[i]`, `*p`) of a symbolic
    input through any alias, after the accessor rewrites; None if there is none."""
    head, body = _split_body(src)
    for n in names:
        for a in _pointer_aliases(body, [n]):
            rest = re.sub(rf'\b(?:const\s+)?\w+\s*\*+\s*(?:const\s+|restrict\s+)*{re.escape(a)}\b', ' ', body)
            if re.search(rf'(?<![&\w.>])\b{re.escape(a)}\s*\[|\*\s*{re.escape(a)}\b', rest):
                return f"{n} via {a}" if a != n else n
    return None


def _rewrite_scalar_broadcast(src: str, names) -> str:
    """Rewrite scalar reads of broadcast inputs (in the body only) to a symbolic
    accessor: `(T) *name` -> `salt_scalar(name).cast<T>()`, `*name` -> `salt_scalar(name)`."""
    if not names:
        return src
    head, body = _split_body(src)
    if not body:
        return src
    for n in _pointer_aliases(body, names):
        # Local pointer declarations contain '*name' too; retain those verbatim.
        parts = re.split(
            rf'(\b(?:const\s+)?\w+\s*\*\s*(?:restrict\s+)?{re.escape(n)}\b(?=\s*[=;]))',
            body)
        # Preserve the increment on the pointer, not on the loaded scalar.
        ptr = rf'{re.escape(n)}\b(\s*\+\+)?'
        for i in range(0, len(parts), 2):
            part = re.sub(rf'\(\s*(\w+)\s*\)\s*\*\s*{ptr}',
                          lambda m: f'salt_scalar({n}{m[2] or ""}).cast<{m[1]}>()', parts[i])
            part = re.sub(rf'\*\s*{ptr}',
                          lambda m: f'salt_scalar({n}{m[1] or ""})', part)
            if re.search(rf'\*\s*{re.escape(n)}\b', part):
                raise NotImplementedError(f"unrewritten scalar read of {n} remains "
                                          "(unhandled access pattern)")
            parts[i] = part
        body = "".join(parts)
    body = _rebind_tainted_locals(body)  # propagate symbolic taint to dependent scalar locals
    return head + body


def _rewrite_deref_reads(body: str, n: str) -> str:
    """Rewrite `*p++` (read, then advance) and `*p` reads of pointer `n`, with an optional `(T)` cast, to
    salt_scalar_postinc / salt_scalar. A `*` after a word, `)` or `]` (a declaration `int8_t *p`, a product) is
    not a dereference, nor a store `*p = v` / `*p += v` / `*p++ = v`, nor an operand of sizeof."""
    e = re.escape(n)
    # not a store: `*p = v`, a compound `*p += v` (<<=, >>=, ...), `*p++ = v`
    pat = re.compile(rf'(?:\(\s*(\w+)\s*\)\s*)?\*\s*{e}\b(\s*\+\+)?'
                     rf'(?!\s*(?:\[|(?:[-+*/%&|^]|<<|>>)?=(?!=)|\+\+\s*=(?!=)))')
    def sub(m):
        before = body[:m.start()].rstrip()
        if m[1] is None and (before[-1:].isalnum() or before[-1:] in "_)]" or re.search(r'\bsizeof\s*\($', before)):
            return m[0]
        read = f"salt_scalar_postinc({n})" if m[2] else f"salt_scalar({n})"
        return f"{read}.cast<{m[1]}>()" if m[1] else read
    return pat.sub(sub, body)


def _rewrite_index_reads(src: str, names) -> str:
    """Rewrite scalar element reads `p[i]` of symbolic-buffer pointers to
    salt_scalar accessors (int element types have no NaN-payload sentinel)."""
    if not names:
        return src
    head, body = _split_body(src)
    if not body:
        return src
    for n in _pointer_aliases(body, names):
        body = re.sub(rf'\(\s*(\w+)\s*\)\s*{re.escape(n)}\s*\[\s*([^\]]+?)\s*\]',
                      rf'salt_scalar({n} + (\2)).cast<\1>()', body)   # (T) p[i] → .cast<T>()
        body = re.sub(rf'\b{re.escape(n)}\s*\[\s*([^\]]+?)\s*\]',
                      rf'salt_scalar({n} + (\1))', body)
        body = _rewrite_deref_reads(body, n)
    body = _rebind_tainted_locals(body)
    return head + body


def _rewrite_scalar_stores(src: str, names, value=None) -> str:
    """Rewrite scalar stores into symbolic output buffers (`out[i] = v;`, `*out = v;`, `*out++ = v;`) to
    `salt_deref(...) = salt_as<T>(v)`, T the buffer's element type. A native store reaches only the backing C
    array, which the symbolic model never reads; `v` may be a constant or a symbolic value."""
    if not names:
        return src
    head, body = _split_body(src)
    if not body:
        return src
    elem = lambda p: f"std::remove_cv_t<std::remove_reference_t<decltype(*({p}))>>"
    val = value or (lambda p, v: f"salt_as<{elem(p)}>({v})")      # value(pointer, rhs) -> the stored C++ value
    for n in _pointer_aliases(body, names):
        e = re.escape(n)
        body = re.sub(rf'(?<![\w.>])\b{e}\s*\[\s*([^\]]+?)\s*\]\s*=(?!=)\s*([^;]+);',
                      lambda m: f"salt_deref({n} + ({m[1]})) = {val(n, m[2].strip())};", body)
        body = re.sub(rf'(?<![\w)\]])\*\s*{e}\s*\+\+\s*=(?!=)\s*([^;]+);',
                      lambda m: f"salt_deref({n}) = {val(n, m[1].strip())}; {n} += 1;", body)
        body = re.sub(rf'(?<![\w)\]])\*\s*{e}\s*=(?!=)\s*([^;]+);',
                      lambda m: f"salt_deref({n}) = {val(n, m[1].strip())};", body)
    return head + body


def _rewrite_fp_scratch(src: str, outputs) -> tuple:
    """A candidate that RVV-stores into a local float array (`float t[4]; __riscv_vse32_v_f32m1(t, v, vl);`)
    and then copies elements out in C (`o[0] = t[0];`): the array becomes a scoped symbolic buffer
    (SaltScratch), its element reads and stores symbolic, and the C stores into the float outputs symbolic
    (salt_fp_value decodes a native float). Only arrays declared without an initializer (an initializer's
    values would not be in the fresh buffer). Returns the source and the arrays rewritten (none: unchanged)."""
    head, body = _split_body(src)
    if not body:
        return src, []
    decl = re.compile(r'\bfloat\s+((?:\w+\s*\[[^\]=;]+\]\s*,\s*)*\w+\s*\[[^\]=;]+\])\s*;')
    stored = lambda n: re.search(rf'__riscv_vs\w*\(\s*{re.escape(n)}\b', body)
    arrays, kept = [], []
    def scratch(m):
        names = [re.match(r'\s*(\w+)', d)[1] for d in m[1].split(",")]
        mine = [n for n in names if stored(n)]
        if not mine:
            return m[0]
        arrays.extend(n for n in mine if n not in arrays)
        kept.append(m[0] + "".join(f' SaltScratch _salt_scratch_{n}({n}, sizeof({n}), "{n}");' for n in mine))
        return f"/*SALT_SCRATCH_DECL_{len(kept) - 1}*/"     # shielded from the read and store rewrites
    body = decl.sub(scratch, body)
    if not arrays:
        return src, []
    fpv = lambda ptr, v: f"salt_fp_value({v})"
    out = _rewrite_scalar_stores(head + body, arrays, fpv)       # stores first: an lvalue is not a read
    out = _rewrite_index_reads(out, arrays)
    out = _rewrite_scalar_stores(out, outputs, fpv)
    h, b = _split_body(out)
    b = _rebind_tainted_locals(b)
    for i, d in enumerate(kept):
        b = b.replace(f"/*SALT_SCRATCH_DECL_{i}*/", d)
    return h + b, arrays


def _rewrite_scalar_movs(src: str) -> str:
    """Rewrite scalar buffer-to-buffer moves `*dst = *src++;` (packw remainder
    paths) to symbolic accessors: `salt_deref(dst) = salt_scalar(src); src += 1;`."""
    head, body = _split_body(src)
    if not body:
        return src
    return head + re.sub(r'\*\s*(\w+)\s*=\s*\*\s*(\w+)\s*\+\+\s*;',
                         r'salt_deref(\1) = salt_scalar(\2); \2 += 1;', body)


# ---- C-level tests on floats the verifier cannot track ----------------------------------------------
# A kernel runs natively; only intrinsics and the rewritten scalar reads (salt_scalar, salt_param) see symbolic
# values. A float read natively from an input or weight buffer is a NaN-payload sentinel, so a C `if`, `?:`,
# comparison or isnan-style test on it, directly or through locals, an int conversion or a helper, would pick a
# branch for that one bit pattern: the harness refuses such a kernel instead. Reads through a param-struct
# pointer are trackable (symbolic, rewritten, or concrete), as are literals.
_CTOK = re.compile(r"""\s+|//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|
    (?P<num>0[xX][0-9a-fA-F.pP+-]+[fFlLuU]*|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFlLuU]*)|
    (?P<id>[A-Za-z_]\w*)|(?P<op><<=|>>=|->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%<>=!&|^~?:;,.(){}\[\]#])""",
                   re.S | re.X)
_CMP_OPS = {"==", "!=", "<", "<=", ">", ">="}
_FP_TESTS = {"isnan", "isinf", "isfinite", "isnormal", "signbit", "fpclassify", "__builtin_isnan",
             "__builtin_isinf", "__builtin_isfinite", "__builtin_signbit", "__builtin_fpclassify"}
_FLOAT_TYPES = {"float", "double", "_Float16", "__fp16", "xnn_float16", "float16_t", "float32_t"}


def _ctokens(src):
    """The tokens of `src` for the C-level analyses, with the kernel's own macros expanded (a test or a native use
    inside a macro body is seen where it is used). Preprocessor lines are dropped and every #if branch kept, a
    superset of what is compiled; a macro whose body stringizes or pastes (# ##), or is variadic, stays a call,
    which the analyses treat conservatively."""
    src = re.sub(r"\\\n", " ", src)
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    macros = {}
    for m in re.finditer(r"^[ \t]*#[ \t]*define[ \t]+(\w+)(\([^)]*\))?[ \t]*(.*)$", src, re.M):
        params = [p.strip() for p in m[2][1:-1].split(",")] if m[2] else None
        body = re.sub(r"//.*$", "", m[3])
        if "#" in body or (params and any(p == "..." or p.endswith("...") for p in params)):
            continue
        macros[m[1]] = ([p for p in params if p] if params is not None else None, _raw_ctokens(body))
    toks = _raw_ctokens(re.sub(r"^[ \t]*#.*$", "", src, flags=re.M))
    return _expand_macros(toks, macros, 0)


def _raw_ctokens(src):
    return [("num", m["num"]) if m["num"] else ("id", m["id"]) if m["id"] else ("op", m["op"])
            for m in _CTOK.finditer(src) if m["num"] or m["id"] or m["op"]]


def _expand_macros(toks, macros, depth, active=frozenset()):
    """Expand `macros` (name -> (params or None, body tokens)) in `toks`; a macro is not expanded inside itself."""
    if not macros or depth > 8:
        return toks
    out, i = [], 0
    while i < len(toks):
        k, v = toks[i]
        if k != "id" or v not in macros or v in active:
            out.append(toks[i]); i += 1
            continue
        params, body = macros[v]
        if params is None:                                  # object-like
            out += _expand_macros(body, macros, depth + 1, active | {v}); i += 1
            continue
        if i + 1 >= len(toks) or toks[i + 1] != ("op", "("):
            out.append(toks[i]); i += 1
            continue
        d, j = 0, i + 1
        while j < len(toks):
            d += {"(": 1, ")": -1}.get(toks[j][1], 0) if toks[j][0] == "op" else 0
            if d == 0:
                break
            j += 1
        args = _csplit(toks[i + 2:j]) if j > i + 2 else []
        if len(args) != len(params):
            out.append(toks[i]); i += 1                      # not this macro's arity: left as a call
            continue
        sub = dict(zip(params, args))
        exp = []
        for t in body:
            exp += [("op", "(")] + sub[t[1]] + [("op", ")")] if t[0] == "id" and t[1] in sub else [t]
        out += _expand_macros(exp, macros, depth + 1, active | {v})
        i = j + 1
    return out


def _prm_ids(prm):
    """A parameter's identifiers up to its first `[` (an array parameter's bound, `x[XNN_MIN_ELEMENTS(1)]`, is no
    part of its name): the name is the last."""
    end = prm.index(("op", "[")) if ("op", "[") in prm else len(prm)
    return [v for k, v in prm[:end] if k == "id"]


_ATTR_MACRO = re.compile(r"[A-Z_][A-Z0-9_]*|__attribute__")


def _param_list_before(toks, i):
    """(index of the `(`, index of the `)`) of the parameter list a function body opening at toks[i] follows, past
    attribute macros between the two (`) XNN_OOB_READS {`, `) __attribute__((x)) {`, `) XNN_MIN_ELEMENTS(1) {`),
    or None when toks[i] does not open a function body."""
    j = i - 1
    while j >= 0 and toks[j][0] == "id" and _ATTR_MACRO.fullmatch(toks[j][1]):
        j -= 1
    while j >= 0 and toks[j] == ("op", ")"):
        d, k = 0, j
        while k >= 0:
            d += {")": 1, "(": -1}.get(toks[k][1], 0) if toks[k][0] == "op" else 0
            if d == 0:
                break
            k -= 1
        if k < 1:
            return None
        before = k - 1                            # an attribute's argument group: the list is further back
        while before >= 0 and toks[before][0] == "id" and _ATTR_MACRO.fullmatch(toks[before][1]):
            before -= 1
        if before < k - 1 and before >= 0 and toks[before] == ("op", ")"):
            j = before
            continue
        return (k, j) if toks[k - 1][0] == "id" else None
    return None


def _cfunctions(toks):
    """(name, parameter tokens, body tokens) of each function definition at file scope (attribute macros after
    the parameter list, as XNNPACK's `XNN_OOB_READS`, included)."""
    out, i, depth = [], 0, 0
    while i < len(toks):
        if toks[i] == ("op", "{") and depth == 0 and i and (pl := _param_list_before(toks, i)):
            k, pend = pl
            d, e = 0, i
            while e < len(toks):
                d += {"{": 1, "}": -1}.get(toks[e][1], 0) if toks[e][0] == "op" else 0
                if d == 0:
                    break
                e += 1
            out.append((toks[k - 1][1], toks[k + 1:pend], toks[i + 1:e]))
            i = e + 1
            continue
        depth += {"{": 1, "}": -1}.get(toks[i][1], 0) if toks[i][0] == "op" else 0
        i += 1
    return out


def _csplit(ts, sep=","):
    parts, cur, d = [], [], 0
    for t in ts:
        if t[0] == "op" and t[1] in "([":
            d += 1
        elif t[0] == "op" and t[1] in ")]":
            d -= 1
        if t == ("op", sep) and d == 0:
            parts.append(cur)
            cur = []
        else:
            cur.append(t)
    return parts + [cur] if cur else parts


def _c_float_tests(src: str) -> list:
    """The C-level tests on untrackable floats in `src` (see above), as `function: tokens` strings."""
    fns = _cfunctions(_ctokens(src))
    names = {f[0] for f in fns}
    structp, fptr, flt, clean, stmts = {}, {}, {}, {}, {}
    for name, ptoks, body in fns:
        structp[name], fptr[name], flt[name], clean[name] = set(), set(), set(), set()
        for prm in _csplit(ptoks):
            ids = _prm_ids(prm)
            if not ids:
                continue
            if "struct" in ids and ("op", "*") in prm:
                structp[name].add(ids[-1])
            elif any(v in _FLOAT_TYPES for v in ids):
                (fptr if ("op", "*") in prm else flt)[name].add(ids[-1])
        st, cur = [], []
        for t in body:
            if t[0] == "op" and t[1] in ";{}":
                st.append(cur)
                cur = []
            else:
                cur.append(t)
        stmts[name] = st + [cur]

    def decl(st):                                 # (type ids, name, init, is pointer) of `T name = init`
        if ("op", "=") not in st:
            return None
        i = st.index(("op", "="))
        if i < 2 or st[i - 1][0] != "id":
            return None
        return [v for k, v in st[:i - 1] if k == "id"], st[i - 1][1], st[i + 1:], ("op", "*") in st[:i - 1]
    for name, _, _ in fns:
        for st in stmts[name]:
            d = decl(st)
            if d and any(v in _FLOAT_TYPES for v in d[0]):
                (fptr if d[3] else flt)[name].add(d[1])

    def trackable(fn, ts):                        # only literals, param-struct reads, clean names
        for i, (k, v) in enumerate(ts):
            if k != "id" or v in _FLOAT_TYPES or v in ("const", "volatile", "sizeof"):
                continue
            if v in structp[fn] or v in clean[fn]:
                continue
            if i and ts[i - 1][1] in (".", "->"):   # a field name
                continue
            if i + 1 < len(ts) and ts[i + 1] == ("op", "(") and v not in names:
                continue                            # a cast-like or library call's name; its args are checked
            return False
        return True
    for _ in range(6):                            # fixpoint: clean locals, then clean helper parameters
        for name, _, _ in fns:
            for st in stmts[name]:
                d = decl(st)
                if d and not d[3] and trackable(name, d[2]) and not any(
                        v in _FP_TESTS or (v not in names and i + 1 < len(d[2]) and d[2][i + 1] == ("op", "("))
                        for i, (k, v) in enumerate(d[2]) if k == "id" and v not in ("float", "double")):
                    clean[name].add(d[1])
        for hname, ptoks, _ in fns:
            for idx, prm in enumerate(_csplit(ptoks)):
                ids = _prm_ids(prm)
                if not ids or not any(v in _FLOAT_TYPES for v in ids) or ("op", "*") in prm:
                    continue
                sites, ok = 0, True
                for cname, _, body in fns:
                    for i in range(len(body) - 1):
                        if body[i] == ("id", hname) and body[i + 1] == ("op", "("):
                            d, j = 0, i + 1
                            while j < len(body):
                                d += {"(": 1, ")": -1}.get(body[j][1], 0) if body[j][0] == "op" else 0
                                if d == 0:
                                    break
                                j += 1
                            args = _csplit(body[i + 2:j])
                            sites += 1
                            ok = ok and idx < len(args) and trackable(cname, args[idx])
                referenced = any(("id", hname) in b for n2, _, b in fns if n2 != hname)
                if ok and (sites or not referenced):  # every call passes a trackable value, or it is dead code
                    clean[hname].add(ids[-1])
    found = []
    for name, _, _ in fns:
        taint = {v for v in flt[name] if v not in clean[name]}

        def tainted(ts):
            for i, (k, v) in enumerate(ts):
                if k == "id" and v in taint:
                    return True
                if k == "id" and v in fptr[name] and (
                        (i + 1 < len(ts) and ts[i + 1] == ("op", "[")) or
                        (i and ts[i - 1] == ("op", "*") and (i < 2 or ts[i - 2][0] == "op"))):
                    return True
            return False
        # A name only ever declared as a pointer (`const float* i0 = input[0];`) holds an address, not a float: a
        # test on it compares concrete pointers (`i0 == zero`), and a float read through it is caught as a deref.
        ptr_only = _pointer_only_names(name, fns)
        for _ in range(6):                        # through any local or assignment, whatever its type
            for st in stmts[name]:
                d = decl(st)
                if d and not d[3] and tainted(d[2]):
                    taint.add(d[1])
                elif len(st) > 2 and st[0][0] == "id" and st[1][0] == "op" and st[1][1].endswith("=") \
                        and st[1][1] not in _CMP_OPS and st[0][1] not in ptr_only and tainted(st[2:]):
                    taint.add(st[0][1])
        for st in stmts[name]:
            for i, (k, v) in enumerate(st):
                if not ((k == "op" and v in _CMP_OPS) or
                        (k == "id" and v in _FP_TESTS and i + 1 < len(st) and st[i + 1] == ("op", "("))):
                    continue
                lo, d = i - 1, 0                  # the operands: same paren depth, up to && || ? : , =
                while lo >= 0:
                    t = st[lo]
                    if t[0] == "op" and t[1] in ")]":
                        d += 1
                    elif t[0] == "op" and t[1] in "([":
                        if d == 0:
                            break
                        d -= 1
                    elif d == 0 and t[0] == "op" and t[1] in ("&&", "||", "?", ":", ",", "="):
                        break
                    lo -= 1
                hi, d = i + 1, 0
                while hi < len(st):
                    t = st[hi]
                    if t[0] == "op" and t[1] in "([":
                        d += 1
                    elif t[0] == "op" and t[1] in ")]":
                        if d == 0:
                            break
                        d -= 1
                    elif d == 0 and t[0] == "op" and t[1] in ("&&", "||", "?", ":", ",", "="):
                        break
                    hi += 1
                if tainted(st[lo + 1:hi]):
                    found.append(f"{name}: " + " ".join(v for _, v in st[lo + 1:hi]))
    return found


# ---- Native uses of values the symbolic model cannot follow ----------------------------------------------------
# What a kernel computes natively is invisible to the model; the native-access guard (core/native_guard.hpp)
# refuses at run time a native store into a buffer and native float arithmetic on a sentinel (a symbolic float read
# natively, or extracted from a vector), which every such operation quiets. What the run-time check cannot see is
# refused here, for two kinds of native value: 'f', a float that may be a sentinel (a read through a float pointer or
# array, a float extract, or anything computed from one), and 'i', an integer extracted from a vector, which the
# model knows only through the reduction side channel (its native value is a marker, so its only modeled use is
# `*out += s`). Sinks: a test (comparison, isnan-style call, condition of if/while/switch/?:, operand of ! && ||),
# an index, the address of a tainted local (a bit-cast), a call to a library function (fabsf keeps a sentinel's
# bits), a conversion of an 'f' value to an integer, a store into a struct member, and an 'i' value handed back to
# an intrinsic. An 'f' value handed to an intrinsic is decoded there (and refused if it was changed natively).
_INT_TYPES = {"char", "short", "int", "long", "unsigned", "signed", "bool", "_Bool", "int8_t", "int16_t", "int32_t",
              "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "size_t", "ssize_t", "ptrdiff_t", "intptr_t",
              "uintptr_t"}
_QUALS = {"const", "volatile", "restrict", "__restrict", "__restrict__", "XNN_RESTRICT", "static", "register",
          "inline", "XNN_INLINE", "extern"}
_FP_EXTRACT = re.compile(r"__riscv_vfmv_f_s_\w+|vgetq?_lane_f(?:16|32|64)|vdup[hsd]_laneq?_f(?:16|32|64)")
_INT_EXTRACT = re.compile(r"__riscv_vmv_x_s_\w+|vgetq?_lane_[su](?:8|16|32|64)|vdup[bhsd]_laneq?_[su](?:8|16|32|64)"
                          r"|v(?:add|max|min)vq?_[su](?:8|16|32|64)|vaddlvq?_[su](?:8|16|32)")
_NEON_INTRINSIC = re.compile(r"v[a-z0-9]+(?:_[a-z0-9]+)+")


def _typeish(v):
    return v in _FLOAT_TYPES or v in _INT_TYPES or v in ("void", "auto") or v.endswith("_t")


def _pointer_only_names(fname, fns) -> set:
    """Names `fname` declares only as pointers (or arrays of them), never as a scalar."""
    for name, ptoks, body in fns:
        if name != fname:
            continue
        ts = list(ptoks) + [("op", ",")] + list(body)
        ptr, scalar = set(), set()
        for i, (k, v) in enumerate(ts):
            if k != "id" or i == 0 or _typeish(v) or v in _QUALS:
                continue
            j = i - 1
            while j >= 0 and ts[j][0] == "id" and ts[j][1] in _QUALS:
                j -= 1
            if j >= 0 and ts[j] == ("op", "*"):
                while j >= 0 and (ts[j] == ("op", "*") or (ts[j][0] == "id" and ts[j][1] in _QUALS)):
                    j -= 1
                if j >= 0 and ts[j][0] == "id" and (_typeish(ts[j][1]) or (j and ts[j - 1][1] in ("struct", "union"))):
                    ptr.add(v)
            elif j >= 0 and ts[j][0] == "id" and (_typeish(ts[j][1]) or (j and ts[j - 1][1] in ("struct", "union"))):
                scalar.add(v)
        return ptr - scalar
    return set()


def _struct_typed_names(ptoks, body, structs) -> set:
    """The names a function declares (a parameter or a local, a pointer, an array or a copy) with the struct or
    union type of one of `structs`: `const struct xnn_f32_lrelu_params* restrict params`, `params[restrict 1]`."""
    out = set()
    for ts in (list(ptoks), list(body)):
        for i, (k, v) in enumerate(ts):
            if k == "id" and v in structs and i and ts[i - 1] in (("id", "struct"), ("id", "union")):
                j = i + 1
                while j < len(ts) and (ts[j] == ("op", "*") or (ts[j][0] == "id" and ts[j][1] in _QUALS)):
                    j += 1
                if j < len(ts) and ts[j][0] == "id" and not (j + 1 < len(ts) and ts[j + 1] == ("op", "(")):
                    out.add(ts[j][1])
    return out


def _member_read_end(ts, i):
    """For a struct-typed name at ts[i]: the index of the `.`/`->` of a native read of one of its fields
    (`p->scalar.slope`, `q[0].zero_point`), or None (`&p->scalar.slope`, a pointer an intrinsic loads through, the
    bare name, a declaration)."""
    amp = lambda j: ts[j] == ("op", "&") and (j == 0 or ts[j - 1][0] == "op" and ts[j - 1][1] not in (")", "]"))
    if i and (amp(i - 1) or (i > 1 and ts[i - 1] == ("op", "(") and amp(i - 2))):
        return None
    e = i + 1
    while e < len(ts) and ts[e] == ("op", "["):
        d = 0
        for j in range(e, len(ts)):
            d += {"[": 1, "]": -1}.get(ts[j][1], 0) if ts[j][0] == "op" else 0
            if d == 0:
                break
        e = j + 1
    return e if e < len(ts) and ts[e] in (("op", "->"), ("op", ".")) else None


def _native_param_reads(src: str, structs) -> list:
    """Every native read left in `src` of a field of a param struct of a type in `structs`, as `function: tokens`.
    With the struct bound symbolically (bind: its backing bytes are zero) such a read sees 0, not the symbolic
    value: only salt_param reads, and loads through `&p->field`, follow it."""
    out = []
    for fname, ptoks, body in _cfunctions(_ctokens(src)) if structs else ():
        names = _struct_typed_names(ptoks, body, structs)
        for i, (k, v) in enumerate(body):
            if k == "id" and v in names and (e := _member_read_end(body, i)) is not None:
                out.append(f"{fname}: a native read of a param field: " + " ".join(t for _, t in body[i:e + 4])[:120])
    return list(dict.fromkeys(out))


def _native_param_uses(src: str, structs) -> list:
    """The native uses (_native_scalar_uses: a test, a conversion, a library call, a bit-cast, an index) of the
    fields read from a param struct of a type in `structs` that the harness binds with NaN-payload sentinels
    (bindScalarRef). Handed to an intrinsic, a field's sentinel is decoded to its symbolic value; tested in C
    (`if (slope < 1.0f)`) it is a NaN, and every comparison takes the NaN branch whatever the symbolic value."""
    if not structs:
        return []
    src = _strip_kernel_redeclarations(src)
    base = set(_native_scalar_uses(src))
    return [h for h in _native_scalar_uses(src, frozenset(structs)) if h not in base]


def _native_scalar_uses(src: str, param_structs=frozenset()) -> list:
    """The native uses of 'f'/'i' values in `src` the model cannot follow (see above), as `function: tokens`.
    With `param_structs`, a native read of a field of a struct of those types is an 'f' value too."""
    fns = _cfunctions(_ctokens(src))
    names = {f[0] for f in fns}
    ret_float = set(re.findall(r"\b(?:float|double|float32_t|float16_t|_Float16)\s+(\w+)\s*\(", src)) & names
    intrinsic = lambda v: v not in names and (v.startswith("__riscv_") or bool(_NEON_INTRINSIC.fullmatch(v)))
    show = lambda ts: " ".join(v for _, v in ts)[:120]

    def group(ts, i):                     # index just past the bracket group opening at ts[i]
        o = ts[i][1]; c = {"(": ")", "[": "]", "{": "}"}[o]; d = 0
        for j in range(i, len(ts)):
            if ts[j] == ("op", o):
                d += 1
            elif ts[j] == ("op", c):
                d -= 1
                if d == 0:
                    return j + 1
        return len(ts)

    def operand_end(ts, e):               # end of the unary expression starting at ts[e]
        while e < len(ts) and ts[e][0] == "op" and ts[e][1] in ("*", "-", "+", "!", "~", "&"):
            e += 1
        if e < len(ts) and ts[e] == ("op", "("):
            return group(ts, e)
        if e < len(ts) and ts[e][0] in ("id", "num"):
            e += 1
        while e < len(ts):
            if ts[e][0] == "op" and ts[e][1] in ("[", "("):
                e = group(ts, e)
            elif ts[e][0] == "op" and ts[e][1] in (".", "->") and e + 1 < len(ts):
                e += 2
            elif ts[e][0] == "op" and ts[e][1] in ("++", "--"):
                e += 1
            else:
                break
        return e

    def statements(body):
        st, cur = [], []
        for t in body:
            if t[0] == "op" and t[1] in ";{}":
                st.append(cur); cur = []
            else:
                cur.append(t)
        return [s for s in st + [cur] if s]

    def decls(fname, ptoks, body):
        """name -> set of (base 'f'/'i'/'o', pointer depth) over every declaration of it."""
        out = {}
        def add(n, base, depth):
            out.setdefault(n, set()).add((base, depth))
        for prm in _csplit(ptoks):
            ids = _prm_ids(prm)
            if not ids:
                continue
            base = "f" if any(v in _FLOAT_TYPES for v in ids) else "i" if any(v in _INT_TYPES for v in ids) else "o"
            add(ids[-1], base, prm.count(("op", "*")) + prm.count(("op", "[")))
        for st in statements(body):
            s = st
            if s[:2] == [("id", "for"), ("op", "(")]:
                s = s[2:]
            i = 0
            while i < len(s) and s[i][0] == "id" and (s[i][1] in _QUALS or s[i][1] in ("struct", "union")):
                i += 1
            if not (i < len(s) and s[i][0] == "id" and (_typeish(s[i][1]) or (i and s[i - 1][1] in ("struct", "union")))):
                continue
            j = i
            while j < len(s) and s[j][0] == "id" and (_typeish(s[j][1]) or s[j][1] in _QUALS):
                j += 1
            if i and s[i - 1][1] in ("struct", "union"):
                j = max(j, i + 1)
            if j >= len(s) or j == i:
                continue
            tids = [v for k, v in s[i:j] if k == "id"]
            base = "f" if any(v in _FLOAT_TYPES for v in tids) else "i" if any(v in _INT_TYPES for v in tids) else "o"
            for dcl in _csplit(s[j:]):
                eq = dcl.index(("op", "=")) if ("op", "=") in dcl else len(dcl)
                head = dcl[:eq]
                ids = [(n, v) for n, (k, v) in enumerate(head) if k == "id" and v not in _QUALS]
                if not ids:
                    continue
                n, v = ids[0]
                if n + 1 < len(head) and head[n + 1] == ("op", "("):
                    continue                          # a call, not a declarator
                add(v, base, head[:n].count(("op", "*")) + head[n:].count(("op", "[")))
        return out

    param_taint = {n: {} for n in names}      # helper -> {param index: kinds}
    found = []
    param_changed = [False]
    for rnd in range(6):
        param_changed[0], found = False, []
        for fname, ptoks, body in fns:
            dt = decls(fname, ptoks, body)
            seedp = _struct_typed_names(ptoks, body, param_structs) if param_structs else set()
            sts = statements(body)
            prms = [_prm_ids(p) for p in _csplit(ptoks)]
            taint = {}                            # scalar name -> kinds
            for idx, ids in enumerate(prms):
                if ids and idx in param_taint[fname]:
                    taint[ids[-1]] = set(param_taint[fname][idx])
            iarr = set()                          # local arrays holding 'i' values
            fptrs = {n: max(d for b, d in s) for n, s in dt.items() if all(b == "f" and d >= 1 for b, d in s)}
            scalar_of = lambda n: n in dt and all(d == 0 for b, d in dt[n])

            def kinds(ts, sinks=None):
                ks, i = set(), 0
                while i < len(ts):
                    k, v = ts[i]
                    if k == "id" and i + 1 < len(ts) and ts[i + 1] == ("op", "(") and \
                            v not in ("if", "while", "for", "switch", "return", "do", "else", "case"):
                        e = group(ts, i + 1)
                        args = _csplit(ts[i + 2:e - 1])
                        if v == "sizeof":
                            pass
                        elif _FP_EXTRACT.fullmatch(v):
                            ks.add("f")
                        elif _INT_EXTRACT.fullmatch(v):
                            ks.add("i")
                        elif intrinsic(v):
                            for a in args:                # an 'f' argument is decoded by the model, an 'i' one is not
                                if "i" in kinds(a, sinks) and sinks is not None:
                                    sinks.append(f"{fname}: an extracted integer passed to {v}: {show(a)}")
                        elif v in names:
                            for idx, a in enumerate(args):
                                ak = kinds(a, sinks)
                                ks |= ak & {"i"}
                                if ak - param_taint[v].get(idx, set()):
                                    param_taint[v].setdefault(idx, set()).update(ak)
                                    param_changed[0] = True
                            if v in ret_float:
                                ks.add("f")
                        elif _typeish(v) or v in _FLOAT_TYPES:
                            for a in args:
                                ak = kinds(a, sinks)
                                if "f" in ak and v in _INT_TYPES and sinks is not None:
                                    sinks.append(f"{fname}: a native float converted to an integer: {v}({show(a)})")
                                ks |= ak
                        else:                     # a library function (fabsf keeps a sentinel's bits)
                            for a in args:
                                ak = kinds(a, sinks)
                                if ak and sinks is not None:
                                    sinks.append(f"{fname}: a native value passed to {v}: {show(a)}")
                                ks |= ak
                        i = e
                        continue
                    if k == "id" and v in seedp and _member_read_end(ts, i) is not None:
                        ks.add("f")
                    elif k == "id" and v in taint:
                        ks |= taint[v]
                    elif k == "id" and (v in fptrs or v in iarr):
                        derefs, j = 0, i - 1
                        while j >= 0 and ts[j] == ("op", "*") and (j == 0 or ts[j - 1][0] == "op" and ts[j - 1][1] not in (")", "]")):
                            derefs += 1; j -= 1
                        e = i + 1
                        while e < len(ts) and ts[e] == ("op", "["):
                            derefs += 1; e = group(ts, e)
                        if derefs >= (fptrs.get(v) or 1):
                            ks.add("f" if v in fptrs else "i")
                    i += 1
                return ks

            nonlocal_changed = [False]
            for _ in range(8):                    # propagation through locals and local arrays
                for st in sts:
                    for i, (k, v) in enumerate(st):
                        if not (k == "op" and v.endswith("=") and v not in _CMP_OPS):
                            continue
                        lhs_end, lo = i, i - 1    # the assigned lvalue: id, subscripts, members, a deref
                        while lo >= 0 and st[lo][1] in ("]", ")") and st[lo][0] == "op":
                            d, lo2 = 0, lo
                            while lo2 >= 0:
                                d += {"]": 1, ")": 1, "[": -1, "(": -1}.get(st[lo2][1], 0) if st[lo2][0] == "op" else 0
                                if d == 0:
                                    break
                                lo2 -= 1
                            lo = lo2 - 1
                        if lo < 0 or st[lo][0] != "id":
                            continue
                        target, rhs = st[lo][1], st[i + 1:]
                        rk = kinds(rhs)
                        if lo + 1 < lhs_end and st[lo + 1] == ("op", "["):
                            if "i" in rk and target not in iarr and target not in fptrs:
                                iarr.add(target); nonlocal_changed[0] = True
                        elif lo + 1 == lhs_end and not (lo and st[lo - 1][1] in (".", "->", "*")) and \
                                (target not in dt or scalar_of(target)):
                            if rk - taint.get(target, set()):
                                taint.setdefault(target, set()).update(rk); nonlocal_changed[0] = True
                        break
            # sinks
            for st in sts:
                sinks = []
                kinds(st, sinks)
                for i, (k, v) in enumerate(st):
                    span = None
                    if (k == "op" and v in _CMP_OPS) or (k == "op" and v in ("&&", "||", "?")) or \
                            (k == "id" and v in _FP_TESTS and i + 1 < len(st) and st[i + 1] == ("op", "(")):
                        lo, d = i - 1, 0
                        while lo >= 0:
                            t = st[lo]
                            if t[0] == "op" and t[1] in ")]":
                                d += 1
                            elif t[0] == "op" and t[1] in "([":
                                if d == 0:
                                    break
                                d -= 1
                            elif d == 0 and t[0] == "op" and t[1] in ("&&", "||", "?", ":", ",", "=", "return"):
                                break
                            lo -= 1
                        hi, d = i + 1, 0
                        while hi < len(st):
                            t = st[hi]
                            if t[0] == "op" and t[1] in "([":
                                d += 1
                            elif t[0] == "op" and t[1] in ")]":
                                if d == 0:
                                    break
                                d -= 1
                            elif d == 0 and t[0] == "op" and t[1] in ("&&", "||", "?", ":", ",", "="):
                                break
                            hi += 1
                        span = st[lo + 1:hi] if v != "?" else st[lo + 1:i]
                    elif k == "id" and v in ("if", "while", "switch") and i + 1 < len(st):
                        j = i + 1
                        if st[j][0] == "id" and j + 1 < len(st) and st[j + 1] == ("op", "("):
                            j += 1                # `if XNN_UNLIKELY(c)`
                        if st[j] == ("op", "("):
                            span = st[i + 1:group(st, j)]
                    elif k == "op" and v == "!":
                        span = st[i + 1:operand_end(st, i + 1)]
                    elif k == "op" and v == "[" and i:
                        span = st[i + 1:group(st, i) - 1]
                        if kinds(span):
                            sinks.append(f"{fname}: a native value used as an index: {show(span)}")
                        span = None
                    elif k == "op" and v == "&" and i + 1 < len(st) and st[i + 1][0] == "id" and taint.get(st[i + 1][1]) \
                            and ("f" in taint[st[i + 1][1]] or i == 0 or   # `&` on a float can only be its address
                                 (st[i - 1][0] == "op" and st[i - 1][1] not in (")", "]"))):
                        sinks.append(f"{fname}: the address of a native value taken: & {st[i + 1][1]}")
                    elif k == "op" and v == "(" and i + 2 < len(st):
                        e = group(st, i)
                        inner = st[i + 1:e - 1]
                        if inner and all(t[0] == "id" and (t[1] in _INT_TYPES or t[1] in _QUALS) for t in inner) and \
                                e < len(st) and "f" in kinds(st[e:operand_end(st, e)]):
                            sinks.append(f"{fname}: a native float converted to an integer: {show(st[i:operand_end(st, e)])}")
                    elif k == "op" and v.endswith("=") and v not in _CMP_OPS:
                        rk = kinds(st[i + 1:])
                        if not rk:
                            continue
                        if i >= 2 and st[i - 2][1] in (".", "->"):
                            sinks.append(f"{fname}: a native value stored into a struct member: {show(st)}")
                        elif "f" in rk and i >= 1 and st[i - 1][0] == "id" and st[i - 1][1] in dt and \
                                any(b == "i" and d == 0 for b, d in dt[st[i - 1][1]]):
                            sinks.append(f"{fname}: a native float converted to an integer: {show(st)}")
                    if span is not None and kinds(span):
                        sinks.append(f"{fname}: a test on a native value: {show(span)}")
                # declarations converting an 'f' init to an integer
                s = st[2:] if st[:2] == [("id", "for"), ("op", "(")] else st
                tids = []
                for t in s:
                    if t[0] == "id" and (_typeish(t[1]) or t[1] in _QUALS):
                        tids.append(t[1])
                    else:
                        break
                if tids and any(t in _INT_TYPES for t in tids) and not any(t in _FLOAT_TYPES for t in tids):
                    for dcl in _csplit(s[len(tids):]):
                        if ("op", "=") in dcl and ("op", "*") not in dcl[:dcl.index(("op", "="))] and \
                                "f" in kinds(dcl[dcl.index(("op", "=")) + 1:]):
                            sinks.append(f"{fname}: a native float converted to an integer: {show(st)}")
                found += sinks
        if not param_changed[0]:
            break
    return list(dict.fromkeys(found))


def _symbolic_tested_reads(src: str) -> str:
    """A kernel the C-level checks above would refuse (a test or another native use of an input float it reads
    natively: a zero-skip's `if (a_val != 0.0f)`): every scalar read through a float pointer in its entry function
    (`p[i]`, `*p`, `*p++`) becomes a symbolic accessor (salt_scalar: the element's symbolic value, found by address),
    and the locals they initialize SymbolicScalars (_rebind_tainted_locals). A test on one is then decided by the
    assumptions or forks (verify's path loop), its arithmetic is symbolic, and an intrinsic takes it through the
    symbolic-scalar overloads; a use the symbolic type has no meaning for (fabsf, an int conversion, a native store)
    no longer compiles, a refusal. The result is used only if both checks then pass; otherwise, and for every
    kernel they accept as it is, the source is returned unchanged."""
    if not (_c_float_tests(src) or _native_scalar_uses(_strip_kernel_redeclarations(src))):
        return src
    head, body = _split_body(src)
    if not body:
        return src
    ptr = r'(?:const\s+)?(?:float|float32_t)\s*\*\s*(?:(?:const|restrict|__restrict|__restrict__|XNN_RESTRICT)\s+)*(\w+)'
    names = sorted(set(re.findall(r'(?<![\w*])' + ptr + r'(?!\s*\[)', head[head.rfind("test_"):] + body)))
    if not names:
        return src
    for n in names:
        e = re.escape(n)
        body = re.sub(rf'(?<![&\w.>])\b{e}\s*\[\s*([^\[\]]+?)\s*\](?!\s*(?:=[^=]|\+=|-=|\*=|/=|\[))',
                      rf'salt_scalar({n} + (\1))', body)                 # p[i], not &p[i], a store, or p[i][j]
        body = re.sub(rf'(?<![\w)\]])\*\s*{e}\s*\+\+', f'salt_scalar_postinc({n})', body)   # *p++
        body = re.sub(rf'(?<![\w)\]])\*\s*{e}\b(?!\s*(?:\+\+|=[^=]|\[))', f'salt_scalar({n})', body)  # *p
    body = _rebind_tainted_locals(body)
    out = head + body
    if _c_float_tests(out) or _native_scalar_uses(_strip_kernel_redeclarations(out)):
        return src
    return out


def _kernel_source(path) -> str:
    """A kernel's source as the harness compiles it (_symbolic_tested_reads: unchanged unless the C-level checks
    would refuse it as it is)."""
    return _symbolic_tested_reads(open(path).read())


def _tests_clamp_in_c(body: str) -> bool:
    """The kernel tests a clamp bound in C (`if (vmax != vmax)`, directly or through a local read from
    `params->scalar.min/max`), rather than only handing the bounds to intrinsics."""
    names = re.findall(r"\b(\w+)\s*=\s*\w+\s*->\s*scalar\.(?:min|max)\b", body)
    bound = r"\w+\s*->\s*scalar\.(?:min|max)\b" + "".join(rf"|\b{re.escape(n)}\b" for n in names)
    return any(re.search(bound, c) for c in re.findall(r"\b(?:if|while)\b\s*([^{;]*)", body))


def _refuse_c_float_tests(neon_src: str, rvv_src: str):
    for side, src in (("reference", neon_src), ("candidate", rvv_src)):
        if hits := _c_float_tests(src):
            raise NotImplementedError(f"{side}: C-level test on a float value the verifier cannot track "
                                      f"symbolically: {hits[0]}")
        if hits := _native_scalar_uses(_strip_kernel_redeclarations(src)):
            raise NotImplementedError(f"{side}: native use of a value the verifier cannot track symbolically: "
                                      f"{hits[0]}")


_FN_DEF = re.compile(r'^(?P<pre>[ \t]*(?:static\s+)?(?:inline\s+)?(?:const\s+)?\w+[\s\*]+(?P<name>\w+)\s*)'
                     r'\((?P<params>[^()]*)\)(?P<post>\s*\{)', re.M)


def _template_float_params(body: str, entry: str):
    """A helper the candidate defines with a scalar float parameter (`fmax_vf(v, float b, vl)`) becomes a
    template over that parameter's type, so a symbolic value passed to it stays symbolic inside: a C-level
    test on it is then decided or refused like any symbolic test. A plain float still deduces float.
    Returns the body and the templated parameter names (they hold a symbolic value in the helper)."""
    names = []
    def fix(m):
        if m.group("name") in (entry, "if", "for", "while", "switch", "return", "sizeof"):
            return m.group(0)
        ps, tps = m.group("params").split(","), []
        for i, prm in enumerate(ps):
            if mm := re.fullmatch(r'\s*(const\s+)?float\s+(\w+)\s*', prm):
                t = f"SaltF{len(tps)}"
                tps.append(t)
                names.append(mm.group(2))
                ps[i] = f" {mm.group(1) or ''}{t} {mm.group(2)}"
        if not tps:
            return m.group(0)
        head = "template<" + ", ".join(f"typename {t}" for t in tps) + ">\n"
        return head + m.group("pre") + "(" + ",".join(ps).strip() + ")" + m.group("post")
    return _FN_DEF.sub(fix, body), names


def _rewrite_symbolic_params(src: str, st: str, pname: str, leaves) -> str:
    """Rewrite `params->field` reads to a symbolic accessor so the solver proves
    over ALL valid param values. `(T) params->f` -> `salt_param<ft>(...).cast<T>()`."""
    head, body = _split_body(src)
    if not body:
        return src
    for path, ctype, _ in sorted(leaves, key=lambda l: -len(l[0])):  # longest first
        if path == "_":
            continue
        acc = f"salt_param<{ctype}>({pname}, offsetof({st}, {path}))"
        esc = re.escape(f"{pname}->{path}")
        body = re.sub(rf'\(\s*(\w+)\s*\)\s*{esc}\b', rf'{acc}.cast<\1>()', body)
        body = re.sub(rf'{esc}\b', acc, body)
    # XNNPACK idiom `(T)(-(U)params->f)`: the inner `(U)params->f` became a
    # `.cast<U>()` accessor above, but the outer C-style `(T)(...)` cast can't
    # apply to a SymbolicScalar. Rewrite `(T) <sym>` → symbolic `<sym>.cast<T>()`.
    _ACC = r"salt_param<\w+>\(\w+,\s*offsetof\([^)]+\)\)(?:\.cast<\w+>\(\))?"
    _CT = r"u?int(?:8|16|32|64)_t|size_t|float"
    _prev = None
    while _prev != body:
        _prev = body
        body = re.sub(rf'\(\s*({_CT})\s*\)\s*\(\s*-\s*({_ACC})\s*\)', r'(-\2).cast<\1>()', body)
        body = re.sub(rf'\(\s*({_CT})\s*\)\s*-\s*({_ACC})',           r'(-\2).cast<\1>()', body)
        body = re.sub(rf'\(\s*({_CT})\s*\)\s*\(\s*({_ACC})\s*\)',     r'\2.cast<\1>()', body)
        body = re.sub(rf'\(\s*({_CT})\s*\)\s*({_ACC})',               r'\2.cast<\1>()', body)
    body = _rebind_tainted_locals(body)
    # A local now holding a SymbolicScalar can't take a C-style cast either:
    # `(int16_t)a_zero_point`, `(size_t)(-shift)` → `.cast<T>()` (C's conversion).
    for n in sorted(set(re.findall(r'\bauto\s+(\w+)\s*=', body)), key=len, reverse=True):
        _prev = None
        while _prev != body:
            _prev = body
            body = re.sub(rf'\(\s*({_CT})\s*\)\s*\(\s*-\s*{n}\s*\)', rf'(-{n}).cast<\1>()', body)
            body = re.sub(rf'\(\s*({_CT})\s*\)\s*-\s*{n}\b', rf'(-{n}).cast<\1>()', body)
            body = re.sub(rf'\(\s*({_CT})\s*\)\s*\(\s*{n}\s*\)', rf'{n}.cast<\1>()', body)
            # chained C casts: `(uint32_t)(int32_t)n` became `(uint32_t)n.cast<int32_t>()` above
            body = re.sub(rf'\(\s*({_CT})\s*\)\s*{n}((?:\.cast<\w+>\(\))+)', rf'{n}\2.cast<\1>()', body)
            body = re.sub(rf'\(\s*({_CT})\s*\)\s*{n}\b(?!\s*(?:[\[(.]|->))', rf'{n}.cast<\1>()', body)
    # `(T)(expr)` where expr holds a symbolic value: C's conversion, `salt_as<T>(expr)`.
    syms = set(re.findall(r'\bauto\s+(\w+)\s*=', body))
    def _cast_groups(text):
        out, i = [], 0
        cast = re.compile(rf'\(\s*({_CT})\s*\)\s*(-\s*)?\(')     # (T)(expr) and (T)-(expr)
        while (m := cast.search(text, i)):
            depth, j = 1, m.end()
            while j < len(text) and depth:
                depth += {"(": 1, ")": -1}.get(text[j], 0)
                j += 1
            inner = text[m.end():j - 1]
            symbolic = "salt_" in inner or ".cast<" in inner or any(re.search(rf"\b{re.escape(n)}\b", inner) for n in syms)
            out.append(text[i:m.start()])
            neg = "-" if m.group(2) else ""
            out.append(f"salt_as<{m.group(1)}>({neg}({_cast_groups(inner)}))" if symbolic else text[m.start():m.end()] + _cast_groups(inner) + ")")
            i = j
        out.append(text[i:])
        return "".join(out)
    body = _cast_groups(body)
    if f"{pname}->" in body:  # residual param read the rewrite missed → fail loudly
        raise NotImplementedError(f"unrewritten param read of {pname} remains "
                                  "(unhandled access pattern)")
    return head + body


def _field_leaf_map(leaves):
    """field-name → (dotted_path, ctype), dropping the dummy `_` member."""
    return {p.split(".")[-1]: (p, t) for p, t, _ in leaves if p != "_"}


def _emit_param_constraints(st: str, leaves) -> list:
    """Validity asserts (e.g. output_min ≤ output_max, shift∈[0,31], multiplier>0)
    from the curated PARAM_CONSTRAINTS, against the symbolic params buffer."""
    f2l = _field_leaf_map(leaves)
    out = []
    for op, a, b in PARAM_CONSTRAINTS.get(st, []):
        if a not in f2l:
            continue
        pa, ta = f2l[a]
        bits_a = _BYTES[ta] * 8
        meth = "bv_" + op[3:].lower()          # BV_SLE -> bv_sle
        at = f"_b_params.loadScalar(offsetof({st}, {pa}), {bits_a})"
        if isinstance(b, int):
            bt = f"ctx.bv_val({bits_a}, {b})"
        elif b in f2l:
            pb, tb = f2l[b]
            bt = f"_b_params.loadScalar(offsetof({st}, {pb}), {_BYTES[tb] * 8})"
        else:
            continue
        out.append(f"    ctx.assert_(ctx.{meth}({at}, {bt}));")
    return out


def _emit_param_ranges(st: str, leaves, ranges: dict) -> list:
    """Per-field [lo, hi] bounds (from the kernel profile) — restrict symbolic
    params to their valid domain (e.g. a narrowed multiplier that must fit int16),
    which also helps the solver. Signed vs unsigned chosen by the field type."""
    f2l = _field_leaf_map(leaves)
    out = []
    for fname, bounds in (ranges or {}).items():
        if fname not in f2l or not bounds:
            continue
        lo, hi = bounds
        path, ct = f2l[fname]
        if ct == "float":                     # inclusive; FP comparisons also exclude NaN
            v = f"load_as_fp32(ctx.tm, _b_params.loadScalar(offsetof({st}, {path}), 32))"
            out.append(f"    ctx.assert_(ctx.land_(ctx.fp32_geq({v}, mk_fp32_from_float(ctx.tm, {float(lo).hex()}f)), "
                       f"ctx.fp32_leq({v}, mk_fp32_from_float(ctx.tm, {float(hi).hex()}f))));")
            continue
        bits = _BYTES[ct] * 8
        ge, le = ("bv_sge", "bv_sle") if not ct.startswith("u") else ("bv_uge", "bv_ule")
        v = f"_b_params.loadScalar(offsetof({st}, {path}), {bits})"
        out.append(f"    ctx.assert_(ctx.{ge}({v}, mk_bv_val(ctx.tm, {bits}, {lo})));")
        out.append(f"    ctx.assert_(ctx.{le}({v}, mk_bv_val(ctx.tm, {bits}, {hi})));")
    return out


def _emit_input_range(buf_var: str, n_expr: str, lo, hi) -> list:
    """Constrain each F32 element of a symbolic input buffer to finite [lo, hi]
    (excludes NaN/inf — FP comparisons are false for NaN). Optional; weakens the
    claim to finite inputs, so it must be recorded in the verdict."""
    lo_f, hi_f = float(lo), float(hi)   # ensure valid float literals (e.g. "-1" → -1.0f)
    return [
        f"    {{  // input-range [{lo_f}, {hi_f}] on {buf_var}",
        f"        Term _lo = mk_fp32_from_float(ctx.tm, {lo_f}f), _hi = mk_fp32_from_float(ctx.tm, {hi_f}f);",
        f"        for (size_t _i = 0; _i < ({n_expr}); _i++) {{",
        f"            Term _fp = load_as_fp32(ctx.tm, {buf_var}.loadScalar(_i * 4, 32));",
        f"            ctx.assert_(ctx.land_(ctx.fp32_geq(_fp, _lo), ctx.fp32_leq(_fp, _hi)));",
        f"        }}",
        f"    }}",
    ]


def _seed_side_channel(buf: str, off: str) -> str:
    """C++ writing a nonzero canary into the native memory of a side-channel element (the model never reads it)."""
    return (f"{{ uint32_t _seed = 0x13572468u; std::memcpy(reinterpret_cast<uint8_t*>({buf}.baseAddr()) + ({off}), "
            f"&_seed, 4); }} ")


def _native_guarded(call: str, side: str, ctx_fmt: str, ctx_args: str, allow: str = "", post: str = "",
                    pre: str = "") -> str:
    """A kernel call inside the native-access guard (core/native_guard.hpp): a native store into a registered
    buffer, native float arithmetic on a symbolic value, or an intrinsic without a model is refused (status
    NATIVE_ACCESS), since the symbolic model cannot see it. `allow`: the NativeAllow entries a side channel may
    write natively (rsum's `*out += sum`); `post`: C++ run after the call with the snapshot `_ns` in scope; `pre`:
    C++ run before the snapshot (seeding a side-channel element, so `*out = sum` and `*out += sum` differ)."""
    tag = "reference" if side == "neon" else "candidate"
    exempt = ", ".join(f'"{n}"' for n in dict.fromkeys(re.findall(r'\{\s*"(\w+)"', allow)))
    return (f'        {{ {pre}snprintf(salt_fault_line, sizeof salt_fault_line, '
            f'"{{\\"status\\":\\"NATIVE_ACCESS\\",{ctx_fmt},\\"side\\":\\"{tag}\\"}}\\n", {ctx_args});\n'
            f'        NativeSnapshot _ns = native_guard_begin(ctx);\n'
            f'        try {{\n'
            f'        {{ NativeProtect _np(ctx, {{{exempt}}});\n{call}\n        }}\n'
            f'        native_guard_end(ctx, _ns, "{tag}", NativeAllow{{{allow}}});\n{post}'
            f'        }} catch (const NativeAccess& _na) {{\n'
            f'            fprintf(stderr, "NATIVE_ACCESS: %s\\n", _na.what());\n'
            f'            printf("{{\\"status\\":\\"NATIVE_ACCESS\\",{ctx_fmt},\\"side\\":\\"{tag}\\"}}\\n", {ctx_args});\n'
            f'            return {3 if side == "neon" else 1};\n'
            f'        }} }}')


def _tracked_calls(ref_call: str, cand_call: str, after_ref: str, pairs: dict, inputs, own,
                   ctx_fmt: str, ctx_args: str, layouts: dict = None, ref_allow: str = "", ref_post: str = "",
                   cand_allow: str = "", cand_post: str = "", ref_pre: str = "", cand_pre: str = "") -> str:
    """C++ for the two kernel calls with memory footprints (core/memcheck.hpp), each inside the native-access
    guard (_native_guarded; `*_allow`/`*_post` are its side-channel allowance and check).

    The reference running past a buffer means the fixture is too small (error 3);
    the candidate doing so, or touching bytes outside the reference's footprint,
    is a memory violation. SALT_MEMCHECK=strict fails on footprint errors (see
    memcheck.hpp), `report` (default) prints them as MEMORY_WARNING and keeps
    verifying, `off` skips the comparison; over-reads inside a buffer are always
    warnings, and a candidate overrun past capacity always fails, since symbolic
    execution cannot continue past it. `ctx_fmt`/`ctx_args` are the
    printf fields identifying the query (shape/batch, config). `layouts` maps a
    candidate output to the C++ byte extent of its caller-owned packed layout."""
    q = lambda s: '"' + s + '"'
    pairs_cpp = "{" + ", ".join(f"{{{q(c)}, {q(r)}}}" for c, r in pairs.items()) + "}"
    inputs_cpp = "{" + ", ".join(q(b) for b in sorted(set(inputs))) + "}"
    own_cpp = "{" + ", ".join(q(b) for b in sorted(set(own))) + "}"
    layouts_cpp = ("std::map<std::string, long long>{" +
                   ", ".join(f"{{{q(b)}, (long long)({e})}}" for b, e in sorted((layouts or {}).items())) + "}")
    ref_call = _native_guarded(ref_call, "neon", ctx_fmt, ctx_args, ref_allow, ref_post, ref_pre)
    cand_call = _native_guarded(cand_call, "rvv", ctx_fmt, ctx_args, cand_allow, cand_post, cand_pre)
    return (
        f'    ctx.mark_assumptions();   // what is asserted so far: the inputs\' and parameters\' domains\n'
        f'#ifdef SALT_CVC5_EXTENSIONS\n'
        f'    const char* _mcv = std::getenv("SALT_MEMCHECK");\n'
        f'    const std::string _mcm = _mcv ? _mcv : "report";\n'
        f'    footprint_begin(ctx);\n'
        f'    try {{\n{ref_call}\n    }} catch (const BufferOOB& _e) {{\n'
        f'        footprint_end(ctx);\n'
        f'        printf("{{\\"status\\":\\"FIXTURE_ERROR\\",{ctx_fmt},\\"error\\":\\"reference %s past the capacity of %s at %lld\\"}}\\n", {ctx_args}, _e.op.c_str(), _e.buffer.c_str(), _e.offset);\n'
        f'        return 3;\n'
        f'    }}\n'
        f'    Footprints _fp_ref = footprint_end(ctx);\n'
        f'{after_ref}'
        f'    footprint_begin(ctx);\n'
        f'    try {{\n{cand_call}\n    }} catch (const BufferOOB& _e) {{\n'
        f'        footprint_end(ctx);\n'
        f'        fprintf(stderr, "%s\\n", _e.what());\n'
        f'        printf("{{\\"status\\":\\"MEMORY\\",{ctx_fmt},\\"violations\\":[{{\\"buffer\\":\\"%s\\",\\"access\\":\\"%s past capacity\\",\\"lo\\":%lld,\\"hi\\":%lld}}]}}\\n", {ctx_args}, _e.buffer.c_str(), _e.op.c_str(), _e.offset, _e.offset);\n'
        f'        return 1;\n'
        f'    }}\n'
        f'    Footprints _fp_cand = footprint_end(ctx);\n'
        f'    if (_mcm != "off") {{\n'
        f'        MemoryReport _mem = memory_violations(_fp_ref, _fp_cand, {pairs_cpp}, {inputs_cpp}, {own_cpp}, {layouts_cpp});\n'
        f'        if (_mem.warnings != "[]")\n'
        f'            printf("{{\\"status\\":\\"MEMORY_WARNING\\",{ctx_fmt},\\"violations\\":%s}}\\n", {ctx_args}, _mem.warnings.c_str());\n'
        f'        if (_mem.errors != "[]") {{\n'
        f'            printf("{{\\"status\\":\\"%s\\",{ctx_fmt},\\"violations\\":%s}}\\n", _mcm == "strict" ? "MEMORY" : "MEMORY_WARNING", {ctx_args}, _mem.errors.c_str());\n'
        f'            if (_mcm == "strict") return 1;\n'
        f'        }}\n'
        f'    }}\n'
        f'#else\n{ref_call}\n{after_ref}{cand_call}\n#endif\n')


def _nonvacuity_block(diff_cpp: str, ctx_fmt: str, ctx_args: str) -> str:
    """C++ checking that a verdict can mean something (SALT_NONVACUITY=1).

    Before the equivalence query: the assumptions alone (input ranges, packing
    constraints, parameter profiles) must be satisfiable, else every query is
    trivially UNSAT; and some input must change the reference's output, else the
    comparison checks nothing (a clamp to [0,0] makes every output 0). `diff_cpp`
    builds `Term _diff`: the reference output differs from its value in the
    current model. Failures return 1; NONVACUOUS is informational."""
    return (
        "#ifdef SALT_CVC5_EXTENSIONS\n"
        '    if (const char* _nvv = std::getenv("SALT_NONVACUITY"); _nvv && std::string(_nvv) == "1") {\n'
        "        if (timeout_s > 0) ctx.set_query_timeout_ms((uint64_t)(timeout_s * 250.0));\n"
        "        ctx.solver->push();\n"
        "        CheckResult _sa = ctx.check_cvc5();\n"
        '        const char* _nvs = "NONVACUITY_UNKNOWN";\n'
        '        if (_sa == CheckResult::Unsat) _nvs = "VACUOUS_ASSUMPTIONS";\n'
        "        else if (_sa == CheckResult::Sat) {\n"
        "            Term _diff = ctx.tm.mkFalse();\n"
        f"{diff_cpp}"
        "            ctx.solver->assertFormula(_diff);\n"
        "            CheckResult _sb = ctx.check_cvc5();\n"
        '            _nvs = _sb == CheckResult::Unsat ? "VACUOUS_CONSTANT_OUTPUT"\n'
        '                 : _sb == CheckResult::Sat ? "NONVACUOUS" : "NONVACUITY_UNKNOWN";\n'
        "        }\n"
        "        ctx.solver->pop();\n"
        f'        printf("{{\\"status\\":\\"%s\\",{ctx_fmt}}}\\n", _nvs, {ctx_args});\n'
        "        if (_nvs[0] == 'V') return 1;\n"
        "    }\n"
        "#endif\n")


def _output_is_read(src: str, name: str) -> bool:
    """True if an output buffer is *read* (loaded) anywhere — i.e. it's an in-out
    accumulator / in-place buffer, where the prior contents are part of the input.
    Detected by the output pointer appearing as the argument to a load intrinsic."""
    return bool(re.search(
        rf"(vld\w*|__riscv_vle\w+|vget\w*q?_\w+)\s*\(\s*&?\s*{re.escape(name)}\b", src))


# Multi-dim fixture: swept dims + chosen strides.
class FixtureSpec:
    def __init__(self, dims, strides, extents, out_buffers, out_extent, elem, kind,
                 out_region=None, weights=None, params=None, indirect=None,
                 out_extents=None, dual_weights=None, out_elem=None, out_kind=None,
                 probes=None, odd_tail_dim=None, grow_dim=None, grow_cross=None,
                 out_elems=None, shared_out=None, scalar_movs=False,
                 extra_params=None, raw_decls=None, index_reads=None, pre_defines=None, rvv_alias=None,
                 scalar_reads=None, config_tag=None, preconditions=None, nullable=None,
                 pad=None):
        self.dims = dims              # dim_name -> [values]
        self.strides = strides        # stride_name -> C++ expr (over dims)
        self.extents = extents        # buffer_name -> byte-count expr
        self.out_buffers = out_buffers
        self.out_extent = out_extent  # contiguous output element count (if no out_region)
        self.elem = elem
        self.kind = kind
        self.out_elem = out_elem or elem   # output element type (rdsum: int8 in, int32 out)
        self.out_kind = out_kind or kind   # output ElementKind (matches out_elem)
        self.out_region = out_region  # {rows,cols,row_stride,col_stride} → strided compare
        self.weights = weights or {}  # name -> (byte-extent expr, uniform fill literal) — concrete
        self.params = params          # (struct_name, {field: value}) — concrete struct, or None
        self.indirect = indirect or {}  # name -> {count, row} — T** pointer table over a symbolic buffer
        self.out_extents = out_extents   # {name: elem-count expr} — multi-output contiguous compare (each output ANDed)
        self.out_elems = out_elems or {}  # {output_name: C type} — per-output element type for MIXED-type multi-output (argmaxpool: output=float, index=uint32_t); defaults to out_elem
        self.dual_weights = dual_weights or {}  # name -> {count, neon_tile, rvv_tile} — per-dialect weight packing
        self.probes = probes or []    # extra explicit shapes [{dim: value, ...}] appended after the dims cross product
        self.odd_tail_dim = odd_tail_dim  # dim whose odd value routes the last output lane through a scalar `*out += vgetq_lane` store (rsum/rdsum tail) — captured via scalar side-channel
        self.grow_dim = grow_dim          # dim to grow continuously (unbounded sweep over argv[1..2]/budget) instead of cross-product; the rest of `dims` is the per-row representative inner set
        self.grow_cross = grow_cross      # (d0, d1): grow BOTH dims cross-product-wise as an expanding N×N square frontier until budget (dims must be empty)
        self.shared_out = shared_out or []  # outputs with unwritten holes (packw extra_bytes gaps) → force shared init even if never read
        self.scalar_movs = scalar_movs    # rewrite `*dst = *src++;` to salt_deref/salt_scalar accessors (packw remainder)
        self.extra_params = extra_params or {}  # name -> (struct, {field: value}) — 2nd+ concrete param structs (qd8 quantization_params)
        self.raw_decls = raw_decls or []  # raw C++ lines appended after buffer decls (extra constraints; cvc5 API)
        self.rvv_alias = rvv_alias or {}  # arg name -> (C++ expr, buffer name): the candidate's own copy (built in raw_decls)
        self.index_reads = index_reads or []  # local ptr names whose `p[i]` reads become salt_scalar (int scalars, no sentinel)
        self.pre_defines = pre_defines or []  # #define lines before kernel bodies (e.g. force XNN_ARCH_ARM64=0)
        self.scalar_reads = scalar_reads or []  # local input pointers dereferenced as *p / *p++
        self.config_tag = config_tag
        self.preconditions = preconditions or []   # [(C expr over dims/strides, "reference"|"candidate")]
        self.nullable = nullable or {}    # input name -> variant dim: pass NULL when the dim is nonzero
        self.pad = pad                    # C++ bool over element index `e` (and dims): a shared_out layout pad, don't-care


def _without_function(src: str, name: str) -> str:
    """`src` without the definition of function `name` (a helper the harness never calls, such as a benchmark's
    pack_weights layout converter), braces matched outside comments and strings."""
    m = re.search(rf"^[^\n;{{}}]*\b{re.escape(name)}\s*\([^;{{]*\)\s*\{{", src, re.M)
    if not m:
        return src
    depth, i = 0, m.end() - 1
    scan = re.sub(r"/\*.*?\*/|//[^\n]*|\"(?:\\.|[^\"\\])*\"", lambda x: " " * len(x[0]), src, flags=re.S)
    while i < len(scan):
        depth += scan[i] == "{"
        depth -= scan[i] == "}"
        i += 1
        if depth == 0:
            return src[:m.start()] + src[i:]
    return src


def _kernel_only(src: str, entry: str = "test_rvv") -> str:
    """`src` without the top-level functions that `entry` does not reach (directly or through the helpers it
    calls): a benchmark's layout converter or another unused helper must not steer the analysis of the kernel's
    own code (its weight stepping, its vsetvl LMULs). `src` unchanged when `entry` is not defined in it."""
    scan = re.sub(r"/\*.*?\*/|//[^\n]*|\"(?:\\.|[^\"\\])*\"", lambda x: re.sub(r"[^\n]", " ", x[0]), src, flags=re.S)
    scan = re.sub(r"^[ \t]*#[^\n]*", lambda x: " " * len(x[0]), scan, flags=re.M)
    head = re.compile(r"\b(\w+)\s*\([^;{}()]*(?:\([^;{}()]*\)[^;{}()]*)*\)\s*(?:\w+\s*)*\{")
    defs, depth, i = {}, 0, 0
    while i < len(scan):
        if scan[i] == "{":
            depth += 1
        elif scan[i] == "}":
            depth -= 1
        elif depth == 0:
            m = head.match(scan, i)
            if m and m.group(1) not in ("if", "for", "while", "switch", "return", "sizeof"):
                j, d = m.end() - 1, 0
                while j < len(scan):
                    d += scan[j] == "{"
                    d -= scan[j] == "}"
                    j += 1
                    if d == 0:
                        break
                defs[m.group(1)] = (scan.rfind("\n", 0, i) + 1, j)   # from the start of the definition's line
                i = j
                continue
        i += 1
    if entry not in defs:
        return src
    keep, todo = set(), [entry]
    while todo:
        f = todo.pop()
        if f not in keep:
            keep.add(f)
            s, e = defs[f]
            todo += [g for g in defs if g not in keep and re.search(rf"\b{re.escape(g)}\s*\(", scan[s:e])]
    out, pos = [], 0
    for name, (s, e) in sorted(defs.items(), key=lambda x: x[1][0]):
        if name not in keep:
            out.append(src[pos:s])
            pos = e
    return "".join(out) + src[pos:]


def _rvv_chunked_weights(rvv_src: str, ptr: str = "w"):
    """Chunk width of an RVV kernel's packed-weight reads, or None.

    NEON kernels read packed weights in fixed tiles (gemm: [bias(8)][k0(8)]...).
    An RVV kernel that steps the weight pointer by `vl` (`w += vl`, conv's
    `w += 28 * vl`) reads vl-wide chunks of the same planes instead, so the
    fixture must lay the same logical weights out per side. Returns VLMAX for
    the kernel's e32 LMUL as a C++ expression in `vlen`."""
    # the kernel's own code: not a benchmark's layout converter (pack_weights) or another helper it never calls
    rvv_src = _kernel_only(_without_function(rvv_src, "pack_weights"))
    # `w += vl`, `w += 28 * vl`; or a weight pointer formed as an offset in vl units (`w_row = w_strip + (1 + i * kc) *
    # vl`, the VLEN-128 igemm, whose strips are vl wide).
    if not (re.search(rf"\b{ptr}\w*\s*\+=\s*(\d+\s*\*\s*)?vl\b", rvv_src) or
            re.search(rf"\b{ptr}\w*\s*=\s*{ptr}\w*\s*\+[^;]*\bvl\b", rvv_src)):
        return None
    lmuls = set(re.findall(r"__riscv_vsetvl_e32m(\d)\s*\(", rvv_src))
    if len(lmuls) != 1:
        raise NotImplementedError(f"vl-chunked weights with e32 LMUL {sorted(lmuls) or 'unknown'}")
    return f"(vlen / 32) * {lmuls.pop()}"


def _rvv_padded_tile(rvv_src: str, ptr: str = "w"):
    """Tile width of an RVV GEMM that steps its weights by a full `nr = __riscv_vsetvlmax_e32mL()` tile
    (`w += nr`, `w += 4 * nr`; upstream's 4v layout, tail tile zero-padded), as a C++ expression in `vlen`,
    or None."""
    rvv_src = _kernel_only(_without_function(rvv_src, "pack_weights"))
    m = re.search(r"\bnr\s*=\s*__riscv_vsetvlmax_e32m(\d)\s*\(\s*\)", rvv_src)
    if not m or not re.search(rf"\b{ptr}\w*\s*\+=\s*(\d+\s*\*\s*)?nr\b", rvv_src):
        return None
    return f"(vlen / 32) * {m[1]}"


def _transpose_fixture(elem: str) -> "FixtureSpec":
    # block_height×block_width → block_width×block_height, any element width.
    # Padded strides (+8B/row) make the output strided. grow_cross expands an N×N
    # square frontier until the budget.
    return FixtureSpec(
        dims={},
        grow_cross=("block_width", "block_height"),
        strides={"input_stride": f"block_width * sizeof({elem}) + 8",
                 "output_stride": f"block_height * sizeof({elem}) + 8"},
        extents={"input": "block_height * input_stride + 64",     # bytes
                 "output": "block_width * output_stride + 64"},
        out_buffers=["output"], out_extent=None,
        out_region={"rows": "block_width", "cols": "block_height",
                    "row_stride": "output_stride", "col_stride": f"sizeof({elem})"},
        elem=elem, kind=_kind(elem))


def _qd8_qc4w_gemm_fixture(rvv_src: str = "") -> "FixtureSpec":
    # qd8-f32-qc4w 1-row GEMM: int8 activations × packed 4-bit weights → f32 out.
    # ONE shared symbolic `w`: both kernels read 16-column tiles (the RVV kernels
    # set vl = min(nc, 16) explicitly), at any VLEN.
    # Tile layout: ksum 16×i32 | (kc/2) nibble planes 16×u8 | scale 16×f32 | bias 16×f32.
    # raw_decls assert ksum ≡ 0 (mod 16) — the packer invariant. cvc5-only.
    tile = "(16 * 4 + ((kc + 1) / 2) * 16 + 16 * 4 + 16 * 4)"
    return FixtureSpec(
        dims={"mr": [1], "nc": [1, 2, 4, 5, 8, 15, 16], "kc": [2, 4, 6, 8, 10]},
        strides={"a_stride": "kc", "cm_stride": "256 * sizeof(float)",   # unused at mr <= 1
                 "cn_stride": "16 * sizeof(float)"},
        extents={"a": "kc + 72",                       # +8 OOB pad (NEON vld1_s8 tail) +64
                 "w": f"((nc + 15) / 16) * {tile} + 64",
                 "c": "nc * sizeof(float) + 64"},
        out_buffers=["c"], out_extent=None,
        out_region={"rows": "mr", "cols": "nc",
                    "row_stride": "cm_stride", "col_stride": "sizeof(float)"},
        params=("xnn_f32_qc4w_minmax_params",
                {"scalar.min": "-1e30f", "scalar.max": "1e30f", "scalar.kernel_zero_point": "8"}),
        extra_params={"quantization_params": ("xnn_qd8_quantization_params",
                                              {"zero_point": "5", "inv_scale": "0.5f"})},
        index_reads=["a0"],
        raw_decls=[
            "    { auto& _tm = ctx.tm; Op _lo4 = _tm.mkOp(Kind::BITVECTOR_EXTRACT, {3, 0});",
            f"      size_t _tilesz = {tile};",
            "      size_t _fpoff = 16 * 4 + ((kc + 1) / 2) * 16;   // scale plane offset in tile",
            "      for (size_t _t = 0; _t < (nc + 15) / 16; _t++)",
            "        for (size_t _i = 0; _i < 16; _i++) {",
            "          Term _ks = _b_w.loadScalar(_t * _tilesz + _i * 4, 32);",
            "          ctx.solver->assertFormula(_tm.mkTerm(Kind::EQUAL, "
            "{_tm.mkTerm(_lo4, {_ks}), _tm.mkBitVector(4, 0)}));",
            "          // |ksum| <= 128*kc: the packer's ksum is 16 times a sum of kc signed nibbles",
            "          ctx.assert_(ctx.bv_sge(_ks, mk_bv_val(ctx.tm, 32, -128 * (int64_t)kc)));",
            "          ctx.assert_(ctx.bv_sle(_ks, mk_bv_val(ctx.tm, 32, 128 * (int64_t)kc)));",
            "          // scale plane: a channelwise quantization scale, positive and finite (tensor.c);",
            "          // bias plane: the bias tensor's values, unconstrained",
            "          { Term _fp = load_as_fp32(ctx.tm, _b_w.loadScalar(_t * _tilesz + _fpoff + _i * 4, 32));",
            "            ctx.assert_(ctx.land_(ctx.fp32_geq(_fp, mk_fp32_from_float(ctx.tm, 0x1p-149f)),"
            " ctx.fp32_leq(_fp, mk_fp32_from_float(ctx.tm, 0x1.fffffep+127f)))); }",
            "        } }",
        ] + (_qd8_wide_tiles() if re.search(r"\bnr\s*=\s*__riscv_vsetvlmax_e32m4\s*\(\s*\)", rvv_src) else []),
        rvv_alias=({"w": ("w_rvv_arr.data()", "w_rvv")}
                   if re.search(r"\bnr\s*=\s*__riscv_vsetvlmax_e32m4\s*\(\s*\)", rvv_src) else None),
        elem="int8_t", kind="SINT", out_elem="float", out_kind="F32")


def _qd8_wide_tiles():
    """A candidate with upstream's 4v tiles (nr = vsetvlmax_e32m4) gets its own `w`: the same logical weights
    (the shared symbolic 16-column tiles, constraints included) copied column by column into nr-wide tiles
    [ksum nr*i32 | (kc+1)/2 nibble rows of nr bytes | scale nr*f32 | bias nr*f32], tail columns zero."""
    return [
        "    const size_t _T = (vlen / 32) * 4, _NP = (kc + 1) / 2;",
        "    const size_t _t16 = 16 * 4 + _NP * 16 + 16 * 4 + 16 * 4, _tT = _T * 4 + _NP * _T + _T * 4 + _T * 4;",
        "    std::vector<int8_t> w_rvv_arr(((nc + _T - 1) / _T) * _tT + 64, 0);",
        '    auto& _b_wr = ctx.registerBuffer("w_rvv", w_rvv_arr.size()); _b_wr.bind(w_rvv_arr.data());',
        "    for (size_t _col = 0; _col < nc; _col++) {",
        "      size_t _s = (_col / 16) * _t16, _j = _col % 16, _d = (_col / _T) * _tT, _k = _col % _T;",
        "      _b_wr.storeScalar(_d + _k * 4, _b_w.loadScalar(_s + _j * 4, 32), 32);                          // ksum",
        "      for (size_t _p = 0; _p < _NP; _p++)",
        "        _b_wr.storeScalar(_d + _T * 4 + _p * _T + _k, _b_w.loadScalar(_s + 64 + _p * 16 + _j, 8), 8);   // nibbles",
        "      _b_wr.storeScalar(_d + _T * 4 + _NP * _T + _k * 4, _b_w.loadScalar(_s + 64 + _NP * 16 + _j * 4, 32), 32);   // scale",
        "      _b_wr.storeScalar(_d + _T * 8 + _NP * _T + _k * 4, _b_w.loadScalar(_s + 128 + _NP * 16 + _j * 4, 32), 32);  // bias",
        "    }",
    ]


def _qc8w_fixture(*, dynamic: bool, indirect: bool = False) -> "FixtureSpec":
    """NR=8, KR=1 byte layout shared by these three one-row NEON/RVV pairs.

    Each tile is [8*i32 correction/bias][ks_count*kc*8*i8][8*f32 scale],
    followed by [8*f32 bias] for qd8. No qc4w ksum/nibble constraints apply.
    Integers have their full bit-vector domain; packed FP planes are finite
    (scale nonnegative). Params are three concrete profiles, not symbolic.
    """
    depth = "(ks_count * kc)" if indirect else "kc"
    tile = f"(8 * sizeof(int32_t) + {depth} * 8 + {2 if dynamic else 1} * 8 * sizeof(float))"
    output = "float" if dynamic else "int8_t"
    dims = {"param_case": [0, 1, 2], "mr": [1],
            "nc": [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17],
            "kc": [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17]}
    strides = {"cm_stride": f"nc * sizeof({output})",
               "cn_stride": f"8 * sizeof({output})"}
    extents = {"w": f"((nc + 7) / 8) * {tile} + 64",
               "c": f"nc * sizeof({output}) + 64"}
    if dynamic:
        params = ("xnn_f32_minmax_params", {
            "scalar.min": "param_case == 0 ? -1e30f : (param_case == 1 ? -1.0f : -32.0f)",
            "scalar.max": "param_case == 0 ? 1e30f : (param_case == 1 ? 1.0f : 64.0f)",
        })
        extra_params = {"quantization_params": ("xnn_qd8_quantization_params", {
            "zero_point": "param_case == 0 ? 5 : (param_case == 1 ? 0 : -7)",
            "inv_scale": "param_case == 0 ? 0.5f : (param_case == 1 ? 1.0f : 2.0f)",
        })}
    else:
        params = ("xnn_qs8_qc8w_conv_minmax_params", {
            "fp32_neon.magic_bias": "12582912.0f",
            "fp32_neon.magic_bias_less_output_zero_point":
                "0x4B400000 - (param_case == 0 ? 0 : (param_case == 1 ? 5 : -7))",
            "fp32_neon.output_min": "param_case == 0 ? -128 : -64",
            "fp32_neon.output_max": "param_case == 0 ? 127 : 63",
        })
        extra_params = {}
    raw = [
        "    {",
        f"      const size_t _tile = {tile};",
        f"      const size_t _fp_off = 8 * sizeof(int32_t) + {depth} * 8;",
        "      Term _flo = mk_fp32_from_float(ctx.tm, -0x1.fffffep127f);",
        "      Term _fhi = mk_fp32_from_float(ctx.tm, 0x1.fffffep127f);",
        "      Term _zero = mk_fp32_from_float(ctx.tm, 0.0f);",
        "      for (size_t _t = 0; _t < (nc + 7) / 8; ++_t)",
        f"        for (size_t _p = 0; _p < {2 if dynamic else 1}; ++_p)",
        "          for (size_t _i = 0; _i < 8; ++_i) {",
        "            Term _fp = load_as_fp32(ctx.tm, _b_w.loadScalar(",
        "                _t * _tile + _fp_off + (_p * 8 + _i) * sizeof(float), 32));",
        "            ctx.assert_(ctx.land_(ctx.fp32_geq(_fp, _p == 0 ? _zero : _flo),",
        "                                  ctx.fp32_leq(_fp, _fhi)));",
        "          }",
        "    }",
    ]
    indirect_spec = {}
    weights = {}
    if indirect:
        # ks is a byte count of pointers; a_offset is bytes (int8 elements).
        # zero_mode covers no sentinel, all sentinel, and mixed pointer tables.
        dims["nc"] = [1, 4, 7, 8, 9, 17]
        dims["kc"] = [1, 7, 8, 9]
        dims.update(ks_count=[1, 3], a_offset=[0, 3], zero_mode=[0, 1, 2])
        strides["ks"] = "ks_count * sizeof(void*)"
        indirect_spec = {"a": {"count": "ks_count", "row": "a_offset + kc + 8"}}
        # A distinct sentinel address and replacement buffer. Symbolic zero_data
        # is stronger than assuming a particular quantized padding value.
        weights["zero"] = ("16", "0")
        extents["zero_data"] = "kc + 8"
        raw += [
            "    for (size_t _i = 0; _i < _a_rows; ++_i)",
            "      if (zero_mode == 1 || (zero_mode == 2 && _i % 2 == 0))",
            "        a_tbl[_i] = zero_arr.data();",
        ]
    else:
        strides["a_stride"] = "kc + 8"
        extents["a"] = "kc + 8"  # NEON loads 8 bytes even in the K tail.
    return FixtureSpec(
        dims=dims, strides=strides, extents=extents,
        out_buffers=["c"], out_extent="nc",
        elem="int8_t", kind="SINT", out_elem=output, out_kind=_kind(output),
        params=params, extra_params=extra_params, indirect=indirect_spec,
        weights=weights, raw_decls=raw, scalar_reads=["a0"], index_reads=["a0"],
        config_tag="qc8w_finite_fp_concrete_params_arm64",
        pre_defines=["#undef XNN_ARCH_ARM64", "#define XNN_ARCH_ARM64 1"])


def _qd8_qc8w_gemm_fixture() -> "FixtureSpec":
    return _qc8w_fixture(dynamic=True)


def _qd8_qc8w_igemm_fixture() -> "FixtureSpec":
    return _qc8w_fixture(dynamic=True, indirect=True)


def _qs8_qc8w_gemm_fixture() -> "FixtureSpec":
    return _qc8w_fixture(dynamic=False)


def _packw_fixture() -> "FixtureSpec":
    # x32-packw-x2-gemm-goi (NR=2, KR=1, SR=1): packs g groups of GOI weights+bias
    # into [bias×2][k-major 2-wide]… blocks + an extra_bytes gap. Pure BV → full domain.
    # Unwritten bytes (gaps + odd-nc bias slot) → shared_out; odd-nc remainder → scalar movs.
    # bias may be NULL (the kernel then writes zero bias): variant bias_null.
    blk_u32 = "(2 + 2 * kc + extra_bytes / sizeof(uint32_t))"   # block elems incl. gap
    return FixtureSpec(
        dims={"g": [1, 2], "nc": [1, 2, 3, 4, 5], "kc": [1, 2, 3, 4], "extra_bytes": [0, 8],
              "bias_null": [0, 1]},
        nullable={"bias": "bias_null"},
        strides={"nr": "2", "kr": "1", "sr": "1"},
        extents={"weights": "g * nc * kc * sizeof(uint32_t) + 64",
                 "bias": "g * nc * sizeof(uint32_t) + 64",
                 "scale": "16", "params": "16",                   # unused void* args
                 "packed_weights": f"g * ((nc + 1) / 2) * {blk_u32} * sizeof(uint32_t) + 64"},
        out_buffers=["packed_weights"],
        out_extent=f"g * ((nc + 1) / 2) * {blk_u32}",
        shared_out=["packed_weights"], scalar_movs=True,
        index_reads=["bias"],     # a candidate may copy bias in C (`out[0] = bias[0];`): symbolic reads, stores
        # Tile columns past nc are padding the gemm consumer computes and discards.
        pad=(f"[&] {{ size_t _q = e % (((nc + 1) / 2) * {blk_u32}), _t = _q / {blk_u32}, _r = _q % {blk_u32};"
             f" return _r >= 2 + 2 * kc || 2 * _t + (_r < 2 ? _r : (_r - 2) % 2) >= nc; }}()"),
        elem="uint32_t", kind="UINT")


def _gemm_fixture(rvv_src: str = "") -> "FixtureSpec":
    # Full f32-gemm (mr≤4, nr=8): c[m][n] = clamp(bias[n] + Σ_k a[m][k]·w[k][n]).
    # kc in BYTES. Weights DUAL-PACKED (planes = 1+kc_floats: bias + weight rows):
    # NEON tile 8; the RVV side gets tile 8 too unless it steps `w += vl`, in which
    # case it gets vl-wide chunks (the VLEN=128 translations do).
    chunk = _rvv_chunked_weights(rvv_src)
    return FixtureSpec(
        dims={"mr": [1, 2, 3, 4], "nc": [1, 4, 8, 12], "kc": [4, 8]},   # mr 1-4 (both kernels 4-row); nc=12 → 2nd tile (cn_stride + a-rewind); kc BYTES
        strides={"a_stride": "kc",                       # a row pitch (bytes), tight
                 # c row pitch: a constant (as for igemm), so no stride depends on nc and
                 # the outer-loop induction over nc applies; rows stay apart for nc <= 256.
                 "cm_stride": "256 * sizeof(float)",
                 "cn_stride": "8 * sizeof(float)"},      # next nr-tile (unused at nc≤8)
        extents={"a": "mr * kc + 64", "c": "mr * cm_stride + 64"},
        out_buffers=["c"], out_extent=None,
        out_region={"rows": "mr", "cols": "nc",
                    "row_stride": "cm_stride", "col_stride": "sizeof(float)"},
        dual_weights={"w": {"planes": "1 + kc / sizeof(float)",   # bias + kc_floats weight rows
                            "ncols": "nc", "neon_tile": "8",
                            "rvv_tile": chunk or _rvv_padded_tile(rvv_src) or "8", "rvv_packed": bool(chunk)}},
        params=("xnn_f32_minmax_params",
                {"scalar.min": "-1e30f", "scalar.max": "1e30f"}),   # binds for full-domain inputs; narrowed for finite bands
        elem="float", kind="F32")


# Per-element-type minmax params for the pool family: (struct, {field: concrete}).
# The clamp must bind for some inputs, or a translation that drops or misapplies
# it still verifies. Integer bounds bind inside the full domain; the wide float
# bounds bind for full-domain inputs and are narrowed for finite input bands
# (see _binding_clamp).
_MINMAX_PARAMS = {
    "float":   ("xnn_f32_minmax_params", {"scalar.min": "-1e30f", "scalar.max": "1e30f"}),
    "int8_t":  ("xnn_s8_minmax_params",  {"scalar.min": "-50",    "scalar.max": "60"}),
    "uint8_t": ("xnn_u8_minmax_params",  {"scalar.min": "16",     "scalar.max": "240"}),
}


def _binding_clamp(fields: dict, input_range) -> dict:
    """With a finite input band, outputs never approach the fixtures' wide
    ±1e30 float clamp, so the clamp is never exercised. Bound it at half the
    band instead: the solver can then produce both clamped and unclamped outputs."""
    if not input_range:
        return fields
    lo, hi = input_range
    narrowed = {"-1e30f": f"{lo / 2}f", "1e30f": f"{hi / 2}f"}
    return {f: narrowed.get(v, v) for f, v in fields.items()}


def _maxpool_fixture(elem: str) -> "FixtureSpec":
    # Indirect-input: `const T** input` table of kernel_elements rows into a shared
    # symbolic buffer; per-channel max-reduction + [min,max] clamp. Max order matches
    # both sides → FULL domain (no input-range). f32 uses FP max, s8/u8 integer max.
    # GROW kernel_elements past the 9p pass boundary; inner set = channels×output_pixels.
    # The kernel advances `input` and `output` only by the increments, so each
    # output pixel gets its own kernel_elements pointers and its own output slot.
    return FixtureSpec(
        dims={"channels": [1, 2, 3, 4, 5, 7, 8],
              "output_pixels": [1, 2]},
        grow_dim="kernel_elements",
        strides={"input_offset": "0", "input_pixel_stride": "0",
                 "input_increment": "kernel_elements * sizeof(void*)",
                 "output_increment": f"channels * sizeof({elem})"},
        extents={"output": f"output_pixels * channels * sizeof({elem}) + 64"},
        out_buffers=["output"], out_extent="output_pixels * channels", out_region=None,
        # input[k] addresses [input_offset, input_offset+channels) elements per row.
        indirect={"input": {"count": "output_pixels * kernel_elements", "row": "input_offset + channels"}},
        params=_MINMAX_PARAMS[elem],
        elem=elem, kind=_kind(elem))


def _argmaxpool_fixture() -> "FixtureSpec":
    # Argmax pooling: per-channel max over a pooling_elements window, writing BOTH max
    # VALUES (float* output) and argmax POSITIONS (uint32_t* index) → dual MIXED-type
    # output (out_extents + out_elems). The two compares pin tie-breaking (strict `>`,
    # first-wins). GROW pooling_elements past the 9p8x boundary; inner = channels×output_pixels.
    # Each output pixel gets its own pointers and output/index slots (see maxpool).
    return FixtureSpec(
        dims={"channels": [1, 2, 3, 4, 5, 7, 8], "output_pixels": [1, 2]},
        grow_dim="pooling_elements",
        strides={"input_offset": "0", "input_pixel_stride": "0",
                 "input_increment": "pooling_elements * sizeof(void*)",
                 "output_increment": "channels * sizeof(float)",
                 "index_increment": "channels * sizeof(uint32_t)"},
        extents={"output": "output_pixels * channels * sizeof(float) + 64",
                 "index":  "output_pixels * channels * sizeof(uint32_t) + 64"},
        out_buffers=["output", "index"], out_extent=None, out_region=None,
        out_extents={"output": "output_pixels * channels", "index": "output_pixels * channels"},   # ANDed
        out_elems={"output": "float", "index": "uint32_t"},        # mixed-type: values F32, indices UINT
        indirect={"input": {"count": "output_pixels * pooling_elements", "row": "input_offset + channels"}},
        params=None,                          # argmaxpool has no params struct
        elem="float", kind="F32")


def _igemm_fixture(ks: int = 1, rvv_src: str = "") -> "FixtureSpec":
    # f32 igemm (mr<=1, nr=8). `a` is a table of ks/sizeof(void*) row pointers into
    # one shared symbolic buffer; the kernel adds a_offset to every entry except
    # those equal to `zero`. kc and ks are BYTES (the kernel steps p -= sizeof(void*)).
    # zero_mode 1 points every other entry at the all-zero row (the caller's padding
    # contract), exercising the skip path. Weights per 8-column tile are [bias(8)]
    # then ks_count*kc_floats rows of 8, laid out per side as in the gemm fixture.
    # FP-mul needs input-range for tractability.
    chunk = _rvv_chunked_weights(rvv_src)
    return FixtureSpec(
        # mr=[1] only: kernel asserts mr<=1. nc up to 16 crosses the NR=8 tile
        # (cn_stride + the a-table rewind); 1/4/5 hit the nc&4, nc&2, nc&1 tails.
        dims={"mr": [1], "nc": [1, 4, 5, 8, 9, 16], "kc": [4, 8], "ks": [ks * 8],
              "a_offset": [0, 4], "zero_mode": [0, 1]},
        # cm_stride is unused at mr <= 1 (both kernels ignore it); a constant keeps the
        # derived strides independent of nc, which the nc induction needs.
        strides={"cm_stride": "256 * sizeof(float)", "cn_stride": "8 * sizeof(float)"},
        extents={"c": "nc * sizeof(float) + 64"},
        out_buffers=["c"], out_extent="mr * nc", out_region=None,
        indirect={"a": {"count": "ks / sizeof(void*)",
                        "row": "a_offset / sizeof(float) + kc / sizeof(float) + 4"}},
        dual_weights={"w": {"planes": "1 + (ks / sizeof(void*)) * (kc / sizeof(float))",
                            "ncols": "nc", "neon_tile": "8",
                            "rvv_tile": chunk or _rvv_padded_tile(rvv_src) or "8", "rvv_packed": bool(chunk)}},
        weights={"zero": ("a_offset + kc + 64", "0.0f")},   # concrete all-zero row
        raw_decls=["    for (size_t _i = 0; _i < _a_rows; ++_i)",
                   "      if (zero_mode == 1 && _i % 2 == 0) a_tbl[_i] = zero_arr.data();"],
        params=("xnn_f32_minmax_params",         # binds for full-domain inputs; narrowed for finite bands
                {"scalar.min": "-1e30f", "scalar.max": "1e30f"}),
        elem="float", kind="F32")


def _dwconv_fixture(ksize: int = 3, rvv_src: str = "") -> "FixtureSpec":
    # Depthwise conv, kernel_size=ksize. `input` is a T** table with ksize pointers
    # per output pixel (input_stride advances it), rows padded to the NEON
    # 4-channel tile it over-reads. NEON weights per 4-channel tile are
    # [bias(4)][tap0(4)]...; an RVV kernel stepping `w += vl` reads vl-wide chunks,
    # so each side gets the same logical weights in its own layout. The clamp must
    # bind: the old params{} clamped every output to [0,0]. FP-mul needs input-range.
    ow_vals = [1, 2] if ksize <= 3 else [1]      # ksize products/output — cap work
    chunk = _rvv_chunked_weights(rvv_src)
    # input_offset applies to every entry except `zero`; zero_mode 1 points every
    # other entry at the all-zero row (the caller's padding), as for igemm.
    return FixtureSpec(
        dims={"output_width": ow_vals,           # inner representative set; channels is grow_dim
              "input_offset": [0, 16], "zero_mode": [0, 1]},
        grow_dim="channels",                     # grow channels 1,2,3,… until budget
        strides={"input_stride": f"{ksize} * sizeof(void*)", "output_increment": "0",
                 "input_pixel_stride": "0"},
        extents={"output": "output_width * channels * sizeof(float) + 64"},
        out_buffers=["output"], out_extent="output_width * channels", out_region=None,
        indirect={"input": {"count": f"output_width * {ksize}",
                            "row": "input_offset / sizeof(float) + ((channels + 3) / 4) * 4"}},
        dual_weights={"weights": {"planes": f"1 + {ksize}", "ncols": "channels", "neon_tile": "4",
                                  "rvv_tile": chunk or "4", "rvv_packed": bool(chunk)}},
        weights={"zero": ("((channels + 3) / 4) * 4 * sizeof(float) + 64", "0.0f")},   # concrete all-zero row
        raw_decls=["    for (size_t _i = 0; _i < _input_rows; ++_i)",
                   "      if (zero_mode == 1 && _i % 2 == 0) input_tbl[_i] = zero_arr.data();"],
        params=_MINMAX_PARAMS["float"],
        elem="float", kind="F32")


def _rdsum_fixture(elem: str, out_elem: str) -> "FixtureSpec":
    # Column reduction over a rows×channels matrix: output[c] += Σ_r input[r*input_stride+c].
    # int8/uint8 input, int32/uint32 INOUT output. NEON ref is the 7p7x variant. `zero` MUST
    # be a CONCRETE all-zeros buffer — the kernel's caller contract, load-bearing for
    # equivalence (a symbolic zero yields real CEX once channels exceed one NEON block).
    # ROWS GROW unbounded; channels = fixed representative set hitting every tail path.
    # Odd channels route the last lane through a scalar `*out += vgetq_lane` store → odd_tail_dim.
    return FixtureSpec(
        dims={"channels": [1, 2, 4, 7, 8, 15, 16, 17, 32, 33]},   # representative configs; rows is grow_dim
        grow_dim="rows",                              # rows grow 1,2,3,… until budget — unbounded verification
        strides={"input_stride": "channels"},        # tight row pitch (bytes; int8 ⇒ =channels)
        extents={"input": "rows * input_stride + 64",                 # symbolic matrix + OOB pad
                 "output": f"channels * sizeof({out_elem}) + 64"},
        out_buffers=["output"], out_extent="channels", out_region=None,
        weights={"zero": ("channels + 64", "0")},     # concrete all-zeros padding row (contract)
        params=("xnn_qs8_rsum_params", {}),           # dummy struct (unused; &params must be valid)
        odd_tail_dim="channels",   # odd channel → scalar `*out += vgetq_lane` tail (captured via side-channel)
        elem=elem, kind=_kind(elem), out_elem=out_elem, out_kind=_kind(out_elem))


def _strided_slice_fixture() -> "FixtureSpec":
    # y[i] = x[i*stride] — pure strided gather, no arithmetic → full-domain (the
    # FP-aware compare is bit-exact for a copy). Two bounds (n_out, stride); x is
    # read at stride steps so its extent is n_out*stride.
    return FixtureSpec(
        dims={"n_out": [1, 2, 4, 5, 8], "stride": [1, 2, 3]},
        strides={},
        extents={"x": "n_out * stride * sizeof(float) + 64",
                 "y": "n_out * sizeof(float) + 64"},
        out_buffers=["y"], out_extent="n_out", out_region=None,
        elem="float", kind="F32")


def _dwconv2d_chw_fixture() -> "FixtureSpec":
    # 3x3 s1 p1 depthwise conv on ONE channel plane (CHW), direct 2D. input/output are
    # flat H×cols planes (input_width is a BYTE pitch). Output contiguous → flat compare.
    # `weights` = bias + 9 taps (symbolic). `zero` = CONCRETE all-zeros halo. padding_top==1.
    # Small fixed grid (not production sizes — intractable for cvc5 FP); conv is local +
    # size-invariant so it covers all behaviors. FP-mul ⇒ needs --input-range.
    return FixtureSpec(
        dims={"input_height": [1, 2, 3, 4, 5, 6], "input_width": [16, 20, 32]},  # input_width in BYTES
        strides={"padding_top": "1"},                 # asserted == 1
        extents={"input": "input_height * input_width + 256",   # flat H×cols plane + OOB pad
                 "weights": "10 * sizeof(float) + 64",          # bias + 3x3 taps
                 "output": "input_height * input_width + 256"},
        out_buffers=["output"],
        out_extent="input_height * input_width / sizeof(float)",  # H*cols contiguous elements
        out_region=None,
        weights={"zero": ("input_width + 64", "0.0f")},          # concrete all-zeros halo row
        params=_MINMAX_PARAMS["float"],
        elem="float", kind="F32")


def _conv_hwc2chw_fixture(rvv_src: str = "") -> "FixtureSpec":
    # 3x3 s2 p1 FULL conv, 3 input channels (RGB), HWC in → CHW out (MobileNet first layer).
    # `input` HWC interleaved; `weights` = bias + 3 ic × 3x3 per oc (28 floats/oc); `zero` =
    # CONCRETE all-zeros HWC halo. CHW output with contiguous strides → flat compare.
    # Derived: output_width=(input_width+1)/2, H_out=(input_height+1)/2 (stride 2). Small grid
    # (not the 224x224x3 production size — ~150k symbolic floats, hopeless for cvc5 FP); conv
    # is local + size-invariant so it covers production behavior. FP-mul ⇒ needs --input-range.
    # Budget-gated (cheapest-first until --timeout); symbolic weights are ~88s/shape and grow.
    return FixtureSpec(
        dims={"input_height": [1, 2, 3, 4, 5],
              "input_width": [2, 3, 4, 5, 7, 8],
              "output_channels": [1, 2, 3, 4, 5, 8]},
        strides={"output_y_start": "0",
                 "output_y_end": "(input_height + 1) / 2",          # H_out (stride 2)
                 "input_padding_top": "1",
                 "output_height_stride": "((input_width + 1) / 2) * sizeof(float)",                       # contiguous rows
                 "output_channel_stride": "((input_height + 1) / 2) * ((input_width + 1) / 2) * sizeof(float)"},  # contiguous planes
        extents={"input": "input_height * input_width * 3 * sizeof(float) + 256",   # HWC + OOB pad
                 "output": "output_channels * ((input_height + 1) / 2) * ((input_width + 1) / 2) * sizeof(float) + 64"},
        out_buffers=["output"],
        out_extent="output_channels * ((input_height + 1) / 2) * ((input_width + 1) / 2)",  # CHW contiguous
        out_region=None,
        # NEON weights per 4-output-channel tile: 28 planes (bias, 27 taps) of 4. The
        # NEON tile is padded (the kernel reads a FULL 4-channel block per group); an
        # RVV kernel stepping `w += 28 * vl` gets vl-wide chunks instead. XNNPACK's
        # packer (xnn_pack_f32_dconv_oki_w) fills a tile's padding channels with the
        # last valid one: with fewer than 4 channels the kernel aliases the channel
        # pointers and the last store (channel 3) is the one that lands.
        dual_weights={"weights": {"planes": "28", "ncols": "output_channels", "neon_tile": "4",
                                  "rvv_tile": _rvv_chunked_weights(rvv_src) or "4",
                                  "rvv_packed": bool(_rvv_chunked_weights(rvv_src)),
                                  "pad": "replicate"}},
        weights={"zero": ("input_width * 3 * sizeof(float) + 64", "0.0f")},   # concrete all-zeros HWC halo row
        params=_MINMAX_PARAMS["float"],
        elem="float", kind="F32")


def _concat2_fixture() -> "FixtureSpec":
    # y = [a[0..na-1], b[0..nb-1]] — pure concatenation, full-domain. Two inputs,
    # one contiguous output of na+nb (both halves fully written).
    return FixtureSpec(
        dims={"na": [1, 2, 4, 5], "nb": [1, 2, 3, 8]},
        strides={},
        extents={"a": "na * sizeof(float) + 64",
                 "b": "nb * sizeof(float) + 64",
                 "y": "(na + nb) * sizeof(float) + 64"},
        out_buffers=["y"], out_extent="na + nb", out_region=None,
        elem="float", kind="F32")


def _split2_fixture() -> "FixtureSpec":
    # a = x[0..na-1]; b = x[na..na+nb-1] — pure split, full-domain. ONE input,
    # TWO contiguous outputs (a:na, b:nb) — exercises the multi-output compare.
    return FixtureSpec(
        dims={"na": [1, 2, 4, 5], "nb": [1, 2, 3, 8]},
        strides={},
        extents={"x": "(na + nb) * sizeof(float) + 64",
                 "a": "na * sizeof(float) + 64",
                 "b": "nb * sizeof(float) + 64"},
        out_buffers=["a", "b"], out_extent=None, out_region=None,
        out_extents={"a": "na", "b": "nb"},   # per-output element counts, ANDed
        elem="float", kind="F32")


def _vmulcaddc_fixture() -> "FixtureSpec":
    # 2-row per-channel: output = clamp(input*scale + bias). `channels` in BYTES.
    # Weights DUAL-PACKED (NEON scale[4]+bias[4] per 4-float tile; RVV per VLMAX(e32m4)
    # tile) from ONE shared symbolic logical set → both recover the same scale/bias at any
    # vlen. FP-mul → input-range.
    return FixtureSpec(
        dims={"rows": [1, 2, 3], "channels": [4, 16, 20, 32, 48]},   # BYTES; 20=5 floats (tail)
        strides={"input_stride": "channels", "output_stride": "channels"},   # tight (contiguous rows)
        extents={"input": "rows * channels + 64",
                 "output": "rows * channels + 64"},
        out_buffers=["output"], out_extent="rows * (channels / sizeof(float))", out_region=None,
        dual_weights={"weights": {"planes": "2",                         # plane0=scale, plane1=bias
                                  "ncols": "channels / sizeof(float)",   # logical channels
                                  "neon_tile": "4",                      # NEON fixed block
                                  "rvv_tile": "(vlen / 32) * 4",         # RVV VLMAX(e32m4)
                                  "rvv_packed": True}},                  # RVV advances by chunk (w += 2*vl)
        params=("xnn_f32_minmax_params", {"scalar.min": "-1e30f", "scalar.max": "1e30f"}),
        elem="float", kind="F32")


def _ibilinear_fixture() -> "FixtureSpec":
    # Bilinear interp (decomposed, no FMA → matched FP order). `input` is a T** table of
    # output_pixels*4 corner pointers. `weights` = alphah,alphav per pixel, broadcast both
    # sides → symbolic shared. FP sub/mul/add → input-range. No params struct.
    return FixtureSpec(
        dims={"channels": [1, 2, 4, 5, 8], "output_pixels": [1, 2]},
        strides={"input_offset": "0", "output_increment": "0"},
        extents={"weights": "output_pixels * 2 * sizeof(float) + 64",
                 "output": "output_pixels * channels * sizeof(float) + 64"},
        out_buffers=["output"], out_extent="output_pixels * channels", out_region=None,
        indirect={"input": {"count": "output_pixels * 4", "row": "input_offset + channels"}},
        elem="float", kind="F32")


def _match_fixture(sig, src="", rvv_src=""):
    # `src` lets us read a size constant the signature can't reveal (igemm/igemm-ks3
    # and dwconv/dwconv-9p share identical signatures — only `ks`/tap-count differ).
    # `rvv_src` tells weight-tiled fixtures how the candidate steps its weights.
    names = {p.name for p in sig.params}
    structs = {p.struct_name for p in sig.params if p.is_struct_ptr}
    if {"mr", "nc", "kc", "w", "c"} <= names:
        # Quantization and pointer-table arguments alone do not determine the
        # weight layout. Dispatch quantized families before generic f32 GEMM.
        if "xnn_qs8_qc8w_conv_minmax_params" in structs:
            return _qs8_qc8w_gemm_fixture()
        if "quantization_params" in names:
            if "xnn_f32_qc4w_minmax_params" in structs:
                return _qd8_qc4w_gemm_fixture(rvv_src)
            if "xnn_f32_minmax_params" in structs:
                if {"ks", "zero", "zero_data", "a_offset"} <= names:
                    return _qd8_qc8w_igemm_fixture()
                if "a_stride" in names:
                    return _qd8_qc8w_gemm_fixture()
            raise NotImplementedError("quantized GEMM signature needs an explicit weight-layout fixture")
    if {"input_height", "input_width", "padding_top"} <= names:
        return _dwconv2d_chw_fixture()    # 3x3 s1 p1 depthwise on a CHW plane (direct 2D)
    if {"input_height", "input_width", "output_y_start", "output_channels"} <= names:
        return _conv_hwc2chw_fixture(rvv_src)   # 3x3 s2 p1 c3 full conv, HWC in -> CHW out
    if {"input_stride", "output_stride", "block_width", "block_height"} <= names:
        elem = next((p.pointee for p in sig.params if p.is_pointer), "uint32_t")
        return _transpose_fixture(elem)   # element width from the buffer type
    if {"rows", "channels", "input_stride", "output_stride", "weights"} <= names:
        return _vmulcaddc_fixture()       # 2-row scale+bias+clamp (dual-packed weights)
    if ({"rows", "channels", "input_stride", "zero"} <= names
            and "output_stride" not in names and "weights" not in names):
        # rdsum: int8/uint8 input → int32/uint32 INOUT column-sum, concrete-zero padding.
        inp = next(p for p in sig.params
                   if p.is_pointer and p.points_to_const and p.name == "input")
        outp = next(p for p in sig.params
                    if p.is_pointer and not p.points_to_const and p.name == "output")
        ie = {"signed char": "int8_t", "unsigned char": "uint8_t"}.get(inp.pointee, inp.pointee)
        oe = {"int": "int32_t", "unsigned int": "uint32_t"}.get(outp.pointee, outp.pointee)
        return _rdsum_fixture(ie, oe)
    if ({"mr", "nc", "kc", "ks"} <= names
            and any(p.is_ptr_to_ptr for p in sig.params)):
        m = re.search(r'\bks\s*==\s*(\d+)', src)          # assert(ks == N)
        return _igemm_fixture(int(m.group(1)) if m else 1, rvv_src)
    if ({"channels", "output_width", "weights", "input_offset"} <= names
            and any(p.is_ptr_to_ptr for p in sig.params)):
        m = re.search(r'input\s*\[\s*ow\s*\*\s*(\d+)', src)  # input[ow * K + ...]
        return _dwconv_fixture(int(m.group(1)) if m else 3, rvv_src)
    if ({"output_pixels", "channels", "weights"} <= names
            and "kernel_elements" not in names
            and any(p.is_ptr_to_ptr for p in sig.params)):
        return _ibilinear_fixture()       # indirect 4-corner bilinear (f32)
    if {"n_out", "stride", "x", "y"} <= names and len(sig.params) == 4:
        return _strided_slice_fixture()   # pure strided gather (copy)
    if {"na", "nb", "a", "b", "y"} <= names and len(sig.params) == 5:
        return _concat2_fixture()         # pure concatenation (2 in → 1 out)
    if {"na", "nb", "x", "a", "b"} <= names and len(sig.params) == 5:
        return _split2_fixture()          # pure split (1 in → 2 out, multi-output compare)
    if {"mr", "nc", "kc", "a_stride", "cm_stride", "cn_stride"} <= names:
        return _gemm_fixture(rvv_src)
    if {"g", "nc", "kc", "nr", "kr", "sr", "extra_bytes"} <= names:
        return _packw_fixture()           # x32-packw NR=2 GOI weight packing (pure BV)
    if ({"output_pixels", "pooling_elements", "channels", "index"} <= names
            and any(p.is_ptr_to_ptr for p in sig.params)):
        return _argmaxpool_fixture()      # f32 dual-output: max values + argmax indices
    if ({"output_pixels", "kernel_elements", "channels", "input_offset"} <= names
            and any(p.is_ptr_to_ptr for p in sig.params)):
        # `const float**` pointee normalizes to "float *" — strip the inner `*`.
        # libclang canonicalizes int8_t→"signed char"; map back to the stdint name.
        ind = next(p.pointee for p in sig.params if p.is_ptr_to_ptr)
        elem = ind.replace("*", "").strip()
        elem = {"signed char": "int8_t", "unsigned char": "uint8_t"}.get(elem, elem)
        if elem in _MINMAX_PARAMS:        # f32/s8/u8 maxpool; f16 defers.
            return _maxpool_fixture(elem)
    return None


def generate_multidim(neon_path, rvv_path, name, neon, rvv, spec, input_range=None,
                      params_symbolic=False) -> str:
    weights = set(spec.weights)
    indirect = set(spec.indirect)
    dual = set(spec.dual_weights)
    param_names = {p.name for p in neon.params if p.is_struct_ptr}
    inputs = [p.name for p in neon.params
              if p.is_pointer and p.points_to_const and not p.is_struct_ptr
              and not p.is_ptr_to_ptr
              and p.name not in spec.dims and p.name not in weights and p.name not in dual]
    outputs = [p.name for p in neon.params
               if p.is_pointer and not p.points_to_const and not p.is_ptr_to_ptr
               and p.name not in spec.dims]
    elem = spec.elem
    oelem, okind = spec.out_elem, spec.out_kind   # output type (rdsum: int8 in, int32 out)
    ulp_tol = _ulp_tol_for(name)                  # 0 = bit-exact; >0 relaxes FP outputs to a ULP bound
    neon_src, rvv_src = _kernel_source(neon_path), _kernel_source(rvv_path)
    _refuse_c_float_tests(neon_src, rvv_src)
    decls = [f"    const size_t {s} = {e};" for s, e in spec.strides.items()]
    # Entry asserts over the integer arguments, checked before any buffer is
    # built. A reference assert failing means the fixture derived an illegal
    # argument (error); a candidate-only assert failing means the candidate
    # rejects a legal shape (a failure, not a skip).
    for expr, who in spec.preconditions:
        status, code = ("PRECONDITION", 3) if who == "reference" else ("CANDIDATE_REJECTS", 1)
        lit = expr.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")   # printf format
        decls.append(f'    if (!({expr})) {{ printf("{{\\"status\\":\\"{status}\\",\\"assert\\":\\"{lit}\\"}}\\n"); return {code}; }}')
    # Indirect inputs: `const T** name` is a table of `count` row pointers into one
    # shared symbolic buffer. Both kernels read the same table → same symbols, so
    # an intrinsic load resolves to identical bytes on each side.
    _ibind = "bindScalarRef" if elem in _FP_CTYPES else "bind"  # FP indirect rows read scalar-wise (RVV vfmacc_vf)
    for nm, spc in spec.indirect.items():
        cnt, row = spc["count"], spc["row"]
        decls += [
            f"    const size_t _{nm}_rows = ({cnt}), _{nm}_row = ({row});  // ptr count, elems/row",
            f"    std::vector<{elem}> {nm}_data(_{nm}_rows * _{nm}_row + 16, 0);",
            # +16 elem slack matches the backing pad so XNN_OOB_READS tail over-reads stay in-bounds
            f'    auto& _b_{nm} = ctx.registerBuffer("{nm}", (_{nm}_rows * _{nm}_row + 16) * sizeof({elem})); _b_{nm}.{_ibind}({nm}_data.data());',
            f"    std::vector<const {elem}*> {nm}_tbl(_{nm}_rows);",
            f"    for (size_t _k = 0; _k < _{nm}_rows; _k++) {nm}_tbl[_k] = {nm}_data.data() + _k * _{nm}_row;",
        ]
    # extents are byte-counts; size the typed backing array to cover them.
    nelem = lambda ext: f"({ext}) / sizeof({elem}) + 1"
    # FP inputs may be read scalar-wise (`*a0++` → `_vf_`); the NaN-payload
    # sentinel bridges those reads to the same symbolic bytes the intrinsic
    # loads see. Int inputs use the salt_scalar rewrite instead.
    bindm = "bindScalarRef" if elem in _FP_CTYPES else "bind"
    for b in inputs:                    # symbolic inputs (e.g. activations)
        ext = spec.extents[b]
        decls += [f"    std::vector<{elem}> {b}_arr({nelem(ext)}, 0);",
                  f'    auto& _b_{b} = ctx.registerBuffer("{b}", ({ext})); _b_{b}.{bindm}({b}_arr.data());']
    for b in outputs:
        ext = spec.extents[b]
        _oeb = spec.out_elems.get(b, oelem)         # per-output C type (argmaxpool: output=float, index=uint32_t)
        _onb = f"({ext}) / sizeof({_oeb}) + 1"      # backing count in that type
        # In-out / accumulator outputs (rdsum: `output += Σ`) load prior contents, so
        # both sides must start from ONE shared symbolic image; pure-write outputs keep
        # independent poison so a missing write is caught.
        inout = (_output_is_read(neon_src, b) or _output_is_read(rvv_src, b)
                 or b in spec.shared_out)   # unwritten holes (packw gaps) need one shared image too
        decls += [f"    std::vector<{_oeb}> {b}_neon_arr({_onb}, 0), {b}_rvv_arr({_onb}, 0);",
                  f'    auto& _b_{b}_neon = ctx.registerBuffer("{b}_neon", ({ext})); _b_{b}_neon.bind({b}_neon_arr.data());',
                  f'    auto& _b_{b}_rvv  = ctx.registerBuffer("{b}_rvv",  ({ext})); _b_{b}_rvv.bind({b}_rvv_arr.data());']
        if inout:
            decls.append(
                f'    for (size_t _i = 0; _i < ({ext}); _i++) '
                f'_b_{b}_rvv.setByte(_i, _b_{b}_neon.getByte(_i));  // shared init (in-out)')
        if b in spec.shared_out:        # remember the shared image: holes the reference leaves are don't-care
            decls.append(
                f'    std::vector<Term> _{b}_init; _{b}_init.reserve({ext}); '
                f'for (size_t _i = 0; _i < ({ext}); _i++) _{b}_init.push_back(_b_{b}_neon.getByte(_i));')
    for b, (ext, fill) in spec.weights.items():   # concrete uniform weights (layout-invariant)
        decls += [f"    std::vector<{elem}> {b}_arr({nelem(ext)}, {fill});",
                  f'    ctx.registerConcreteBuffer("{b}", {b}_arr.data(), ({ext}));']
    ebits = _BYTES[elem] * 8
    # Dual-packed weights: one shared symbolic logical set `wlog` (plane-major:
    # wlog[p*N + col]) is placed into TWO physical buffers at each dialect's tile
    # offsets. `planes` = logical planes per column (vmulcaddc: 2 = scale,bias;
    # GEMM: 1+kc_floats = bias + weight rows). NEON tiles are uniform/padded (width
    # neon_tile); RVV tiles are `rvv_packed` (advance by chunk, vmulcaddc) or padded
    # (advance by full tile, GEMM `w += nr`). Both kernels recover wlog[p*N+col].
    # FP layouts get the scalar-ref sentinel too: RVV conv reads `w[k]` natively.
    # An RVV (Phase-1) reference in the reference slot reads its weights the way an RVV candidate does: in chunks of
    # its own vl when it steps the weight pointer by a vl taken from the remaining count (`w += vl` after
    # `vl = __riscv_vsetvl_e32m8(nc)`), not in the Neon kernel's fixed tile. A vl from a constant
    # (`__riscv_vsetvl_e32m1(4)`, the AE translations' 128-bit lanes) keeps the Neon tile.
    ref_chunk = None
    if "__riscv_" in neon_src and re.search(r"__riscv_vsetvl_e32m\d\s*\(\s*[A-Za-z_]", _without_function(neon_src, "pack_weights")):
        try:
            ref_chunk = _rvv_chunked_weights(neon_src)
        except NotImplementedError:
            ref_chunk = None
    for nm, dw in spec.dual_weights.items():
        P, N, NT, RT = dw["planes"], dw["ncols"], dw["neon_tile"], dw["rvv_tile"]
        packed = dw.get("rvv_packed", False)
        if ref_chunk:
            NT = ref_chunk
        # Padding columns of a tile: unconstrained (never stored), or copies of the
        # last valid column when the packer replicates it ("pad": "replicate").
        rep = dw.get("pad") == "replicate"
        ncols_n = f"((_{nm}_N + _{nm}_NT - 1) / _{nm}_NT) * _{nm}_NT" if rep else f"_{nm}_N"
        ncols_r = f"((_{nm}_N + _{nm}_RT - 1) / _{nm}_RT) * _{nm}_RT" if rep else f"_{nm}_N"
        decls += [
            f"    const size_t _{nm}_P = ({P}), _{nm}_N = ({N});  // planes, logical columns",
            f'    auto& _wlog_{nm} = ctx.registerBuffer("{nm}_log", _{nm}_P * _{nm}_N * sizeof({elem}));',
            f"    const size_t _{nm}_NT = ({NT});",
            f"    std::vector<{elem}> {nm}_neon_arr(((_{nm}_N + _{nm}_NT - 1) / _{nm}_NT) * _{nm}_P * _{nm}_NT + 8, 0);",
            f'    auto& _wN_{nm} = ctx.registerBuffer("{nm}_neon", {nm}_neon_arr.size() * sizeof({elem})); _wN_{nm}.{bindm}({nm}_neon_arr.data());',
            f"    size_t _{nm}_RT = ({RT}); if (_{nm}_RT < 1) _{nm}_RT = 1;",
            f"    std::vector<{elem}> {nm}_rvv_arr(((_{nm}_N + _{nm}_RT - 1) / _{nm}_RT) * _{nm}_P * _{nm}_RT + 8, 0);",
            f'    auto& _wR_{nm} = ctx.registerBuffer("{nm}_rvv", {nm}_rvv_arr.size() * sizeof({elem})); _wR_{nm}.{bindm}({nm}_rvv_arr.data());',
        ]
        if ref_chunk:   # RVV reference, chunked: tile width vl=min(remaining,NT), offset advances by P*vl
            decls += [
                f"    {{ size_t _off = 0, _c = 0;",
                f"      while (_c < _{nm}_N) {{",
                f"        size_t _vl = (_{nm}_N - _c < _{nm}_NT) ? (_{nm}_N - _c) : _{nm}_NT;",
                f"        for (size_t _j = 0; _j < _vl; _j++)",
                f"          for (size_t _p = 0; _p < _{nm}_P; _p++) {{",
                f"            Term _v = _wlog_{nm}.loadScalar((_p * _{nm}_N + _c + _j) * sizeof({elem}), {ebits});",
                f"            _wN_{nm}.storeScalar((_off + _p * _vl + _j) * sizeof({elem}), _v, {ebits});",
                f"          }}",
                f"        _off += _{nm}_P * _vl; _c += _vl;",
                f"      }} }}",
            ]
        else:
            decls += [
            f"    for (size_t _col = 0; _col < {ncols_n}; _col++) {{",
            f"      size_t _t = _col / _{nm}_NT, _j = _col % _{nm}_NT, _lc = _col < _{nm}_N ? _col : _{nm}_N - 1;",
            f"      for (size_t _p = 0; _p < _{nm}_P; _p++) {{",
            f"        Term _v = _wlog_{nm}.loadScalar((_p * _{nm}_N + _lc) * sizeof({elem}), {ebits});",
            f"        _wN_{nm}.storeScalar((_t * _{nm}_P * _{nm}_NT + _p * _{nm}_NT + _j) * sizeof({elem}), _v, {ebits});",
            f"      }}",
            f"    }}",
        ]
        if packed:   # RVV chunked: tile width vl=min(remaining,RT), offset advances by P*vl
            decls += [
                f"    {{ size_t _off = 0, _c = 0;",
                f"      while (_c < _{nm}_N) {{",
                f"        size_t _vl = (_{nm}_N - _c < _{nm}_RT) ? (_{nm}_N - _c) : _{nm}_RT;",
                f"        for (size_t _j = 0; _j < _vl; _j++)",
                f"          for (size_t _p = 0; _p < _{nm}_P; _p++) {{",
                f"            Term _v = _wlog_{nm}.loadScalar((_p * _{nm}_N + _c + _j) * sizeof({elem}), {ebits});",
                f"            _wR_{nm}.storeScalar((_off + _p * _vl + _j) * sizeof({elem}), _v, {ebits});",
                f"          }}",
                f"        _off += _{nm}_P * _vl; _c += _vl;",
                f"      }} }}",
            ]
        else:        # RVV padded: uniform tiles of width RT (advance by P*RT, like GEMM `w += nr`)
            decls += [
                f"    for (size_t _col = 0; _col < {ncols_r}; _col++) {{",
                f"      size_t _t = _col / _{nm}_RT, _j = _col % _{nm}_RT, _lc = _col < _{nm}_N ? _col : _{nm}_N - 1;",
                f"      for (size_t _p = 0; _p < _{nm}_P; _p++) {{",
                f"        Term _v = _wlog_{nm}.loadScalar((_p * _{nm}_N + _lc) * sizeof({elem}), {ebits});",
                f"        _wR_{nm}.storeScalar((_t * _{nm}_P * _{nm}_RT + _p * _{nm}_RT + _j) * sizeof({elem}), _v, {ebits});",
                f"      }}",
                f"    }}",
            ]
    if spec.params and params_symbolic and spec.params[0] in ("xnn_f32_minmax_params", "xnn_f32_qc4w_minmax_params"):
        # Symbolic clamp bounds under XNNPACK's operator contract
        # (xnn_subgraph_check_output_min_max, src/subgraph/validation.c): neither
        # NaN, min <= max (equal allowed), infinities allowed. fp.leq is false on
        # NaN, so the one assertion states all three. Kernels read the fields
        # natively; the scalar-ref sentinels carry them to the intrinsic models.
        st = spec.params[0]
        _ld = lambda f: f"load_as_fp32(ctx.tm, _b_params.loadScalar(offsetof({st}, {f}), 32))"
        decls += [f"    {st} params{{}};",
                  f'    auto& _b_params = ctx.registerBuffer("params", sizeof({st}));',
                  "    _b_params.bindScalarRef(&params);",
                  f"    ctx.assert_(ctx.tm.mkTerm(Kind::FLOATINGPOINT_LEQ, {{{_ld('scalar.min')}, {_ld('scalar.max')}}}));"]
    elif spec.params:                   # concrete params struct (plain-C field reads)
        st, fields = spec.params
        fields = _binding_clamp(fields, input_range)
        decls.append(f"    {st} params{{}};")
        decls += [f"    params.{f} = {v};" for f, v in fields.items()]
        decls.append(f'    ctx.registerConcreteBuffer("params", &params, sizeof({st}));')
    for pname, (st, fields) in spec.extra_params.items():  # extra param structs (qd8)
        decls.append(f"    {st} {pname}_v{{}};")
        if params_symbolic and st == "xnn_qd8_quantization_params":
            # xnn_qd8_asymmetric_quantization_params (src/xnnpack/quantization.h): an int8 zero
            # point and inv_scale = (rmax - rmin) / 255 (1 for an empty range): positive, finite.
            _q = lambda f, b: f"_b_{pname}.loadScalar(offsetof({st}, {f}), {b})"
            decls += [f'    auto& _b_{pname} = ctx.registerBuffer("{pname}", sizeof({st}));',
                      f"    _b_{pname}.bindScalarRef(&{pname}_v);",
                      f"    ctx.assert_(ctx.bv_sge({_q('zero_point', 32)}, mk_bv_val(ctx.tm, 32, -128)));",
                      f"    ctx.assert_(ctx.bv_sle({_q('zero_point', 32)}, mk_bv_val(ctx.tm, 32, 127)));",
                      f"    {{ Term _s = load_as_fp32(ctx.tm, {_q('inv_scale', 32)});",
                      "      ctx.assert_(ctx.land_(ctx.fp32_geq(_s, mk_fp32_from_float(ctx.tm, 0x1p-149f)),"
                      " ctx.fp32_leq(_s, mk_fp32_from_float(ctx.tm, 0x1.fffffep+127f)))); }"]
            continue
        decls += [f"    {pname}_v.{f} = {v};" for f, v in fields.items()]
        decls.append(f'    ctx.registerConcreteBuffer("{pname}", &{pname}_v, sizeof({st}));')
    if input_range and elem == "float":  # optional: restrict F32 inputs to finite [lo,hi]
        lo, hi = input_range
        for b in inputs:                  # element count within capacity (floor, no +1)
            decls += _emit_input_range(f"_b_{b}", f"({spec.extents[b]}) / sizeof({elem})", lo, hi)
        for nm in spec.indirect:          # indirect-input bytes also need the finite band
            decls += _emit_input_range(f"_b_{nm}", f"_{nm}_rows * _{nm}_row", lo, hi)
        for nm in spec.dual_weights:       # logical weights (shared by both packings)
            decls += _emit_input_range(f"_wlog_{nm}", f"_{nm}_P * _{nm}_N", lo, hi)
    decls += spec.raw_decls             # raw constraint lines (e.g. qd8 ksum ≡ 0 mod 16)
    cfg_tag = spec.config_tag or ("finite" if input_range else "symbolic")
    if params_symbolic and spec.params and spec.params[0] == "xnn_f32_minmax_params":
        cfg_tag += "+sym-clamp"

    def call_args(side):
        out = []
        for p in neon.params:
            if side == "rvv" and p.name in spec.rvv_alias:   # the candidate's own (retiled) copy of an input
                out.append(spec.rvv_alias[p.name][0])
            elif p.name in spec.nullable:  # optional input (packw bias): NULL in one variant
                out.append(f"({spec.nullable[p.name]} ? nullptr : {p.name}_arr.data())")
            elif p.name in inputs or p.name in weights:
                out.append(f"{p.name}_arr.data()")
            elif p.name in dual:        # dual-packed weights: each side gets its own layout
                out.append(f"{p.name}_neon_arr.data()" if side == "neon"
                           else f"{p.name}_rvv_arr.data()")
            elif p.name in indirect:
                out.append(f"{p.name}_tbl.data()")
            elif p.name in outputs:
                out.append(f"{p.name}_{side}_arr.data()")
            elif p.name in spec.extra_params:   # before param_names: both are struct ptrs
                out.append(f"&{p.name}_v")
            elif p.name in param_names:
                out.append("&params")
            else:                       # dim or stride → pass by name (local/arg)
                out.append(p.name)
        return ", ".join(out)

    def contiguous_cmp(ob, count, celem, ckind):
        if ob in spec.shared_out:       # compare only what the reference wrote, minus layout pads
            return (f"buffers_equal_written(ctx.tm, _b_{ob}_neon, _b_{ob}_rvv, {count}, "
                    f"sizeof({celem}), ElementKind::{ckind}, {ulp_tol}, _{ob}_init"
                    + (", _pad)" if spec.pad else ")"))
        return (f"buffers_equal(ctx.tm, _b_{ob}_neon, _b_{ob}_rvv, {count}, "
                f"sizeof({celem}), ElementKind::{ckind}, {ulp_tol})")

    if spec.out_extents:                # multi-output: AND a contiguous compare per output
        terms = [contiguous_cmp(ob, ext, spec.out_elems.get(ob, oelem), _kind(spec.out_elems.get(ob, oelem)))
                 for ob, ext in spec.out_extents.items()]
        cmp_expr = terms[0]
        for t in terms[1:]:
            cmp_expr = f"ctx.land_({cmp_expr}, {t})"
    else:
        ob = spec.out_buffers[0]
        if spec.out_region:             # strided affine output region
            r = spec.out_region
            cmp_expr = (f"buffers_equal_strided(ctx.tm, _b_{ob}_neon, _b_{ob}_rvv, "
                        f"{r['rows']}, {r['cols']}, {r['row_stride']}, {r['col_stride']}, "
                        f"sizeof({oelem}), ElementKind::{okind}, {ulp_tol})")
        else:
            cmp_expr = contiguous_cmp(ob, spec.out_extent, oelem, okind)
    # Counterexample witness: on SAT, scan the (first) output buffer for the first
    # differing element and extract both models, so the repair LLM gets an actionable
    # "channel C: NEON=x RVV=y" instead of an empty shape. Strided/multi-output regions
    # report the first diff in the leading region; contiguous outputs scan [0,count).
    obits = _BYTES[oelem] * 8
    if spec.out_extents:
        _wob, _wcount = next(iter(spec.out_extents.items()))[0], next(iter(spec.out_extents.values()))
        _woff = f"e * sizeof({oelem})"
    elif spec.out_region:
        r = spec.out_region
        _wob, _wcount = spec.out_buffers[0], f"({r['rows']}) * ({r['cols']})"
        _woff = f"(e / ({r['cols']})) * ({r['row_stride']}) + (e % ({r['cols']})) * ({r['col_stride']})"
    else:
        _wob, _wcount = spec.out_buffers[0], spec.out_extent
        _woff = f"e * sizeof({oelem})"
    _din = inputs[0] if (inputs and inputs[0] in spec.extents) else None
    _cexin = ((f"      fprintf(stderr, \"CEX_IN:\");\n"
               f"      for (size_t _e = 0; _e < ({spec.extents[_din]})/sizeof({elem}); _e++) "
               f"fprintf(stderr, \" %lu\", strtoul(ctx.solver->getValue(_b_{_din}.loadScalar(_e*sizeof({elem}), {ebits})).getBitVectorValue(10).c_str(), nullptr, 10));\n"
               f"      fprintf(stderr, \"\\n\");\n") if _din else "")
    # Under a ULP tolerance the first BIT-differing element may still be within
    # tol; flag the first element whose ULP distance actually EXCEEDS tol, so the
    # repair hint names a genuine violator. Exact (tol=0) keeps raw bit inequality.
    _store = "fi = e; _nv = _nu; _vv = _vu; _found = true; break;"   # raw output bits
    if ulp_tol and oelem in _FP_CTYPES:
        _wb = obits - 1
        _wit_cmp = (
            f"        long long _ka = ((_nu >> {_wb}) & 1ULL) ? ((long long)(1ULL << {_wb}) - (long long)_nu) : (long long)_nu;\n"
            f"        long long _kb = ((_vu >> {_wb}) & 1ULL) ? ((long long)(1ULL << {_wb}) - (long long)_vu) : (long long)_vu;\n"
            f"        long long _ud = (_ka > _kb) ? (_ka - _kb) : (_kb - _ka);\n"
            f"        if (_ud > {ulp_tol}LL) {{ {_store} }}\n")
    else:                               # the first element that fails the comparison, not just its bits
        _wkind = _kind(spec.out_elems.get(_wob, oelem))
        _wit_cmp = (f"        if (!ctx.solver->getValue(element_matches_ref(ctx.tm, _n, _v, ElementKind::{_wkind}))"
                    f".getBooleanValue()) {{ {_store} }}\n")
    _padc = " || _pad(e)" if spec.pad and _wob in spec.shared_out else ""
    pad_decl = (f"    auto _pad = [&](size_t e) -> bool {{ return {spec.pad}; }};\n" if spec.pad else "")
    _wskip = (f"        if (ref_element_untouched(_b_{_wob}_neon, _{_wob}_init, _off, {obits // 8}){_padc}) continue;\n"
              if _wob in spec.shared_out else "")   # holes are don't-care, never the witness
    _fp0 = "true" if spec.out_elems.get(_wob, oelem) in _FP_CTYPES else "false"
    _si0 = "true" if _kind(spec.out_elems.get(_wob, oelem)) == "SINT" else "false"
    _more = ""                          # further outputs (argmaxpool's index), if the first matched
    for _ob, _ext in list((spec.out_extents or {}).items())[1:]:
        _oe = spec.out_elems.get(_ob, oelem)
        _ob_bits = _BYTES[_oe] * 8
        _more += (
            f"    if (!_found) for (size_t e = 0; e < ({_ext}); e++) {{\n"
            f"        size_t _off = e * sizeof({_oe});\n"
            f"        Term _n = ctx.solver->getValue(_b_{_ob}_neon.loadScalar(_off, {_ob_bits}));\n"
            f"        Term _v = ctx.solver->getValue(_b_{_ob}_rvv.loadScalar(_off, {_ob_bits}));\n"
            f"        if (!ctx.solver->getValue(element_matches_ref(ctx.tm, _n, _v, ElementKind::{_kind(_oe)})).getBooleanValue()) {{\n"
            f"            fi = e; _found = true; _wbuf = \"{_ob}\"; _wbits = {_ob_bits};\n"
            f"            _wfp = {'true' if _oe in _FP_CTYPES else 'false'}; _wsint = {'true' if _kind(_oe) == 'SINT' else 'false'};\n"
            f"            _nv = strtoull(_n.getBitVectorValue(10).c_str(), nullptr, 10);\n"
            f"            _vv = strtoull(_v.getBitVectorValue(10).c_str(), nullptr, 10);\n"
            f"            break;\n"
            f"        }}\n"
            f"    }}\n")
    cex_witness = (
        f"    size_t fi = 0; unsigned long long _nv = 0, _vv = 0; bool _found = false;\n"
        f"    const char* _wbuf = \"{_wob}\"; int _wbits = {obits}; bool _wfp = {_fp0}, _wsint = {_si0};\n"
        f"    for (size_t e = 0; e < ({_wcount}); e++) {{\n"
        f"        size_t _off = {_woff};\n"
        + _wskip +
        f"        Term _n = ctx.solver->getValue(_b_{_wob}_neon.loadScalar(_off, {obits}));\n"
        f"        Term _v = ctx.solver->getValue(_b_{_wob}_rvv.loadScalar(_off, {obits}));\n"
        f"        unsigned long long _nu = strtoull(_n.getBitVectorValue(10).c_str(), nullptr, 10);\n"
        f"        unsigned long long _vu = strtoull(_v.getBitVectorValue(10).c_str(), nullptr, 10);\n"
        + _wit_cmp +
        f"    }}\n"
        + _more +
        "    @@CEX_JSON@@\n"
        f"    if (std::getenv(\"SALT_CEX_DUMP\")) {{\n"
        + _cexin +
        f"      fprintf(stderr, \"CEX_NEON:\");\n"
        f"      for (size_t _e = 0; _e < ({_wcount}); _e++) "
        f"fprintf(stderr, \" %lu\", strtoul(ctx.solver->getValue(_b_{_wob}_neon.loadScalar(_e*sizeof({oelem}), {obits})).getBitVectorValue(10).c_str(), nullptr, 10));\n"
        f"      fprintf(stderr, \"\\nCEX_RVV:\");\n"
        f"      for (size_t _e = 0; _e < ({_wcount}); _e++) "
        f"fprintf(stderr, \" %lu\", strtoul(ctx.solver->getValue(_b_{_wob}_rvv.loadScalar(_e*sizeof({oelem}), {obits})).getBitVectorValue(10).c_str(), nullptr, 10));\n"
        f"      fprintf(stderr, \"\\n\"); fflush(stderr);\n"
        f"    }}"
    )
    # Odd-tail scalar store: rsum/rdsum write the final output lane via a native
    # `*out += vgetq_lane(...)` with no vector-store shim. vgetq_lane stashes the lane
    # in the scalar side-channel; reconstruct `prior + lane` into the NEON buffer's last
    # lane so the existing buffer compare covers the odd channel.
    if spec.odd_tail_dim:
        _otb, _od = spec.out_buffers[0], spec.odd_tail_dim
        scalar_tail_clear = "    ctx.clear_scalar_result();\n"
        scalar_tail_inject = (
            f"\n    if ((({_od}) & 1) && ctx.has_scalar_result()) {{\n"
            f"        size_t _olo = (({_od}) - 1) * sizeof({oelem});\n"
            f"        Term _oprior = _b_{_otb}_neon.loadScalar(_olo, {obits});\n"
            f"        _b_{_otb}_neon.storeScalar(_olo, ctx.add(_oprior, ctx.scalar_result()), {obits});\n"
            f"#ifdef SALT_CVC5_EXTENSIONS\n"
            f"        footprint_note(_fp_ref, \"{_otb}_neon\", _olo, _olo + {obits // 8} - 1, true, true);   // the native `*out +=`\n"
            f"#endif\n"
            f"    }}"
        )
    else:
        scalar_tail_clear = scalar_tail_inject = ""
    grow = spec.grow_dim                       # dim to grow continuously (unbounded sweep), else None
    gcross = spec.grow_cross                    # (d0, d1) grown cross-product-wise (expanding square), else None
    inner_dims = [] if gcross else list(spec.dims)   # cross-product dims (the representative configs)
    if gcross:
        dims = list(gcross)                    # verify() takes both grown dims
    elif grow:
        dims = [grow] + inner_dims             # verify() still takes ALL dims
    else:
        dims = inner_dims
    dim_sig = ", ".join(f"size_t {d}" for d in dims)
    dim_call = ", ".join(dims)
    dimfmt = ",".join("%zu" for _ in dims)
    # SALT_CEX_JSON=<dir>: the counterexample's buffers, for a concrete replay (VerificationContext::dump_model_json)
    cex_witness = cex_witness.replace(
        "@@CEX_JSON@@", f'if (std::getenv("SALT_CEX_JSON")) {{ char _sh[256]; snprintf(_sh, sizeof _sh, "{dimfmt}", '
                        f'{dim_call}); ctx.dump_model_json(_sh); }}')

    def nest(i):   # fixed cross-product grid (non-grow mode)
        if i == len(dims):
            # Each combo's query gets the remaining total budget; once it's spent,
            # the rest of the grid reports TIMEOUT without running.
            return (
                f'        if (budget > 0.0 && _rem() <= 0.0) {{\n'
                f'            printf("{{\\"status\\":\\"TIMEOUT\\",\\"dims\\":\\"{dimfmt}\\",\\"config\\":\\"{cfg_tag}\\"}}\\n", {dim_call});\n'
                f'        }} else {{\n'
                f'            int _r = verify({dim_call}, vlen, _rem());\n'
                f'            if (_r == 1) return 1;\n'
                f'        }}')
        d = dims[i]
        vals = ", ".join(str(v) for v in spec.dims[d])
        return (f"      for (size_t {d} : {{{vals}}}) {{\n{nest(i + 1)}\n      }}")

    def grow_inner(i):   # inner cross-product over the representative configs, with per-batch gate
        if i == len(inner_dims):
            return (
                "          if (budget > 0.0 && _rem() <= 0.0) {\n"
                f'              printf("{{\\"status\\":\\"TIMEOUT\\",\\"dims\\":\\"{dimfmt}\\",\\"config\\":\\"{cfg_tag}\\"}}\\n", {dim_call});\n'
                "              _batch_ok = false; break;\n"
                "          }\n"
                f"          int _r = verify({dim_call}, vlen, _rem());\n"
                "          if (_r == 1) return 1;\n"
                "          if (_r == 2) { _batch_ok = false; break; }"
            )
        d = inner_dims[i]
        vals = ", ".join(str(v) for v in spec.dims[d])
        return (f"        for (size_t {d} : {{{vals}}}) {{\n"
                "          if (!_batch_ok) break;\n"
                f"{grow_inner(i + 1)}\n"
                "        }")

    decls_str = "\n".join(decls)
    # Memory footprints: shared read-only buffers pair with themselves, each
    # output's candidate copy with the reference copy; dual-packed weights differ
    # in layout per side, so only their capacity bounds them.
    ro = list(inputs) + list(indirect) + list(weights) + (["params"] if spec.params else []) + list(spec.extra_params)
    mem_pairs = {b: b for b in ro} | {f"{b}_rvv": f"{b}_neon" for b in outputs}
    _nkind = _kind(spec.out_elems.get(_wob, oelem))
    nonvacuity = _nonvacuity_block(
        f"            for (size_t e = 0; e < ({_wcount}); e++) {{\n"
        f"                size_t _off = {_woff};\n"
        + (f"                if (ref_element_untouched(_b_{_wob}_neon, _{_wob}_init, _off, {obits // 8}){_padc}) continue;\n"
           if _wob in spec.shared_out else "")
        + f"                Term _t = _b_{_wob}_neon.loadScalar(_off, {obits});\n"
        f"                _diff = ctx.lor_(_diff, ctx.lnot_(element_equal(ctx.tm, _t, ctx.solver->getValue(_t), ElementKind::{_nkind})));\n"
        f"            }}\n",
        f'\\"dims\\":\\"{dimfmt}\\",\\"config\\":\\"{cfg_tag}\\"', dim_call)
    tracked_calls = _tracked_calls(
        f"{scalar_tail_clear}        neon_kernel::{neon.func_name}({call_args('neon')});",
        f"        rvv_kernel ::{rvv.func_name}({call_args('rvv')});",
        (scalar_tail_inject + "\n") if scalar_tail_inject else "",
        mem_pairs, ro, [f"{nm}_rvv" for nm in dual] + [b for _e, b in spec.rvv_alias.values()],
        f'\\"dims\\":\\"{dimfmt}\\",\\"config\\":\\"{cfg_tag}\\"', dim_call,
        # A packed output's whole layout (holes included) is the caller's buffer.
        {f"{b}_rvv": f"({spec.out_extent}) * sizeof({spec.out_elems.get(b, oelem)})"
         for b in spec.shared_out if spec.out_extent and b in outputs},
        **(dict(ref_allow=(f'{{"{spec.out_buffers[0]}_neon", (({spec.odd_tail_dim}) - 1) * sizeof({oelem}), '
                           f'((({spec.odd_tail_dim}) & 1) && ctx.has_scalar_result()) ? {obits // 8} : 0}}'),
                ref_post=(f'        if ((({spec.odd_tail_dim}) & 1) && ctx.has_scalar_result()) native_guard_sum(_ns, '
                          f'_b_{spec.out_buffers[0]}_neon, (({spec.odd_tail_dim}) - 1) * sizeof({oelem}), "reference");\n'),
                ref_pre=(f'if (({spec.odd_tail_dim}) & 1) ' + _seed_side_channel(
                    f"_b_{spec.out_buffers[0]}_neon", f"(({spec.odd_tail_dim}) - 1) * sizeof({oelem})")))
           if spec.odd_tail_dim else {}))
    neon_body = _strip_simd_includes(neon_src)
    if "__riscv_" in neon_src:          # an RVV (Phase-1) reference: the candidate's front-end cleanup applies to it
        neon_body = _strip_kernel_redeclarations(neon_body)
    rvv_body = _strip_kernel_redeclarations(_strip_simd_includes(rvv_src))
    if spec.scalar_movs:
        neon_body = _rewrite_scalar_movs(neon_body)
        rvv_body = _rewrite_scalar_movs(rvv_body)
    # Integer inputs: the parameters themselves seed the subscript rewrite, so a
    # candidate's own copy (`const int8_t* a0_ptr = a;`) is followed too.
    index_reads = list(spec.index_reads) + ([b for b in inputs if b not in spec.index_reads]
                                            if spec.index_reads and elem not in _FP_CTYPES else [])
    if index_reads:
        neon_body = _rewrite_index_reads(neon_body, index_reads)
        rvv_body = _rewrite_index_reads(rvv_body, index_reads)
        if elem not in _FP_CTYPES:          # stores of the symbolic values (or constants) into the outputs
            neon_body = _rewrite_scalar_stores(neon_body, outputs)
            rvv_body = _rewrite_scalar_stores(rvv_body, outputs)
    if spec.scalar_reads:
        neon_body = _rewrite_scalar_broadcast(neon_body, spec.scalar_reads)
        rvv_body = _rewrite_scalar_broadcast(rvv_body, spec.scalar_reads)
    if elem in _FP_CTYPES:
        rvv_body, _scratch = _rewrite_fp_scratch(rvv_body, outputs)
        if "__riscv_" in neon_src:      # an RVV (Phase-1) reference that stores into local arrays, as a candidate may
            neon_body, _ = _rewrite_fp_scratch(neon_body, outputs)
    if params_symbolic and spec.params and spec.params[0] in ("xnn_f32_minmax_params", "xnn_f32_qc4w_minmax_params"):
        # The candidate's clamp reads become symbolic scalars, so a C-level test on them (a NaN-bound branch,
        # `if (vmax != vmax)`) is decided by the min <= max constraint, or forks, like an integer parameter's.
        # Read natively, the NaN-payload sentinel made C see NaN and take the NaN-bound branch, which XNNPACK
        # never takes. `&p->scalar.max` stays a pointer load of the same symbolic bytes. A Neon reference keeps
        # the sentinel route (XNNPACK's kernels only hand the bounds to intrinsics), and so does an RVV (Phase-1)
        # reference that only hands them to intrinsics; one that tests them in C is rewritten like a candidate.
        cst = spec.params[0]

        def _symbolic_clamp(body, sig):
            for p in (q.name for q in sig.params if q.is_struct_ptr):
                for f in ("min", "max"):
                    body = re.sub(rf"(?<!&)(?<!& )\b{re.escape(p)}\s*->\s*scalar\.{f}\b",
                                  f"salt_param<float>({p}, offsetof({cst}, scalar.{f}))", body)
            body, fparams = _template_float_params(body, sig.func_name)
            return _rebind_tainted_locals(body, fparams)
        rvv_body = _symbolic_clamp(rvv_body, rvv)
        if "__riscv_" in neon_src and _tests_clamp_in_c(neon_body):
            neon_body = _symbolic_clamp(neon_body, neon)
    if params_symbolic:
        for pname, (st, _f) in spec.extra_params.items():
            if st != "xnn_qd8_quantization_params":
                continue
            def _qread(body):   # `q[i].field` read natively (not `&q[i].field`, a pointer load)
                for f, t in (("zero_point", "int32_t"), ("inv_scale", "float")):
                    body = re.sub(rf"(?<!&)\b{pname}\[(\w+)\]\.{f}\b",
                                  rf"salt_param<{t}>({pname}, (\1) * sizeof({st}) + offsetof({st}, {f}))", body)
                return _rebind_tainted_locals(body)
            neon_body, rvv_body = _qread(neon_body), _qread(rvv_body)
    # What the harness cannot follow of the symbolic params, after the rewrites: a native use of a clamp bound's
    # sentinel (a C test on it in a helper the rewrite did not reach sees a NaN), and any native read of a qd8
    # field left (an int zero point's sentinel bits are no value the model knows).
    if params_symbolic:
        _clamp = {spec.params[0]} & {"xnn_f32_minmax_params", "xnn_f32_qc4w_minmax_params"} if spec.params else set()
        _qd8 = {st for st, _f in spec.extra_params.values() if st == "xnn_qd8_quantization_params"}
        for side, b in (("candidate", rvv_body), ("reference", neon_body)):
            if hits := _native_param_uses(b, _clamp) + _native_param_reads(b, _qd8):
                raise NotImplementedError(f"{side}: native use of a value the verifier cannot track symbolically: "
                                          f"{hits[0]}")
    # Integer inputs have no NaN-payload sentinel: a native read through any alias
    # would see the zero-filled backing array and verify a degenerate input.
    if elem not in _FP_CTYPES:
        for side, b in (("reference", neon_body), ("candidate", rvv_body)):
            if hit := _native_reads(b, list(inputs) + list(spec.dual_weights)):
                raise NotImplementedError(f"{side}: unrewritten scalar read of symbolic input {hit}")
    neon_adapters = _neon_void_load_adapters(neon_body)
    pre_defines_str = ("\n".join(spec.pre_defines) + "\n") if spec.pre_defines else ""
    if gcross:
        d0, d1 = gcross
        # Cross-product growth: expand a square frontier. At level _n verify every shape with
        # max(d0,d1)=_n — the new `d0=_n` column (d1=1.._n) and `d1=_n` row (d0=1.._n-1) — so
        # after level N the full N×N grid is verified. Advance only if the whole frontier
        # passes; stop on counterexample (halt) or budget (TIMEOUT → PARTIAL "up to N×N").
        main_loop = (
            "    size_t _grow_last = 0;\n"
            f"    for (size_t _n = _gstart; (_gmax == 0 || _n <= _gmax); _n++) {{\n"
            "      bool _batch_ok = true;\n"
            f"      {{ size_t {d0} = _n;\n"
            f"        for (size_t {d1} = 1; {d1} <= _n && _batch_ok; {d1}++) {{\n"
            "          if (budget > 0.0 && _rem() <= 0.0) {\n"
            f'            printf("{{\\"status\\":\\"TIMEOUT\\",\\"dims\\":\\"{dimfmt}\\",\\"config\\":\\"{cfg_tag}\\"}}\\n", {dim_call});\n'
            "            _batch_ok = false; break;\n"
            "          }\n"
            f"          int _r = verify({dim_call}, vlen, _rem());\n"
            "          if (_r == 1) return 1;\n"
            "          if (_r == 2) { _batch_ok = false; break; }\n"
            "        }\n"
            "      }\n"
            f"      {{ size_t {d1} = _n;\n"
            f"        for (size_t {d0} = 1; {d0} < _n && _batch_ok; {d0}++) {{\n"
            "          if (budget > 0.0 && _rem() <= 0.0) {\n"
            f'            printf("{{\\"status\\":\\"TIMEOUT\\",\\"dims\\":\\"{dimfmt}\\",\\"config\\":\\"{cfg_tag}\\"}}\\n", {dim_call});\n'
            "            _batch_ok = false; break;\n"
            "          }\n"
            f"          int _r = verify({dim_call}, vlen, _rem());\n"
            "          if (_r == 1) return 1;\n"
            "          if (_r == 2) { _batch_ok = false; break; }\n"
            "        }\n"
            "      }\n"
            "      if (!_batch_ok) break;\n"
            "      _grow_last = _n;\n"
            "    }"
        )
        probes_block = ""
    elif grow:
        # Continuous-growth sweep: grow `grow` from _gstart, unbounded (or to _gmax),
        # verifying ALL representative inner configs per value; advance only if the whole
        # batch passes. Stop on counterexample (halt) or budget (a config prints TIMEOUT →
        # PARTIAL = "verified up to here"). Unbounded ⇒ only TIMEOUT/COUNTEREXAMPLE exit.
        main_loop = (
            "    size_t _grow_last = 0;\n"
            f"    for (size_t {grow} = _gstart; (_gmax == 0 || {grow} <= _gmax); {grow}++) {{\n"
            "      bool _batch_ok = true;\n"
            f"{grow_inner(0)}\n"
            "      if (!_batch_ok) break;\n"
            f"      _grow_last = {grow};\n"
            "    }"
        )
        probes_block = ""
    else:
        main_loop = nest(0)
        # Sparse probe shapes appended after the cross-product grid (e.g. deep-row
        # flush-boundary checks pinned to one channel width — see _rdsum_fixture).
        probe_lines = []
        for _pr in spec.probes:
            _pv = ", ".join(f"(size_t){_pr[d]}" for d in dims)   # size_t for %zu / verify() args
            probe_lines.append(
                f'      if (budget > 0.0 && _rem() <= 0.0) {{\n'
                f'          printf("{{\\"status\\":\\"TIMEOUT\\",\\"dims\\":\\"{dimfmt}\\",\\"config\\":\\"{cfg_tag}\\"}}\\n", {_pv});\n'
                f'      }} else {{\n'
                f'          int _r = verify({_pv}, vlen, _rem());\n'
                f'          if (_r == 1) return 1;\n'
                f'      }}')
        probes_block = ("\n" + "\n".join(probe_lines)) if probe_lines else ""
    # Growth mode reads start/max-grow from argv[1]/argv[2] (orchestrator min/max_batch),
    # defaults to a finite budget so an unbounded sweep can't spin forever.
    if gcross or grow:
        budget_default = "600.0"
        grow_argv = ("    size_t _gstart = (argc > 1) ? (size_t)atoi(argv[1]) : 1;\n"
                     "    size_t _gmax   = (argc > 2) ? (size_t)atoi(argv[2]) : 0;   // 0 = unbounded\n")
        if gcross:
            final_status = (f'    printf("{{\\"status\\":\\"ALL_VERIFIED\\",\\"grow_cross\\":\\"{gcross[0]}x{gcross[1]}\\",\\"reached\\":%zu}}\\n", _grow_last);')
        else:
            final_status = (f'    printf("{{\\"status\\":\\"ALL_VERIFIED\\",\\"grow_dim\\":\\"{grow}\\",\\"reached\\":%zu}}\\n", _grow_last);')
    else:
        budget_default = "0.0"
        grow_argv = ""
        final_status = '    printf("{\\"status\\":\\"ALL_VERIFIED\\",\\"start\\":0,\\"end\\":0,\\"configs\\":1}\\n");'
    return f"""// AUTO-GENERATED by emit.py — multi-dim harness ({name}).
#include "salt.hpp"
#include <cassert>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using namespace salt;
#ifdef SALT_CVC5_EXTENSIONS
#define SALT_METHOD(ctx) (ctx).method()
#else
#define SALT_METHOD(ctx) "smt"
#endif

#define restrict

// A witness element's numeric value from its raw bits (FP reinterpreted, ints
// sign-extended when signed).
static inline double _salt_bits_value(unsigned long long b, int bits, bool fp, bool sgn) {{
    if (fp && bits == 32) {{ uint32_t u = (uint32_t)b; float f; memcpy(&f, &u, 4); return f; }}
    if (fp && bits == 64) {{ double d; memcpy(&d, &b, 8); return d; }}
    if (sgn && bits < 64) return (double)((long long)(b << (64 - bits)) >> (64 - bits));
    return (double)b;
}}

// Symbolic accessors for rewritten scalar derefs: reads (`*p`) and stores (`*p = v`).
template <typename T>
static inline SymbolicScalar<T> salt_scalar(const T* p) {{
    auto& _sb = g_ctx->findBuffer((const void*)p);
    return SymbolicScalar<T>(_sb, _sb.ptrToByteOffset((const void*)p));
}}
// `*p++` on a symbolic buffer: the element at p, then p advances (a pointer to const or not).
template <typename T>
static inline SymbolicScalar<T> salt_scalar_postinc(const T*& p) {{ auto v = salt_scalar(p); ++p; return v; }}
template <typename T>
static inline SymbolicScalar<T> salt_scalar_postinc(T*& p) {{ auto v = salt_scalar((const T*) p); ++p; return v; }}
// ... and through a restrict-qualified pointer (a C `restrict` parameter), which a plain T*& does not bind.
template <typename T>
static inline SymbolicScalar<T> salt_scalar_postinc(const T* __restrict__& p) {{ auto v = salt_scalar(p); ++p; return v; }}
template <typename T>
static inline SymbolicScalar<T> salt_scalar_postinc(T* __restrict__& p) {{ auto v = salt_scalar((const T*) p); ++p; return v; }}
// Symbolic accessor for a param-struct field read (`q[0].zero_point`).
template <typename T>
static inline SymbolicScalar<T> salt_param(const void* base, size_t off) {{
    auto& _pb = g_ctx->findBuffer(base);
    return SymbolicScalar<T>(_pb, _pb.ptrToByteOffset(base) + off);
}}
template <typename T>
struct SaltDeref {{
    SymbolicBuffer& b; size_t off;
    SaltDeref& operator=(const SymbolicScalar<T>& v) {{
        Term t = v.term();                     // a float is stored as its IEEE bits, as the RVV stores do
        if constexpr (SymTraits<T>::is_fp && SymTraits<T>::bits == 32) t = rvv_store_fp32_as_bv(g_ctx->tm, t);
        b.storeScalar(off, t, SymTraits<T>::bits); return *this;
    }}
}};
template <typename T>
static inline SaltDeref<T> salt_deref(T* p) {{
    auto& _sb = g_ctx->findBuffer((const void*)p);
    return SaltDeref<T>{{_sb, _sb.ptrToByteOffset((const void*)p)}};
}}
// A local array the kernel stores vectors into: a symbolic buffer while the array is in scope, with fresh
// (unconstrained) contents, so its stores and the element reads after them are modelled.
struct SaltScratch {{
    SymbolicBuffer* b;
    SaltScratch(void* p, size_t n, const char* name) {{
        static size_t _k = 0;
        b = &g_ctx->registerBuffer(std::string("scratch@") + name + "#" + std::to_string(_k++), n);
        b->bind(p);
    }}
    ~SaltScratch() {{ g_ctx->unregisterBuffer(b); }}
    SaltScratch(const SaltScratch&) = delete;
    SaltScratch& operator=(const SaltScratch&) = delete;
}};
// A value C stores into a float buffer: a symbolic value as it is; a native float decoded (a scalar-ref
// sentinel stands for the symbolic input bytes it was copied from, any other float is that constant).
static inline SymbolicScalar<float> salt_fp_value(SymbolicScalar<float> v) {{ return v; }}
static inline SymbolicScalar<float> salt_fp_value(float v) {{ return SymbolicScalar<float>(mk_fp32_from_float(g_ctx->tm, v)); }}

{pre_defines_str}namespace neon_kernel {{
{neon_adapters}
{neon_body}
}} // namespace neon_kernel

namespace rvv_kernel {{
{rvv_body}
}} // namespace rvv_kernel

static int verify_once({dim_sig}, size_t vlen, double timeout_s) {{
    VerificationContext ctx(vlen);
    g_ctx = &ctx;
{decls_str}

{tracked_calls}{pad_decl}{nonvacuity}
    Term eq = {cmp_expr};
    ctx.assert_(ctx.lnot_(eq));
    if (timeout_s > 0) ctx.set_query_timeout_ms((uint64_t)(timeout_s * 1000.0));
    printf("{{\\"status\\":\\"SOLVING\\",\\"dims\\":\\"{dimfmt}\\",\\"config\\":\\"{cfg_tag}\\"}}\\n", {dim_call});
    CheckResult _c = ctx.check();
    if (_c == CheckResult::Unsat) {{
        printf("{{\\"status\\":\\"VERIFIED\\",\\"dims\\":\\"{dimfmt}\\",\\"config\\":\\"{cfg_tag}\\",\\"method\\":\\"%s\\"}}\\n", {dim_call}, SALT_METHOD(ctx));
        return 0;
    }}
    if (_c == CheckResult::Unknown) {{
        printf("{{\\"status\\":\\"TIMEOUT\\",\\"dims\\":\\"{dimfmt}\\",\\"config\\":\\"{cfg_tag}\\"}}\\n", {dim_call});
        return 2;
    }}
{cex_witness}
    printf("{{\\"status\\":\\"COUNTEREXAMPLE\\",\\"dims\\":\\"{dimfmt}\\",\\"config\\":\\"{cfg_tag}\\",\\"fail_buffer\\":\\"%s\\",\\"fail_index\\":%zu,\\"neon_out\\":%llu,\\"rvv_out\\":%llu,\\"neon_bits\\":\\"0x%llx\\",\\"rvv_bits\\":\\"0x%llx\\",\\"neon_val\\":\\"%.9g\\",\\"rvv_val\\":\\"%.9g\\"}}\\n", {dim_call}, _wbuf, fi, _nv, _vv, _nv, _vv, _salt_bits_value(_nv, _wbits, _wfp, _wsint), _salt_bits_value(_vv, _wbits, _wfp, _wsint));
    return 1;
}}

// A C-level branch on a symbolic value the assumptions leave open (a candidate's `if (a != 0.0f)` on an input it
// reads symbolically) forks: the shape is re-run once per outcome, each path asserting the outcomes it took, so the
// paths partition the inputs and the shape counts only if every path verifies (PATHS: how many there were).
int verify({dim_sig}, size_t vlen, double timeout_s) {{
    std::vector<std::vector<bool>> _todo{{std::vector<bool>{{}}}};
    size_t _paths = 0;
    auto _t0 = std::chrono::steady_clock::now();
    while (!_todo.empty()) {{
        salt_forced_branches = _todo.back(); _todo.pop_back(); salt_forced_next = 0;
        double _left = timeout_s;
        if (timeout_s > 0) {{
            _left = timeout_s - std::chrono::duration<double>(std::chrono::steady_clock::now() - _t0).count();
            if (_left <= 0) {{
                printf("{{\\"status\\":\\"TIMEOUT\\",\\"dims\\":\\"{dimfmt}\\",\\"config\\":\\"{cfg_tag}\\"}}\\n", {dim_call});
                salt_forced_branches.clear();
                return 2;
            }}
        }}
        try {{
            int _r = verify_once({dim_call}, vlen, _left);
            if (_r) {{ salt_forced_branches.clear(); return _r; }}
            _paths++;
        }} catch (const UndecidedBranch& _e) {{
            if (salt_forced_branches.size() >= 10) {{          // 1024 paths: give up on this shape
                printf("{{\\"status\\":\\"UNDECIDED\\",\\"dims\\":\\"{dimfmt}\\",\\"config\\":\\"{cfg_tag}\\"}}\\n", {dim_call});
                salt_forced_branches.clear();
                return 3;
            }}
            auto _t = salt_forced_branches, _f = salt_forced_branches;
            _t.push_back(true); _f.push_back(false);
            _todo.push_back(_f); _todo.push_back(_t);
        }}
    }}
    if (_paths > 1)
        printf("{{\\"status\\":\\"PATHS\\",\\"dims\\":\\"{dimfmt}\\",\\"config\\":\\"{cfg_tag}\\",\\"paths\\":%zu}}\\n", {dim_call}, _paths);
    salt_forced_branches.clear();
    return 0;
}}

int main(int argc, char** argv) {{
    setvbuf(stdout, NULL, _IOLBF, 0);   // line-buffer so per-batch status streams over a pipe
    size_t vlen   = (argc > 3) ? (size_t)atoi(argv[3]) : 256;
    double budget = (argc > 4) ? atof(argv[4]) : {budget_default};     // TOTAL wall-clock budget (s); 0 = none
    auto _t0 = std::chrono::steady_clock::now();
    auto _rem = [&]() -> double {{                         // seconds left in the total budget (0 = no limit)
        if (budget <= 0.0) return 0.0;
        double e = std::chrono::duration<double>(std::chrono::steady_clock::now() - _t0).count();
        return budget - e;
    }};
{grow_argv}{main_loop}{probes_block}
{final_status}
    return 0;
}}
"""


def shape_fixture(neon_path: str, rvv_path: str):
    """The fixture and growth plan for a multidimensional kernel pair, or None for
    an elementwise kernel. The fixture supplies the ABI (derived strides, buffer
    layouts, comparison); the plan's axes and variants replace its value lists."""
    from .shape_plan import derive_plan
    neon = parse_signature(neon_path, "test_neon")
    rvv = parse_signature(rvv_path, "test_rvv")
    neon_src, rvv_src = _kernel_source(neon_path), _kernel_source(rvv_path)
    spec = _match_fixture(neon, neon_src, rvv_src)
    if spec is None:
        return None
    dims = dict(spec.dims)
    for g in ([spec.grow_dim] if spec.grow_dim else []) + list(spec.grow_cross or ()):
        dims.setdefault(g, [1])
    plan = derive_plan(neon, [neon_src, rvv_src], spec.strides, dims)
    spec.dims = {a.name: [a.step] for a in plan.axes} | plan.variants
    spec.strides = plan.derived
    spec.grow_dim = spec.grow_cross = None
    spec.probes = []
    spec.preconditions = ([(a, "reference") for a in plan.preconditions]
                          + [(a, "candidate") for a in plan.candidate_checks])
    return neon, rvv, spec, plan


def generate_shape_harness(neon_path: str, rvv_path: str, name: str, input_range=None,
                           params_symbolic=False):
    """Harness that verifies ONE shape per run, for an external growth driver:
    `verify_<name> <vlen> <seconds> <dim...>` with dims in `list(spec.dims)`
    order. Returns (source, spec, plan)."""
    fx = shape_fixture(neon_path, rvv_path)
    if fx is None:
        raise NotImplementedError(f"{name}: no multidimensional fixture")
    neon, rvv, spec, plan = fx
    source = generate_multidim(neon_path, rvv_path, name, neon, rvv, spec, input_range, params_symbolic)
    prefix, sep, _ = source.rpartition("\nint main(")
    if not sep:
        raise ValueError("generated harness has no main()")
    dims = list(spec.dims)
    args = ", ".join(f"(size_t) std::stoull(argv[{i + 3}])" for i in range(len(dims)))
    driver = f"""
// One shape per run: {' '.join(dims)}.
int main(int argc, char** argv) {{
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc != {len(dims) + 3}) {{ fprintf(stderr, "usage: %s <vlen> <seconds> {' '.join(dims)}\\n", argv[0]); return 64; }}
    try {{
        size_t vlen = std::stoull(argv[1]);
        double seconds = std::stod(argv[2]);
        if (!(seconds > 0)) return 64;
        return verify({args}, vlen, seconds);
    }} catch (const std::exception& e) {{
        fprintf(stderr, "SHAPE_ERROR: %s\\n", e.what());
        return 3;
    }}
}}
"""
    return prefix + driver, spec, plan


def generate_harness(neon_path: str, rvv_path: str, name: str,
                     params_mode: str = "symbolic", input_range=None,
                     reference_isa: str = "neon", rvv_bit_exact: bool = True,
                     param_case: list = ()) -> str:
    """rvv_bit_exact=False compares an RVV reference's outputs under the Neon
    rule (+0 = -0, NaN = NaN) instead of bit for bit, so a Phase-2 check against
    the Phase-1 baseline uses the policy the Phase-1 check did, and the two compose.
    param_case: extra C++ conditions on the symbolic params (`_b_params` loads),
    one case of an exhaustive split, for a branch on params the constraints leave
    open (a_multiplier == b_multiplier); each case is a separate proof."""
    if reference_isa not in ("neon", "rvv"):
        raise ValueError("reference_isa must be neon or rvv")
    neon = parse_signature(neon_path, "test_rvv" if reference_isa == "rvv" else "test_neon")
    rvv = parse_signature(rvv_path, "test_rvv")
    if len(neon.params) != len(rvv.params):
        raise ValueError(f"signature arity mismatch: {len(neon.params)} vs {len(rvv.params)}")
    if reference_isa == "rvv":
        # `restrict` is defined away in the harness, which passes disjoint buffers.
        _norm = lambda t: re.sub(r"\s+", " ", re.sub(r"\brestrict\b", "", t)).strip()
        for a, b in zip(neon.params, rvv.params):
            # Current scalar rewrites use argument names from the reference.
            if a.name != b.name:
                raise NotImplementedError("RVV argument names must match for scalar rewrites")
            if (_norm(a.type_spelling), a.pointee, a.is_pointer, a.points_to_const, a.is_ptr_to_ptr,
                a.is_struct_ptr, a.is_int_scalar) != (
                _norm(b.type_spelling), b.pointee, b.is_pointer, b.points_to_const, b.is_ptr_to_ptr,
                b.is_struct_ptr, b.is_int_scalar):
                raise ValueError(f"candidate: signature differs from the reference's at {b.name}: "
                                 f"{_norm(b.type_spelling)} vs {_norm(a.type_spelling)}")
    _refuse_c_float_tests(_kernel_source(neon_path), _kernel_source(rvv_path))
    _fx = _match_fixture(neon, _kernel_source(neon_path), _kernel_source(rvv_path))   # multi-dim fixture family?
    if _fx is not None:
        if reference_isa == "rvv":
            raise NotImplementedError(
                "RVV-to-RVV multidimensional workloads need an explicit shape fixture; "
                "the NEON/RVV fixture sweep is not an optimization workload proof")
        if _fx.config_tag and _fx.config_tag.startswith("qc8w_"):
            if params_mode == "symbolic":
                raise NotImplementedError(
                    "qc8w fixtures use three concrete parameter profiles; "
                    "symbolic-params verification is not implemented")
            if input_range is not None:
                raise NotImplementedError(
                    "qc8w fixtures constrain packed FP planes explicitly; "
                    "input-range applies to homogeneous FP inputs only")
        return generate_multidim(neon_path, rvv_path, name, neon, rvv, _fx, input_range)
    roles = resolve_roles(neon)
    ulp_tol = 0 if reference_isa == "rvv" else _ulp_tol_for(name)   # 0 = bit-exact; >0 relaxes FP outputs to a ULP bound

    # `const void* params` (e.g. f32-f16-vcvt) is parsed as a const pointer and
    # lands in roles.inputs with pointee "void" — but it's an opaque, unused
    # param, not data. Strip it; the kernel call passes nullptr for it.
    void_param_names = {p.name for p in roles.inputs if p.pointee == "void"}
    roles.inputs = [p for p in roles.inputs if p.pointee != "void"]

    if roles.indirect:
        raise NotImplementedError("indirect (T**) inputs not yet supported")
    if len(roles.params) > 1:
        raise NotImplementedError("multiple params structs not yet supported")

    if params_mode not in ("symbolic", "concrete"):
        raise ValueError(f"params_mode must be 'symbolic' or 'concrete', got {params_mode!r}")

    # --- param-struct handling (user-selected mode; default symbolic) ---
    params_names = {p.name for p in roles.params}
    params_setup: list[str] = []
    config_pre = ""             # code emitted before verify() (concrete config table)
    cfg_param = ""              # extra verify() arg in concrete mode
    config_str = '"symbolic"'   # C expr for the verdict's config tag
    n_configs = 1
    is_concrete = False
    sym_param_rewrite = None    # (struct, pname, leaves) when symbolic-int rewrite applies
    if roles.params:
        st = roles.params[0].struct_name
        leaves = parse_struct_leaves(_XNN_MIN_H, st)
        cls = classify_param_struct(leaves)
        if cls == "none":                  # empty/dummy struct (xnn_f32_default_params)
            params_setup.append(f"    {st} params{{}};")
            config_str = '"no-params"'
        elif params_mode == "concrete":    # float OR int → concrete edge configs
            paths, combos = _gen_configs(leaves, struct_name=st)
            if not combos:
                raise NotImplementedError(f"no concrete configs for {st}")
            config_pre = _emit_config_table(st, paths, combos)
            # Register as a concrete buffer so intrinsic param-loads
            # (e.g. vld1_dup_f32(&params->scalar.x)) resolve via findBuffer.
            params_setup += [f"    {st} params{{}};", "    _set_params(params, cfg);",
                             f'    ctx.registerConcreteBuffer("params", &params, sizeof({st}));']
            cfg_param, config_str, n_configs, is_concrete = \
                ", size_t cfg", "_cfg_name(cfg)", len(combos), True
        elif cls == "float" and not any(_native_param_uses(_kernel_source(p), {st}) for p in (neon_path, rvv_path)):
            # symbolic all-float → sentinel: the kernels only hand the fields to intrinsics, which decode it
            params_setup += [f"    {st} params{{}};",
                             f'    auto& _b_params = ctx.registerBuffer("params", sizeof({st}));',
                             "    _b_params.bindScalarRef(&params);"]
        elif cls == "float":
            # A kernel that uses a float field natively (tests `slope < 1.0f` in C) reads it as a symbolic scalar
            # (salt_param<float>, below), so the test is decided by the assumptions or forks; read natively, the
            # sentinel is a NaN and C took the NaN branch whatever the symbolic value. No constraints: as above.
            params_setup += [f"    {st} params{{}};",
                             f'    auto& _b_params = ctx.registerBuffer("params", sizeof({st}));',
                             "    _b_params.bind(&params);"]
            sym_param_rewrite = (st, roles.params[0].name, leaves)
        else:                              # symbolic int — or mixed int+float
            # Int leaves rewrite to salt_param<int>() (symbolic BV); float leaves
            # to salt_param<float>() → SymbolicScalar<float>, which the FP _vf_
            # overloads (core.hpp) consume. Both load from the same symbolic
            # params buffer at disjoint offsets.
            pname = roles.params[0].name
            params_setup += [f"    {st} params{{}};",
                             f'    auto& _b_params = ctx.registerBuffer("params", sizeof({st}));',
                             "    _b_params.bind(&params);"]
            params_setup += _emit_param_constraints(st, leaves)   # validity (shift∈[0,31], …)
            _ranges = get_param_ranges(name)                      # per-field [lo,hi] bounds
            if _ranges:
                params_setup += _emit_param_ranges(st, leaves, _ranges)
            params_setup += [f"    ctx.assert_({c});   // case split" for c in param_case]
            sym_param_rewrite = (st, pname, leaves)               # rewrite params->field in bodies
    if void_param_names and not roles.params:    # only param was the opaque const void*
        config_str = '"no-params"'
    if len(roles.bounds) != 1:
        raise NotImplementedError(f"elementwise slice needs exactly one bound; got "
                                  f"{[p.name for p in roles.bounds]}")
    if not roles.outputs:
        raise ValueError("no output buffer found")
    bound = roles.bounds[0].name

    neon_src = _kernel_source(neon_path)
    rvv_src = _kernel_source(rvv_path)
    # Partition outputs: a native `*name <op>=` store is a scalar reduction (rsum/raddstore).
    # Array (vector-store) outputs are buffer-compared. A PURE reduction (no array output)
    # uses the scalar-result side-channel. A HYBRID (array + reduction — raddstoreexpminusmax:
    # output[] exp values + the `sum` horizontal reduce) compares the array(s) and DEFERS the
    # reduction: it's a horizontal reduce accumulated in a native float, not capturable as one
    # Term — and order-sensitive — so verify the exp output[] now, leave the sum for later.
    _red_store = lambda src: {p.name for p in roles.outputs
                              if re.search(r'\*\s*' + re.escape(p.name) + r'\s*[-+*/|&^]?=', src)}
    _red_outs = _red_store(neon_src + rvv_src)
    # A pure reduction (both kernels store every output as one scalar sum) is compared
    # through the one-capture check below; anything else would be a partial proof.
    if reference_isa == "rvv" and _red_outs and (
            [p for p in roles.outputs if p.name not in _red_outs] or _red_store(neon_src) != _red_store(rvv_src)):
        raise NotImplementedError(
            "RVV scalar reduction stores are not fully modeled; refusing a partial output proof")
    arr_outs = [p for p in roles.outputs if p.name not in _red_outs]
    byte_unit, unit_type = _detect_byte_unit(neon_src, bound)
    unit_bytes = 1
    if byte_unit:
        if unit_type not in _BYTES:
            raise NotImplementedError(f"byte-unit element type {unit_type!r} unknown")
        unit_bytes = _BYTES[unit_type]
        # `batch` is the byte-extent of the INPUT domain (measured in unit_type),
        # so every input must be unit_bytes wide for "batch / sizeof(unit)" to be
        # the element count N. OUTPUTS may be a different width — a convert kernel
        # narrows/widens (e.g. f32-f16-vcvt: float in, uint16 out; qu8-f32-vcvt:
        # uint8 in, float out) but still writes N elements; each output is sized
        # by its own pointee and compared with its own kind.
        if unit_bytes > 1:
            for p in roles.inputs:
                if _BYTES.get(p.pointee) != unit_bytes:
                    raise NotImplementedError("mixed input element sizes with byte-unit batch not yet supported")

    input_names = {p.name for p in roles.inputs}
    output_names = {p.name for p in roles.outputs}

    # Split-complex kernels (f32-vcmul) carve each buffer into [real | imag] via
    # `(uintptr_t) ptr + batch`, so every buffer spans split_mult*bound elements
    # and the output compare must cover both halves (not just [0, bound)).
    split_mult = 2 if re.search(
        r'\(\s*uintptr_t\s*\)\s*\w+\s*\+\s*' + re.escape(bound) + r'\b',
        neon_src + rvv_src) else 1
    cap_expr = f"{bound} + 16" if split_mult == 1 else f"{bound} * {split_mult} + 16"
    cmp_count = bound if split_mult == 1 else f"{bound} * {split_mult}"

    # buffer setup
    decls = []
    for p in roles.inputs:
        ct = p.pointee
        decls.append(f'    std::vector<{ct}> {p.name}_arr(cap, ({ct})0);')
        decls.append(f'    auto& _b_{p.name} = ctx.registerBuffer("{p.name}", cap * sizeof({ct}));'
                     f' _b_{p.name}.bind({p.name}_arr.data());')
    for p in roles.outputs:
        ct = p.pointee
        inout = _output_is_read(neon_src, p.name) or _output_is_read(rvv_src, p.name)
        decls.append(f'    std::vector<{ct}> {p.name}_neon_arr(cap, ({ct})0), {p.name}_rvv_arr(cap, ({ct})0);')
        decls.append(f'    auto& _b_{p.name}_neon = ctx.registerBuffer("{p.name}_neon", cap * sizeof({ct}));'
                     f' _b_{p.name}_neon.bind({p.name}_neon_arr.data());')
        decls.append(f'    auto& _b_{p.name}_rvv  = ctx.registerBuffer("{p.name}_rvv",  cap * sizeof({ct}));'
                     f' _b_{p.name}_rvv.bind({p.name}_rvv_arr.data());')
        if inout:
            # In-out / accumulator: both sides share one symbolic initial image
            # (pure outputs keep independent poison so a missing write is caught).
            decls.append(
                f'    for (size_t _i = 0; _i < cap * sizeof({ct}); _i++) '
                f'_b_{p.name}_rvv.setByte(_i, _b_{p.name}_neon.getByte(_i));  // shared init (in-out)')

    decls.extend(params_setup)
    # The sweep variable {bound} is always an *element* count; for byte-unit
    # kernels convert it to the byte count the kernel expects.
    if byte_unit:
        decls.append(f"    const size_t _kbound = {bound} * {unit_bytes};  // elements -> bytes")
    bound_call = "_kbound" if byte_unit else bound
    if input_range:                     # optional: restrict F32 inputs to finite [lo,hi]
        lo, hi = input_range
        for p in roles.inputs:
            if p.pointee == "float":
                decls += _emit_input_range(f"_b_{p.name}", "cap", lo, hi)
        if config_str == '"symbolic"':
            config_str = '"finite"'

    # positional call args (inputs shared; outputs differ neon/rvv)
    def call_args(side: str) -> str:
        out = []
        for p in neon.params:
            if p.name in input_names:
                out.append(f"{p.name}_arr.data()")
            elif p.name in output_names:
                out.append(f"{p.name}_{side}_arr.data()")
            elif p.name in params_names:
                out.append("&params")
            elif p.name in void_param_names:
                out.append("nullptr")
            elif p.name == bound:
                out.append(bound_call)
            else:
                out.append(p.name)
        return ", ".join(out)

    # output comparison over the logical region [0, bound) per ARRAY output
    cmps = []
    for p in arr_outs:
        cmps.append(f'        buffers_equal(ctx.tm, _b_{p.name}_neon, _b_{p.name}_rvv, '
                    f'{cmp_count}, sizeof({p.pointee}), ElementKind::{"UINT" if reference_isa == "rvv" and rvv_bit_exact else _kind(p.pointee)}, {ulp_tol})')
    cmp_expr = cmps[0] if cmps else ""
    for term in cmps[1:]:
        cmp_expr = f"ctx.land_({cmp_expr}, {term})"

    # counterexample scan: first differing element of the first array output
    first_out = arr_outs[0] if arr_outs else roles.outputs[0]
    fo_ct = first_out.pointee
    fo_bits = _BYTES[first_out.pointee] * 8
    # Witness predicate: under a ULP tolerance, flag the first element whose ULP
    # distance EXCEEDS tol (not the first bit-differing one). Exact (tol=0) keeps
    # raw decimal-string inequality.
    if ulp_tol and fo_ct in _FP_CTYPES:
        _wb = fo_bits - 1
        _lwit = (
            f'        unsigned long long _nu = strtoull(n.getBitVectorValue(10).c_str(), nullptr, 10);\n'
            f'        unsigned long long _vu = strtoull(v.getBitVectorValue(10).c_str(), nullptr, 10);\n'
            f'        long long _ka = ((_nu >> {_wb}) & 1ULL) ? ((long long)(1ULL << {_wb}) - (long long)_nu) : (long long)_nu;\n'
            f'        long long _kb = ((_vu >> {_wb}) & 1ULL) ? ((long long)(1ULL << {_wb}) - (long long)_vu) : (long long)_vu;\n'
            f'        long long _ud = (_ka > _kb) ? (_ka - _kb) : (_kb - _ka);\n'
            f'        if (_ud > {ulp_tol}LL) {{ fi = e; break; }}\n')
    else:
        _lwit = f'        if (n.getBitVectorValue(10) != v.getBitVectorValue(10)) {{ fi = e; break; }}\n'

    # Scalar-reduction output (rsum): the kernel writes its result via a scalar
    # accumulate `*output += vmv_x_s/vget_lane(...)`, NOT a vector store — so it
    # never reaches the symbolic output buffer.  Detect the `*output <op>=` store
    # (not the `T* output` param decl) and compare the captured reduction sum
    # (stashed in ctx by the scalar-extract shims) instead of the (unwritten,
    # unconstrained) output buffer.
    out0 = roles.outputs[0].name
    # Pure reduction (rsum: the only output is a `*out +=`) → scalar side-channel compare.
    # Hybrid (array output(s) + a separate reduction) → arr_outs buffer-compare above, with
    # the reduction deferred (not in arr_outs, so not compared).
    is_scalar_reduction = bool(_red_outs) and not arr_outs
    mem_inputs = sorted(input_names) + (["params"] if roles.params else [])
    mem_pairs = {b: b for b in mem_inputs} | {f"{b}_rvv": f"{b}_neon" for b in output_names}
    mem_fmt = '\\"batch\\":%zu,\\"vlen\\":%zu,\\"config\\":\\"%s\\"'
    mem_args = f"{bound}, vlen, {config_str}"
    # SALT_CEX_JSON=<dir>: the counterexample's buffers, for a concrete replay
    _cex_json = ('    if (std::getenv("SALT_CEX_JSON")) { char _sh[64]; snprintf(_sh, sizeof _sh, "%zu", (size_t)('
                 + bound + ')); ctx.dump_model_json(_sh); }\n')
    if is_scalar_reduction:
        # The output's prior value is the same symbolic value on both sides, read symbolically by a kernel that
        # folds it in itself (`vmv_s_x(*output)`, then `*output = sum`); each side's result is its prior value plus
        # the captured sum for `*out += sum`, or the captured sum for `*out = sum` (native_guard_sum tells which).
        _rp = [q for q in roles.outputs if q.name in _red_outs]
        if len(_rp) != 1 or _rp[0].pointee not in ("int", "unsigned int", "int32_t", "uint32_t"):
            raise NotImplementedError("a scalar reduction into other than one 32-bit integer output")
        _rb = _rp[0].name
        kernel_calls = (
            "    Term _neon_red, _rvv_red;\n    int _form_neon = 0, _form_rvv = 0;\n"
            f"    for (size_t _i = 0; _i < 4; _i++) _b_{_rb}_rvv.setByte(_i, _b_{_rb}_neon.getByte(_i));\n"
            f"    Term _prior = _b_{_rb}_neon.loadScalar(0, 32);\n") + _tracked_calls(
            f'        ctx.clear_scalar_result();\n'
            f'        neon_kernel::{neon.func_name}({call_args("neon")});\n'
            f'        _neon_red = ctx.scalar_result();\n'
            f'        if (ctx.scalar_result_count() != 1) {{ printf("{{\\"status\\":\\"FIXTURE_ERROR\\",\\"batch\\":%zu,\\"error\\":\\"reference captured %zu sums\\"}}\\n", (size_t){bound}, ctx.scalar_result_count()); return 3; }}',
            f'        ctx.clear_scalar_result();\n'
            f'        rvv_kernel ::{rvv.func_name}({call_args("rvv")});\n'
            f'        _rvv_red = ctx.scalar_result();\n'
            f'        if (ctx.scalar_result_count() != 1) {{ printf("{{\\"status\\":\\"CANDIDATE_REJECTS\\",\\"batch\\":%zu,\\"error\\":\\"candidate captured %zu sums\\"}}\\n", (size_t){bound}, ctx.scalar_result_count()); return 1; }}',
            "", mem_pairs, mem_inputs, [f"{b}_rvv" for b in sorted(_red_outs)], mem_fmt, mem_args,
            # The side channel: each kernel's one native `*out += sum` (element 0), checked to be exactly that.
            ref_allow=", ".join(f'{{"{b}_neon", 0, 4}}' for b in sorted(_red_outs)),
            ref_post=f'        _form_neon = native_guard_sum(_ns, _b_{_rb}_neon, 0, "reference", true);\n',
            cand_allow=", ".join(f'{{"{b}_rvv", 0, 4}}' for b in sorted(_red_outs)),
            cand_post=f'        _form_rvv = native_guard_sum(_ns, _b_{_rb}_rvv, 0, "candidate", true);\n',
            ref_pre="".join(_seed_side_channel(f"_b_{b}_neon", "0") for b in sorted(_red_outs)),
            cand_pre="".join(_seed_side_channel(f"_b_{b}_rvv", "0") for b in sorted(_red_outs)))
        cmp_expr = ("ctx.equal(_form_neon == 1 ? ctx.add(_prior, _neon_red) : _neon_red, "
                    "_form_rvv == 1 ? ctx.add(_prior, _rvv_red) : _rvv_red)")
        cex_block = (
            _cex_json +
            '    printf("{\\"status\\":\\"COUNTEREXAMPLE\\",\\"batch\\":%zu,\\"vlen\\":%zu,'
            '\\"config\\":\\"%s\\",\\"fail_index\\":0}\\n", ' + bound + ', vlen, ' + config_str + ');\n'
            '    return 1;'
        )
    else:
        kernel_calls = _tracked_calls(
            f'        neon_kernel::{neon.func_name}({call_args("neon")});',
            f'        rvv_kernel ::{rvv.func_name}({call_args("rvv")});',
            "", mem_pairs, mem_inputs, [f"{b}_rvv" for b in sorted(_red_outs)], mem_fmt, mem_args,
            # A hybrid's deferred reduction outputs are not compared (not part of the claim): native writes allowed.
            ref_allow=", ".join(f'{{"{b}_neon", 0, (size_t) -1}}' for b in sorted(_red_outs)),
            cand_allow=", ".join(f'{{"{b}_rvv", 0, (size_t) -1}}' for b in sorted(_red_outs)))
        cex_block = (
            f'    size_t fi = 0;\n'
            f'    for (size_t e = 0; e < {cmp_count}; e++) {{\n'
            f'        Term n = ctx.solver->getValue(_b_{first_out.name}_neon.loadScalar(e * sizeof({fo_ct}), {fo_bits}));\n'
            f'        Term v = ctx.solver->getValue(_b_{first_out.name}_rvv.loadScalar(e * sizeof({fo_ct}), {fo_bits}));\n'
            + _lwit +
            f'    }}\n'
            + _cex_json +
            '    printf("{\\"status\\":\\"COUNTEREXAMPLE\\",\\"batch\\":%zu,\\"vlen\\":%zu,'
            '\\"config\\":\\"%s\\",\\"fail_index\\":%zu}\\n", ' + bound + ', vlen, ' + config_str + ', fi);\n'
            '    return 1;'
        )

    decls_str = "\n".join(decls)
    if is_scalar_reduction:
        _nv_diff = "            _diff = ctx.lnot_(ctx.equal(_neon_red, ctx.solver->getValue(_neon_red)));\n"
    else:
        _nv_kind = "UINT" if reference_isa == "rvv" and rvv_bit_exact else _kind(fo_ct)
        _nv_diff = (f"            for (size_t e = 0; e < {cmp_count}; e++) {{\n"
                    f"                Term _t = _b_{first_out.name}_neon.loadScalar(e * sizeof({fo_ct}), {fo_bits});\n"
                    f"                _diff = ctx.lor_(_diff, ctx.lnot_(element_equal(ctx.tm, _t, ctx.solver->getValue(_t), ElementKind::{_nv_kind})));\n"
                    f"            }}\n")
    nonvacuity = _nonvacuity_block(_nv_diff, mem_fmt, mem_args)
    neon_body = _strip_simd_includes(neon_src)
    if reference_isa == "rvv":
        # An RVV reference (the Phase-1 baseline of an RVV/RVV proof) is an RVV kernel like the candidate:
        # its own `struct xnn_*;` lines would declare a distinct type inside neon_kernel as well.
        neon_body = _strip_kernel_redeclarations(neon_body)
    rvv_body = _strip_kernel_redeclarations(_strip_simd_includes(rvv_src))
    if is_scalar_reduction:                  # a read of the output's prior value (`*output`): its symbolic value
        for _b in sorted(_red_outs):
            nh, nb = _split_body(neon_body)
            rh, rb = _split_body(rvv_body)
            neon_body = nh + _rebind_tainted_locals(_rewrite_deref_reads(nb, _b))
            rvv_body = rh + _rebind_tainted_locals(_rewrite_deref_reads(rb, _b))
    # Scalar-broadcast inputs (`*input_b`) → symbolic accessor, so the broadcast
    # constant is symbolic rather than the concrete backing (which would be 0).
    scalar_bcast = (_scalar_broadcast_inputs(neon_src, input_names) |
                    _scalar_broadcast_inputs(rvv_src, input_names))
    if scalar_bcast:
        neon_body = _rewrite_scalar_broadcast(neon_body, scalar_bcast)
        rvv_body = _rewrite_scalar_broadcast(rvv_body, scalar_bcast)
    if sym_param_rewrite:
        _st, _pn, _lv = sym_param_rewrite
        neon_body = _rewrite_symbolic_params(neon_body, _st, _pn, _lv)
        try:
            rvv_body = _rewrite_symbolic_params(rvv_body, _st, _pn, _lv)
        except NotImplementedError as e:      # the candidate's own way of reading params (repair can change it)
            raise NotImplementedError(f"candidate: {e}") from None
    # What the harness cannot follow of the params, after the rewrites: with the struct bound symbolically, any
    # native field read left (in a helper, through an alias) sees the zero backing; with sentinels, a native use.
    if roles.params and cls != "none" and not is_concrete:
        _chk = (lambda b: _native_param_reads(b, {st})) if sym_param_rewrite else (lambda b: _native_param_uses(b, {st}))
        if hits := _chk(rvv_body):
            raise NotImplementedError(f"candidate: native use of a value the verifier cannot track symbolically: {hits[0]}")
        if hits := _chk(neon_body):
            raise NotImplementedError(f"reference: native use of a value the verifier cannot track symbolically: {hits[0]}")

    if is_concrete:
        main_loop = f"""    size_t _last = start ? start - 1 : 0;
    for (size_t cfg = 0; cfg < {n_configs}; cfg++) {{
      for (size_t b = start; (end == 0 || b <= end); b++) {{
        if (budget > 0.0 && _rem() <= 0.0) {{ printf("{{\\"status\\":\\"TIMEOUT\\",\\"batch\\":%zu,\\"vlen\\":%zu}}\\n", b, vlen); goto _done; }}
        int r = verify(b, vlen, cfg, _rem());   // per-query limit = remaining total budget
        if (r == 1) return 1;   // counterexample — stop
        if (r == 2) break;      // batch hit the budget — larger batches only get harder; next config
        if (b > _last) _last = b;
      }}
    }}
_done:
    printf("{{\\"status\\":\\"%s\\",\\"start\\":%zu,\\"end\\":%zu,\\"configs\\":{n_configs}}}\\n", std::getenv("SALT_EXECUTE_ONLY") ? "ALL_EXECUTED" : "ALL_VERIFIED", start, _last);"""
    else:
        main_loop = f"""    size_t _last = start ? start - 1 : 0;
    for (size_t b = start; (end == 0 || b <= end); b++) {{
        if (budget > 0.0 && _rem() <= 0.0) {{ printf("{{\\"status\\":\\"TIMEOUT\\",\\"batch\\":%zu,\\"vlen\\":%zu}}\\n", b, vlen); break; }}
        int r = verify_paths(b, vlen, _rem());  // per-query limit = remaining total budget
        if (r == 1) return 1;   // counterexample — stop
        if (r == 2) break;      // batch hit the budget — larger batches only get harder
        _last = b;
    }}
    printf("{{\\"status\\":\\"%s\\",\\"start\\":%zu,\\"end\\":%zu,\\"configs\\":1}}\\n", std::getenv("SALT_EXECUTE_ONLY") ? "ALL_EXECUTED" : "ALL_VERIFIED", start, _last);"""

    if not is_concrete:
        main_loop = """    // Each param branch the constraints leave open doubles the paths of a shape.
    auto verify_paths = [](size_t batch, size_t vlen_, double tlimit) -> int {
        std::vector<std::vector<bool>> todo{{}};
        size_t paths = 0;
        while (!todo.empty()) {
            salt_forced_branches = todo.back(); todo.pop_back(); salt_forced_next = 0;
            try {
                int r = verify(batch, vlen_, tlimit);
                if (r) return r;
                paths++;
            } catch (const UndecidedBranch&) {
                if (salt_forced_branches.size() >= 8) throw;     // 256 paths: give up loudly
                auto t = salt_forced_branches, f = salt_forced_branches;
                t.push_back(true); f.push_back(false);
                todo.push_back(f); todo.push_back(t);
            }
        }
        if (paths > 1) printf("{\\"status\\":\\"PATHS\\",\\"batch\\":%zu,\\"paths\\":%zu}\\n", batch, paths);
        salt_forced_branches.clear();
        return 0;
    };
""" + main_loop
    return f"""// AUTO-GENERATED by emit.py — uniform harness (no Shape/profile).
#include "salt.hpp"
#include <cassert>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using namespace salt;
#ifdef SALT_CVC5_EXTENSIONS
#define SALT_METHOD(ctx) (ctx).method()
#else
#define SALT_METHOD(ctx) "smt"
#endif

#define restrict          // C-only keyword; this harness compiles as C++

// Symbolic accessor for scalar-broadcast input reads (`*input_b`).
template <typename T>
static inline SymbolicScalar<T> salt_scalar(const T* p) {{
    auto& _sb = g_ctx->findBuffer((const void*)p);
    return SymbolicScalar<T>(_sb, _sb.ptrToByteOffset((const void*)p));
}}
// Symbolic accessor for params-struct field reads (`params->scalar.x`).
template <typename T>
static inline SymbolicScalar<T> salt_param(const void* base, size_t off) {{
    auto& _pb = g_ctx->findBuffer(base);
    return SymbolicScalar<T>(_pb, _pb.ptrToByteOffset(base) + off);
}}

namespace neon_kernel {{
{neon_body}
}} // namespace neon_kernel

namespace rvv_kernel {{
{rvv_body}
}} // namespace rvv_kernel

{config_pre}
// returns 0=VERIFIED, 1=COUNTEREXAMPLE, 2=TIMEOUT/UNKNOWN
int verify(size_t {bound}, size_t vlen{cfg_param}, double timeout_s) {{
    VerificationContext ctx(vlen);
    g_ctx = &ctx;
    const size_t cap = {cap_expr};   // over-allocate; exact size not needed
{decls_str}

{kernel_calls}
    // SALT_EXECUTE_ONLY=1 runs both kernels and stops, for source coverage over
    // exactly the sizes a verification run proved. Never a verdict.
    if (const char* _xo = std::getenv("SALT_EXECUTE_ONLY"); _xo && std::string(_xo) == "1") {{
        printf("{{\\"status\\":\\"EXECUTED\\",\\"batch\\":%zu}}\\n", {bound});
        return 0;
    }}
{nonvacuity}
    Term eq = {cmp_expr};
    ctx.assert_(ctx.lnot_(eq));
    if (timeout_s > 0) ctx.set_query_timeout_ms((uint64_t)(timeout_s * 1000.0));
    printf("{{\\"status\\":\\"SOLVING\\",\\"batch\\":%zu,\\"vlen\\":%zu,\\"config\\":\\"%s\\"}}\\n", {bound}, vlen, {config_str});
    CheckResult _r = ctx.check();
    if (_r == CheckResult::Unsat) {{
        printf("{{\\"status\\":\\"VERIFIED\\",\\"batch\\":%zu,\\"vlen\\":%zu,\\"config\\":\\"%s\\",\\"method\\":\\"%s\\"}}\\n", {bound}, vlen, {config_str}, SALT_METHOD(ctx));
        return 0;
    }}
    if (_r == CheckResult::Unknown) {{   // solver gave up (e.g. per-query timeout) — NOT a counterexample
        printf("{{\\"status\\":\\"TIMEOUT\\",\\"batch\\":%zu,\\"vlen\\":%zu,\\"config\\":\\"%s\\"}}\\n", {bound}, vlen, {config_str});
        return 2;
    }}
{cex_block}
}}

int main(int argc, char** argv) {{
    setvbuf(stdout, NULL, _IOLBF, 0);   // line-buffer so per-batch status streams over a pipe
    if (argc < 3) {{ fprintf(stderr, "Usage: %s <start> <end(0=unbounded)> [vlen] [time_budget_s]\\n", argv[0]); return 2; }}
    size_t start = (size_t)atoi(argv[1]);
    size_t end   = (size_t)atoi(argv[2]);                 // 0 = unbounded: sweep until the time budget
    size_t vlen  = (argc > 3) ? (size_t)atoi(argv[3]) : 256;
    double budget = (argc > 4) ? atof(argv[4]) : 0.0;     // TOTAL wall-clock budget (s); 0 = none
    if (end == 0 && budget <= 0.0) budget = 600.0;        // unbounded needs a budget — default 10 min
    auto _t0 = std::chrono::steady_clock::now();
    auto _rem = [&]() -> double {{                         // seconds left in the total budget (0 = no limit)
        if (budget <= 0.0) return 0.0;
        double e = std::chrono::duration<double>(std::chrono::steady_clock::now() - _t0).count();
        return budget - e;
    }};
{main_loop}
    return 0;
}}
"""


# ---- page-backed harness arrays ---------------------------------------------------------------------------------
# The native-access guard (core/native_guard.hpp) protects a symbolic buffer's memory during a kernel call, which
# needs the buffer's array on pages of its own: every array of an element type in a harness is a salt_vector
# (std::vector with a page allocator); vectors of anything else (pointer tables, Terms, branch decisions) are not.
_PAGE_BACKED = re.compile(r"\bstd::vector<\s*((?:const\s+)?(?:float|double|int8_t|uint8_t|int16_t|uint16_t|int32_t|"
                          r"uint32_t|int64_t|uint64_t|int|unsigned|char|signed char|unsigned char|xnn_float16|size_t))"
                          r"\s*>")


def _page_backed(harness: str) -> str:
    return _PAGE_BACKED.sub(r"salt_vector<\1>", harness)


_generate_harness_unpaged = generate_harness
_generate_shape_harness_unpaged = generate_shape_harness


def generate_harness(*args, **kwargs):            # noqa: F811 - the page-backed form of the generator above
    return _page_backed(_generate_harness_unpaged(*args, **kwargs))


def generate_shape_harness(*args, **kwargs):      # noqa: F811
    source, spec, plan = _generate_shape_harness_unpaged(*args, **kwargs)
    return _page_backed(source), spec, plan
