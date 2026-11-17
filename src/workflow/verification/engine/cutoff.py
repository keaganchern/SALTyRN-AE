"""When does a bounded elementwise proof cover every size?

Lemma (strip-mined candidates). Let the candidate's only loop be

    while (n > 0) { vl = vsetvl_e<SEW>m<LMUL>(n); BODY; p_1 += vl; ...; n -= vl; }

where BODY declares every variable it assigns (no state is carried between
iterations except the pointers p_i and n) and contains no further loop. For size
n the kernel runs BODY on consecutive chunks of vl_1, vl_2, ... elements, each vl
in [1, VLMAX], BODY's effect depending only on vl and the chunk's data. If the
reference computes y[i] = op(x[i]) (the elementwise operator's semantics, which
XNNPACK's kernel implements for every n), then proving equivalence for
n = 1..VLMAX proves BODY correct for every vl it can be given, so for every n and
every vl sequence the RVV spec allows vsetvl to return. Verifying up to VLMAX is
therefore a proof for all n.

The same induction covers an unrolled main loop with a fixed stride S,

    while (n >= S) { BODY; p_i += S; n -= S; }   then a tail for n < S,

(S = VLMAX or a multiple, from vsetvlmax): for n < S only the tail runs, and for
n >= S one iteration leaves the kernel on n - S, so proving n = 1..S covers
every n.

`strip_mined(src)` checks the syntactic premise on the candidate's source and
returns the stride; `cutoff(src, vlen)` is the size a bounded run must reach.
Reductions (rsum) carry an accumulator and are rejected.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Loop heads: strip-mined `while (n > 0)` / `for (; n > 0; ...)` (groups 2, 3),
# fixed-stride `while (n >= S)` (4, 5) or `for (; n >= S; n -= S)` (6, 7), or `do`.
_STRIDE = r"(\w+(?:\s*\*\s*\w+)?)"
_LOOP = re.compile(r"\b(while\s*\(\s*(\w+)\s*(?:>|!=)\s*0\s*\)|for\s*\(\s*;\s*(\w+)\s*(?:>|!=)\s*0\s*;[^)]*\)"
                   rf"|while\s*\(\s*(\w+)\s*>=\s*{_STRIDE}\s*\)"
                   rf"|for\s*\(\s*;\s*(\w+)\s*>=\s*{_STRIDE}\s*;[^)]*\)|do)\s*\{{")
_STORE = re.compile(r"__riscv_vs(?!etvl)\w*\s*\(")
_VLMAX = re.compile(r"\b(\w+)\s*=\s*__riscv_vsetvlmax_e(\d+)m(f?)(\d+)\s*\(\s*\)")
_DECL = re.compile(r"^\s*(?:const\s+)?[A-Za-z_]\w*(?:\s*\*+\s*|\s+)(?:(?:const|restrict)\s+)*\**\s*([A-Za-z_]\w*)\s*(?:=|;|\[)", re.M)
_ASSIGN = re.compile(r"(?<![=!<>+\-*/%&|^])\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*([-+*/%&|^]?=)(?!=)")


@dataclass
class StripMine:
    ok: bool
    reason: str
    sew: int = 0
    lmul: float = 0.0
    counter: str = ""
    factor: int = 1          # stride = factor * VLMAX (unrolled main loops)
    local: bool = False      # every vector access in the body is unit-stride at an advanced pointer

    def vlmax(self, vlen: int) -> int:
        return int(vlen // self.sew * self.lmul) * self.factor


def _body(src: str, open_brace: int) -> tuple[str, int]:
    depth, i = 1, open_brace + 1
    while i < len(src) and depth:
        depth += {"{": 1, "}": -1}.get(src[i], 0)
        i += 1
    return src[open_brace + 1:i - 1], i


def _hidden_effects(loop_body: str, src: str, declared: set) -> str | None:
    """Effects the assignment scan cannot see: a function-like macro defined
    outside the body (its assignments are not in the text), or the address of an
    outer variable handed to a call."""
    inner = set(re.findall(r"#\s*define\s+(\w+)\(", loop_body))
    for m in re.finditer(r"#\s*define\s+(\w+)\(([^)]*)\)((?:[^\n]*\\\n)*[^\n]*)", src):
        name, text = m.group(1), m.group(3)
        if name in inner or not re.search(rf"\b{name}\s*\(", loop_body):
            continue
        if re.search(r"(?<![=!<>])=(?!=)|\+\+|--|[-+*/%&|^]=", text):    # a pure macro (XNN_LIKELY) is fine
            return f"invokes macro {name} defined outside the loop"
    for name in re.findall(r"&\s*([A-Za-z_]\w*)\b(?!\s*[\[.])", loop_body):
        if name not in declared:
            return f"takes the address of outer variable {name}"
    return None


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"//[^\n]*", " ", src)


def strip_mined(src: str) -> StripMine:
    src = _strip_comments(src)
    fn = re.search(r"\btest_rvv\s*\([^)]*\)[^{]*\{", src, flags=re.S)
    if not fn:
        return StripMine(False, "no test_rvv definition")
    body, _ = _body(src, fn.end() - 1)
    loops = list(_LOOP.finditer(body))
    # The induction restarts the kernel after one iteration, so nothing before
    # the first loop may consume data (a peeled first chunk, a prologue store).
    pre = body[:loops[0].start()] if loops else body
    if re.search(r"\b\w+\s*[-+]=", pre) or _STORE.search(pre):
        return StripMine(False, "code before the loop advances pointers or stores")
    if loops and (loops[0].group(4) or loops[0].group(6)):
        return _fixed_stride(body, loops)
    if len(loops) != 1:
        return StripMine(False, f"{len(loops)} loops (the lemma needs exactly one)")
    m = loops[0]
    if m.group(1) == "do":
        return StripMine(False, "do-while loop")
    counter = m.group(2) or m.group(3)
    loop_body, _ = _body(body, m.end() - 1)
    vs = re.findall(rf"\bvl\s*=\s*__riscv_vsetvl_e(\d+)m(f?)(\d+)\s*\(\s*{re.escape(counter)}\s*\)", loop_body)
    if len(vs) != 1:
        return StripMine(False, f"loop does not set vl = vsetvl({counter}) exactly once")
    sew, frac, l = int(vs[0][0]), vs[0][1], int(vs[0][2])
    lmul = 1 / l if frac else float(l)
    declared = {d for d in _DECL.findall(loop_body)} | {"vl"}
    advanced = set(re.findall(r"\b(\w+)\s*\+=\s*vl\s*;", loop_body))
    if not re.search(rf"\b{re.escape(counter)}\s*-=\s*vl\s*;", loop_body) and \
       f"{counter} -= vl" not in m.group(0):
        return StripMine(False, f"{counter} is not decremented by vl")
    carried = sorted({name for name, _ in _ASSIGN.findall(loop_body)}
                     - declared - advanced - {counter})
    if carried:
        return StripMine(False, f"state carried across iterations: {', '.join(carried)}")
    if hidden := _hidden_effects(loop_body, src, declared):
        return StripMine(False, hidden)
    return StripMine(True, "strip-mined", sew, lmul, counter, local=_chunk_local(loop_body, advanced))


def _carried(loop_body: str, counter: str, step: str) -> list[str]:
    declared = set(_DECL.findall(loop_body)) | {"vl"}
    advanced = set(re.findall(rf"\b(\w+)\s*\+=\s*{re.escape(step)}\s*;", loop_body))
    return sorted({name for name, _ in _ASSIGN.findall(loop_body)} - declared - advanced - {counter})


def _fixed_stride(body: str, loops) -> StripMine:
    """`while (n >= S) {...}` first, then at most one strip-mined tail loop."""
    m = loops[0]
    counter, stride = m.group(4) or m.group(6), " ".join((m.group(5) or m.group(7)).split())
    loop_body, end = _body(body, m.end() - 1)
    if m.group(6):                                     # the for header holds the decrement
        loop_body += "\n" + m.group(0)
    if any(lm.start() < end for lm in loops[1:]):
        return StripMine(False, "nested loop in the main loop")
    if len(loops) > 2:
        return StripMine(False, f"{len(loops)} loops")
    if not re.search(rf"\b{re.escape(counter)}\s*-=\s*{re.escape(stride)}\s*[;)]", loop_body):
        return StripMine(False, f"main loop does not decrement {counter} by {stride}")
    if carried := _carried(loop_body, counter, stride):
        return StripMine(False, f"state carried across iterations: {', '.join(carried)}")
    if hidden := _hidden_effects(loop_body, body, set(_DECL.findall(loop_body))):
        return StripMine(False, hidden)
    if len(loops) == 2:
        tail = strip_mined_loop(body, loops[1])
        if not tail.ok or tail.counter != counter:
            return StripMine(False, f"tail loop: {tail.reason}")
    names = stride.replace("*", " ").split()
    factor, vl = 1, None
    for n in names:
        if n.isdigit():
            factor *= int(n)
            continue
        d = [v for v in _VLMAX.findall(body) if v[0] == n]
        if len(d) != 1:
            return StripMine(False, f"stride {stride} is not a constant multiple of VLMAX")
        vl = d[0]
    if not vl:
        return StripMine(False, f"stride {stride} is not a multiple of VLMAX")
    lmul = 1 / int(vl[3]) if vl[2] else float(vl[3])
    return StripMine(True, f"fixed stride {stride}, then a tail", int(vl[1]), lmul, counter, factor)


def strip_mined_loop(body: str, m) -> StripMine:
    counter = m.group(2) or m.group(3)
    if not counter:
        return StripMine(False, "not a strip-mined loop")
    loop_body, _ = _body(body, m.end() - 1)
    if not re.search(rf"\bvl\s*=\s*__riscv_vsetvl_e\d+mf?\d+\s*\(\s*{re.escape(counter)}\s*\)", loop_body):
        return StripMine(False, f"loop does not set vl = vsetvl({counter})")
    if not re.search(rf"\b{re.escape(counter)}\s*-=\s*vl\s*;", loop_body):
        return StripMine(False, f"{counter} is not decremented by vl")
    if carried := _carried(loop_body, counter, "vl"):
        return StripMine(False, f"state carried across iterations: {', '.join(carried)}")
    return StripMine(True, "strip-mined", counter=counter)


def _chunk_local(loop_body: str, advanced: set) -> bool:
    """True if each vector memory access in the body reads or writes exactly its
    own chunk, `vl` elements at an advanced pointer, and there is no scalar
    dereference of those pointers. Then the body cannot see beyond its chunk
    without the memory check's help."""
    for m in re.finditer(r"__riscv_v(l|s)(\w*?)_v_\w+\s*\(", loop_body):
        if not re.fullmatch(r"e\d+", m.group(2) or ""):
            return False                                # strided, indexed, segment, fault-first
        depth, i, args, cur = 1, m.end(), [], ""        # top-level arguments, nested calls intact
        while i < len(loop_body) and depth:
            ch = loop_body[i]
            depth += {"(": 1, ")": -1}.get(ch, 0)
            if depth == 1 and ch == ",":
                args.append(cur.strip()); cur = ""
            elif depth:
                cur += ch
            i += 1
        args.append(cur.strip())
        if args[-1] != "vl" or args[0] not in advanced:
            return False
    return not any(re.search(rf"\b{re.escape(p)}\s*\[|\*\s*{re.escape(p)}\b", loop_body) for p in advanced)


def cutoff(src: str, vlen: int) -> int | None:
    """Size a bounded run must verify up to for the lemma, or None if it does not apply."""
    s = strip_mined(src)
    return s.vlmax(vlen) if s.ok else None


def report(sources: dict, verified: dict, vlen: int) -> list[dict]:
    """Per kernel: does the lemma apply, its cutoff at `vlen`, and whether the
    verified bound reaches it (then the bounded proof covers every size)."""
    rows = []
    for name, src in sources.items():
        s = strip_mined(src)
        cut = s.vlmax(vlen) if s.ok else None
        n = verified.get(name)
        rows.append({"kernel": name, "premise": s.reason if not s.ok else "ok: " + s.reason,
                     "cutoff": cut, "verified": n, "locality": "syntactic" if s.local else "needs SALT_MEMCHECK=strict",
                     "all_sizes": bool(cut and n is not None and n >= cut)})
    return rows


def main():
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Strip-mining cutoffs vs verified bounds.")
    ap.add_argument("--vlen", type=int, required=True)
    ap.add_argument("--rvv-dir", type=Path, required=True)
    ap.add_argument("--results", type=Path,
                    help="pipeline run results dir (<kernel>.json with attempts[].max_verified_batch)")
    ap.add_argument("kernels", nargs="*")
    a = ap.parse_args()
    names = a.kernels or sorted(p.stem for p in a.rvv_dir.glob("*.c"))
    verified = {}
    for n in names:
        f = a.results / f"{n}.json" if a.results else None
        if f and f.is_file():
            att = json.loads(f.read_text()).get("attempts", [])
            if att and att[-1].get("verdict") in ("ALL_PASSED", "PARTIAL"):
                verified[n] = att[-1].get("max_verified_batch", 0)
    rows = report({n: (a.rvv_dir / f"{n}.c").read_text() for n in names if (a.rvv_dir / f"{n}.c").is_file()},
                  verified, a.vlen)
    for r in rows:
        print(f"{r['kernel']:24s} cutoff={str(r['cutoff']):>5s} verified={str(r['verified']):>6s} "
              f"{'ALL SIZES' if r['all_sizes'] else '':10s} {r['premise']}"
              + (f"; locality {r['locality']}" if r['cutoff'] else ""))


if __name__ == "__main__":
    main()


# ---- multidimensional kernels: induction on the outermost loop ------------
#
# Each multidimensional operator is parallel in some dimension D (gemm's output
# columns, a pool's output pixels, rdsum's channels, packw's groups). If the
# candidate's OUTERMOST loop runs over D with stride S (1 for `--D`, a cap C for
# `vl = vsetvl(min(D, C))`, VLMAX for `vl = vsetvl(D)`), nothing before it
# consumes data, and its body carries no state except pointers (and scalars
# advanced by an amount the fixture fixes at 0), then K(D) is one iteration on
# the first S units followed by K(D - S) at shifted pointers. A complete growth
# level N with N >= S (in D's units) proves D = 1..S for every value <= N of the
# other sizes, hence, by induction on D, every D. The reference is taken as the
# operator's specification, parallel in D.

_DO_DEC = re.compile(r"\s*while\s*\(\s*--\s*(\w+)\s*(?:!=\s*0\s*)?\)\s*;")
_DO_NZ = re.compile(r"\s*while\s*\(\s*(\w+)\s*(?:!=\s*0\s*|>\s*0\s*)?\)\s*;")
_FOR_DECL = re.compile(r"\bfor\s*\(\s*(?:const\s+)?\w+\s+(\w+)\s*=")
_PTR = re.compile(r"\b(?:const\s+)?\w+\s*\*+\s*(?:const\s+|restrict\s+)*(\w+)")


@dataclass
class OuterLoop:
    ok: bool
    reason: str
    dim: str = ""
    stride: str = ""           # "1", a constant, or "vlmax:<sew>:<lmul>"

    def units(self, vlen: int) -> int:
        if self.stride.startswith("vlmax:"):
            _, sew, lmul = self.stride.split(":")
            return int(vlen // int(sew) * float(lmul))
        return int(self.stride)


def _vlmax_stride(step: str, body: str):
    """`k * vlmax` (vlmax = vsetvlmax_e<SEW>m<LMUL>()) as a stride string, or None."""
    factor, vl = 1, None
    for n in step.replace("*", " ").split():
        if n.isdigit():
            factor *= int(n)
            continue
        d = [v for v in _VLMAX.findall(body) if v[0] == n]
        if len(d) != 1:
            return None
        vl = d[0]
    if not vl:
        return None
    lmul = 1 / int(vl[3]) if vl[2] else float(vl[3])
    return f"vlmax:{vl[1]}:{lmul * factor}"


def outer_loop(src: str, derived: dict | None = None) -> OuterLoop:
    derived = derived or {}
    src = _strip_comments(src)
    fn = re.search(r"\btest_rvv\s*\(([^)]*)\)[^{]*\{", src, flags=re.S)
    if not fn:
        return OuterLoop(False, "no test_rvv definition")
    body, _ = _body(src, fn.end() - 1)
    head = re.search(r"\b(do|while|for)\b\s*(\([^{]*\))?\s*\{", body)
    if not head:
        return OuterLoop(False, "no loop")
    pre = body[:head.start()]
    loop_body, end = _body(body, head.end() - 1)
    tail = body[end:end + 120]
    if head.group(1) == "do":
        if m := _DO_DEC.match(tail):
            dim, stride = m.group(1), "1"
        elif m := _DO_NZ.match(tail):
            dim, stride = m.group(1), None
        else:
            return OuterLoop(False, "do-loop condition not understood")
    else:
        cond = head.group(2) or ""
        m = re.match(r"\(\s*(\w+)\s*(?:>|!=)\s*0\s*\)", cond)
        fx = re.match(r"\(\s*(\w+)\s*>=\s*(\w+(?:\s*\*\s*\w+)?)\s*\)", cond)
        if head.group(1) == "while" and m:
            dim, stride = m.group(1), None
        elif head.group(1) == "while" and fx:              # unrolled main loop, then a tail
            dim, step = fx.group(1), " ".join(fx.group(2).split())
            if not re.search(rf"\b{dim}\s*-=\s*{re.escape(step)}\s*;", loop_body):
                return OuterLoop(False, f"main loop does not decrement {dim} by {step}")
            stride = step if step.isdigit() else _vlmax_stride(step, body)
            if stride is None:
                return OuterLoop(False, f"stride {step} is not a constant or a multiple of VLMAX")
        else:
            return OuterLoop(False, f"outer {head.group(1)} loop not over a size")
    # A NEON tile loop: do { ...; if (D >= T) { ...; D -= T; } else { ...; D = 0; } } while (D != 0).
    tile = re.search(rf"\bif\s*(?:XNN_LIKELY\s*)?\(\s*{dim}\s*>=\s*(\d+)\s*\)\s*\{{", loop_body) \
        if stride is None and head.group(1) == "do" else None
    if tile:
        full, after = _body(loop_body, tile.end() - 1)
        other = re.match(r"\s*else\s*\{", loop_body[after:])
        rest = _body(loop_body, after + other.end() - 1)[0] if other else ""
        t = tile.group(1)
        if re.search(rf"\b{dim}\s*-=\s*{t}\s*;", full) and re.search(rf"\b{dim}\s*=\s*0\s*;", rest) and \
           len(re.findall(rf"\b{dim}\s*(?:[-+*/]?=(?!=)|\+\+|--)", loop_body)) == 2:
            stride = t
    if stride is None:                                   # decremented by vl
        if not re.search(rf"\b{dim}\s*-=\s*vl\s*;", loop_body):
            return OuterLoop(False, f"{dim} is not decremented by vl")
        v = re.search(r"\bvl\s*=\s*__riscv_vsetvl_e(\d+)m(f?)(\d+)\s*\(([^;]*)\)\s*;", loop_body)
        plain = re.search(r"\bvl\s*=\s*\(?([^;?]+\?[^;]+?)\)?\s*;", loop_body)
        if not v and plain:                             # vl = nc > 8 ? 8 : nc, no vsetvl
            arg = " ".join(plain.group(1).split())
            cap = re.fullmatch(rf"{dim}\s*>\s*(\d+)\s*\?\s*\1\s*:\s*{dim}|{dim}\s*<\s*(\d+)\s*\?\s*{dim}\s*:\s*\2", arg)
            if not cap:
                return OuterLoop(False, f"vl = {arg} is not min({dim}, C)")
            v = None
        elif not v:
            return OuterLoop(False, "no vl = vsetvl(...)")
        else:
            arg = " ".join(v.group(4).split())
        cap = re.fullmatch(rf"{dim}\s*>\s*(\d+)\s*\?\s*\1\s*:\s*{dim}|{dim}\s*<\s*(\d+)\s*\?\s*{dim}\s*:\s*\2", arg)
        lmul = (f"1/{v.group(3)}" if v.group(2) else v.group(3)) if v else "1"
        if cap:
            stride = cap.group(1) or cap.group(2)
        elif arg == dim:
            stride = f"vlmax:{v.group(1)}:{eval(lmul)}"
        else:
            return OuterLoop(False, f"vsetvl({arg}) is not {dim} or min({dim}, C)")
    # The loop may run over a local copy of the size (`size_t bw = block_width;`).
    copy = re.search(rf"\b(?:size_t|int|unsigned|uint32_t)\s+{dim}\s*=\s*(\w+)\s*;", pre)
    rest = pre.replace(copy.group(0), "") if copy else pre
    if _STORE.search(rest) or re.search(rf"\b{dim}\s*(?:[-+*/]?=(?!=)|\+\+|--)", rest):
        return OuterLoop(False, "code before the loop stores or changes the size")
    size = copy.group(1) if copy else dim
    pointers = set(_PTR.findall(fn.group(1) + "\n" + pre))
    declared = set(_DECL.findall(loop_body)) | set(_FOR_DECL.findall(loop_body)) | {"vl"}
    for name, op in _ASSIGN.findall(loop_body):
        if name in declared or name in pointers or name == dim:
            continue
        inc = re.search(rf"\b{name}\s*\+=\s*(\w+)\s*;", loop_body)
        if op == "+=" and inc and derived.get(inc.group(1)) == "0":
            continue                                     # advanced by a fixture constant 0
        return OuterLoop(False, f"state carried across iterations: {name}")
    if hidden := _hidden_effects(loop_body, src, declared | pointers):
        return OuterLoop(False, hidden)
    return OuterLoop(True, "ok", size, stride)


# ---- multidimensional kernels: lane decomposition of a channel loop --------
#
# A pool's channels are independent, and a lane-wise candidate shows it in its
# code. Suppose every loop over a channel count D (a local copy `c = D`) is
#
#   candidate:  while (c > 0) { vl = vsetvl(c); BODY; p += vl; c -= vl; }
#               BODY stateless, every vector access a unit-stride vle/vse of vl
#               elements at an advanced pointer, and every vector operation
#               lane-wise (lane i of a result depends on lane i of its operands);
#   reference:  for (; c >= T; c -= T) { BODY; p += T; } then any tail,
#               BODY stateless, its vector accesses full T-lane vld1q/vst1q at
#               advanced pointers,
#
# and D is used nowhere else except by strides that only advance pointers after
# the last channel loop (dead for one output pixel). Then the candidate computes
# each channel by the same lane function whatever the chunking, so
# C(D) = C(T) on channels [0, T) next to C(D - T) on [T, D); the reference splits
# the same way at its first iteration (the parts touch disjoint memory, so the
# passes interleave freely). By induction on D, verifying D = 1..T proves every
# D, with nothing assumed about either kernel. With one output pixel; the outer
# pixel induction extends it to every pixel count.

_LANEWISE = re.compile(
    r"v(?:le|se)\d+|vsetvl|vundefined|vreinterpret|vmv_v|vfmv_v|vmerge|vfmerge|"
    r"v(?:w|n)?(?:add|sub|rsub|mul|mulh|mulhu|mulhsu|div|rem|macc|nmsac|madd|nmsub)u?(?:su)?|"
    r"vand|vor|vxor|vnot|vneg|vsll|vsrl|vsra|vnsrl|vnsra|vnclipu?|vmin|vminu|vmax|vmaxu|"
    r"vs(?:add|sub)u?|vsmul|vssrl|vssra|va(?:add|sub)u?|vadc|vsbc|vmadc|vmsbc|vsext|vzext|"
    r"vms(?:eq|ne|lt|le|gt|ge)u?|vmf(?:eq|ne|lt|le|gt|ge)|vm(?:and|nand|andn|or|nor|orn|xor|xnor|not|mv)|"
    r"vf(?:w|n)?(?:add|sub|rsub|mul|div|rdiv|macc|nmacc|msac|nmsac|madd|nmadd|msub|nmsub)|"
    r"vf(?:min|max|sgnj|sgnjn|sgnjx|abs|neg|sqrt|rec7|rsqrt7|class)|vf(?:w|n)?cvt")
_NEON_ACCESS = re.compile(r"\b(vld1q|vst1q)_([fsu])(\d+)\s*\(\s*(\w+)")


@dataclass
class LaneSplit:
    ok: bool
    reason: str
    dim: str = ""
    base: int = 0              # D = 1..base proves every D


def _fn_body(src: str, name: str):
    fn = re.search(rf"\b{name}\s*\(([^)]*)\)[^{{]*\{{", src, flags=re.S)
    return (_body(src, fn.end() - 1)[0], fn.group(1)) if fn else (None, None)


def _channel_loops(body: str, dim: str):
    """(counter, head match, loop body, end) for every loop over a local copy of dim."""
    out = []
    for cp in re.finditer(rf"\b(?:size_t|int|unsigned|uint32_t)\s+(\w+)\s*=\s*{dim}\s*;", body):
        c = cp.group(1)
        head = re.compile(rf"\b(while\s*\(\s*{c}\s*(?:>|!=)\s*0\s*\)|for\s*\(\s*;\s*{c}\s*(?:>|!=|>=)[^;]*;[^)]*\))\s*\{{")
        m = head.search(body, cp.end())
        if not m:
            return None
        lb, end = _body(body, m.end() - 1)
        out.append((c, m, lb, end))
    return out


def _invariant_assignments(loop_body: str, tail: str, carried: list[str], advanced: set) -> list[str]:
    """The carried names whose value is the same after every iteration: each is
    written before it is read, from an expression over loop-invariant names
    (body-local definitions inlined, no loads), and a tail that writes it writes
    the same expression. NEON argmaxpool's vidx8 = vidx0 + 8 is one."""
    def defs(text):
        return dict(re.findall(r"(?:const\s+)?\w+\s+(\w+)\s*=\s*([^;]+);", text))

    def expand(expr, d):
        for _ in range(32):
            new = re.sub(r"\b(\w+)\b", lambda m: f"({d[m.group(1)]})" if m.group(1) in d else m.group(1), expr)
            if new == expr:
                break
            expr = new
        return " ".join(expr.split())

    written = {n for n, _ in _ASSIGN.findall(loop_body)}
    left = []
    for x in carried:
        first = re.search(rf"\b{x}\b", loop_body)
        rhs = re.findall(rf"(?<![\w.])\b{x}\s*=(?!=)\s*([^;]+);", loop_body)
        if not first or len(rhs) != 1 or not re.match(rf"{x}\s*=(?!=)", loop_body[first.start():]):
            left.append(x)
            continue
        e = expand(rhs[0], defs(loop_body))
        names = set(re.findall(r"\b[A-Za-z_]\w*\b", e))
        t_rhs = re.findall(rf"(?<![\w.])\b{x}\s*=(?!=)\s*([^;]+);", tail)
        if (re.search(r"\bv(?:ld|st)\d", e) or names & (written | advanced) or
                any(expand(r, defs(tail)) != e for r in t_rhs)):
            left.append(x)
    return left


def _lanewise(loop_body: str) -> str | None:
    for name in re.findall(r"__riscv_(\w+)\s*\(", loop_body):
        if name.startswith(("vmv_x", "vmv_s", "vfmv_f", "vfmv_s")) or not _LANEWISE.match(name):
            return f"{name} is not lane-wise"
        # A reinterpret between element widths regroups lanes.
        rw = re.match(r"vreinterpret_v_[a-z]+?(\d+)m\w*?_[a-z]+?(\d+)m", name)
        if name.startswith("vreinterpret") and (not rw or rw.group(1) != rw.group(2)):
            return f"{name} changes the element width"
    return None


def _dead_after(body: str, uses: list[str], last_end: int) -> str | None:
    """Every statement naming one of `uses` advances a pointer after the last channel loop."""
    for u in uses:
        for m in re.finditer(rf"[^;{{}}]*\b{u}\b[^;]*;", body):
            stmt = " ".join(m.group(0).split())
            ok = m.start() >= last_end and re.fullmatch(
                rf"(\w+)\s*(?:\+=\s*{u}|=\s*\([^=]*\)\s*\(\s*\([^)]*\)\s*\1\s*\+\s*{u}\s*\))\s*;", stmt)
            if not ok:
                return f"{u} is used by `{stmt}`"
    return None


def lane_split(rvv_src: str, neon_src: str, dim: str, derived: dict) -> LaneSplit:
    rvv, _ = _fn_body(_strip_comments(rvv_src), "test_rvv")
    ref, _ = _fn_body(_strip_comments(neon_src), "test_neon")
    if rvv is None or ref is None:
        return LaneSplit(False, "kernel definitions not found")
    dependent = [n for n, e in derived.items() if re.search(rf"\b{dim}\b", e)]
    T = None
    for side, body in (("candidate", rvv), ("reference", ref)):
        loops = _channel_loops(body, dim)
        if not loops:
            return LaneSplit(False, f"{side}: no loop over a copy of {dim}")
        for c, m, lb, end in loops:
            if side == "candidate":
                if not re.match(rf"while\s*\(\s*{c}\s*>\s*0", m.group(1)):
                    return LaneSplit(False, f"candidate: loop over {c} is not strip-mined")
                if not re.search(rf"\bvl\s*=\s*__riscv_vsetvl_e\d+mf?\d+\s*\(\s*{c}\s*\)\s*;", lb) or \
                   not re.search(rf"\b{c}\s*-=\s*vl\s*;", lb) or len(re.findall(r"__riscv_vsetvl", lb)) != 1:
                    return LaneSplit(False, f"candidate: loop over {c} does not take vl = vsetvl({c}) once")
                if carried := _carried(lb, c, "vl"):
                    return LaneSplit(False, f"candidate: state carried across iterations: {', '.join(carried)}")
                advanced = set(re.findall(r"\b(\w+)\s*\+=\s*vl\s*;", lb))
                if not _chunk_local(lb, advanced) or any(
                        re.search(rf"(?<![\w.>]){re.escape(a)}\s*=(?!=)", lb) for a in advanced):
                    return LaneSplit(False, "candidate: a vector access is not its own chunk")
                if why := _lanewise(lb):
                    return LaneSplit(False, f"candidate: {why}")
            else:
                fx = re.match(rf"for\s*\(\s*;\s*{c}\s*>=\s*(\d+)\s*;\s*{c}\s*-=\s*\1\s*\)", m.group(1))
                if not fx:
                    return LaneSplit(False, f"reference: loop over {c} is not `for (; {c} >= T; {c} -= T)`")
                t = int(fx.group(1))
                if T not in (None, t):
                    return LaneSplit(False, "reference: channel loops differ in stride")
                T = t
                advanced = set(re.findall(rf"\b(\w+)\s*\+=\s*{t}\s*;", lb))
                tail = re.match(rf"\s*if\s*\(\s*{c}\s*!=\s*0\s*\)\s*\{{", body[end:])
                tail = _body(body, end + tail.end() - 1)[0] if tail else ""
                if carried := _invariant_assignments(lb, tail, _carried(lb, c, str(t)), advanced):
                    return LaneSplit(False, f"reference: state carried across iterations: {', '.join(carried)}")
                for acc in _NEON_ACCESS.finditer(lb):
                    if 128 // int(acc.group(3)) != t or acc.group(4) not in advanced:
                        return LaneSplit(False, f"reference: {acc.group(0)}...) is not a full chunk access")
                if re.search(r"\bv(?:ld|st)[1-4](?!q_[fsu]\d+\s*\()", lb) or \
                   any(re.search(rf"\b{re.escape(p)}\s*\[|\*\s*{re.escape(p)}\b", lb) for p in advanced):
                    return LaneSplit(False, "reference: a memory access other than a full chunk")
            if hidden := _hidden_effects(lb, body, set(_DECL.findall(lb))):
                return LaneSplit(False, f"{side}: {hidden}")
        # dim itself only feeds the copies (and asserts); dependent strides only advance pointers after.
        rest = re.sub(r"\bassert\s*\([^;]*\)\s*;", "", body)
        rest = re.sub(rf"\b(?:size_t|int|unsigned|uint32_t)\s+\w+\s*=\s*{dim}\s*;", "", rest)
        if re.search(rf"\b{dim}\b", rest):
            return LaneSplit(False, f"{side}: {dim} is used outside the channel loops")
        if why := _dead_after(body, dependent, max(end for *_, end in loops)):
            return LaneSplit(False, f"{side}: {why}")
    return LaneSplit(True, "ok", dim, T)
