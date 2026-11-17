#pragma once
// Memory footprint comparison: the candidate may touch only what the reference
// touches. Run each kernel between footprint_begin()/footprint_end(), then
// compare the two snapshots with memory_violations().
//
// Rules, per logical buffer (a shared input, or an output's _neon/_rvv pair).
// Errors:
//   reads before the buffer's start that the reference does not also make;
//   writes outside the reference's write range, or into an input. An output
//   with a declared layout (a packed buffer whose whole extent the caller
//   allocates, holes included) may be written anywhere inside that extent: the
//   holes are don't-care, and NEON packw itself zero-fills one when bias is NULL.
// (Reads past a buffer's capacity fail earlier, as BufferOOB overruns.)
// Warnings:
//   reads past the reference's read/write range plus `slack` bytes
//   (XNN_OOB_READS; XNNPACK pads inputs by XNN_EXTRA_BYTES = 16) that stay
//   inside the buffer. The reference need not read all of a legal buffer (rdsum
//   reuses the first 16 bytes of its zero row), so these are reported only.
//   A read-only buffer with a declared contract extent (`read_layout`, e.g. a
//   zero row of `channels` bytes) may be read anywhere inside that extent plus
//   `slack`, whatever the reference read of it.
// A buffer missing from `pairs` is compared with the reference's buffer of the
// same name. Static memory (findBuffer's concrete fallback windows) is exempt.
// Native scalar reads (`float x = *p`) bypass SymbolicBuffer and are not seen.

#include "symbolic_buffer.hpp"

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace salt_cvc5 {

using Footprints = std::map<std::string, Footprint>;

inline void footprint_begin(VerificationContext& ctx) {
    for (auto* b : ctx.registered_buffers) b->footprintReset();
    SymbolicBuffer::setTracking(true);
}

inline Footprints footprint_end(VerificationContext& ctx) {
    SymbolicBuffer::setTracking(false);
    Footprints out;
    for (auto* b : ctx.registered_buffers) out[b->name()] = b->footprint();
    return out;
}

struct MemoryReport { std::string errors = "[]", warnings = "[]"; };

// Credit an access the harness performs on a kernel's behalf (a native scalar
// store the symbolic buffers cannot see) to that kernel's footprint.
inline void footprint_note(Footprints& fp, const std::string& name, long long lo, long long hi,
                           bool rd, bool wr) {
    Footprint& f = fp[name];
    if (rd) { f.rlo = std::min(f.rlo, lo); f.rhi = std::max(f.rhi, hi); }
    if (wr) { f.wlo = std::min(f.wlo, lo); f.whi = std::max(f.whi, hi); }
}

// `pairs` maps each candidate buffer name to the reference buffer it must stay
// within; `inputs` are buffers neither kernel may write; `own_layout` are
// per-side packings of the same logical data (dual weights), bounded only by
// their capacity; `layout` gives a candidate output's caller-owned extent in
// bytes. Returns a JSON array of violations, "[]" when there are none.
inline MemoryReport memory_violations(const Footprints& ref, const Footprints& cand,
                                      const std::map<std::string, std::string>& pairs,
                                      const std::set<std::string>& inputs,
                                      const std::set<std::string>& own_layout = {},
                                      const std::map<std::string, long long>& layout = {},
                                      const std::map<std::string, long long>& read_layout = {},
                                      long long slack = 16) {
    std::string err = "[", warn = "[";
    auto add = [](std::string& out, const std::string& buf, const char* what, long long lo,
                  long long hi, long long alo, long long ahi) {
        if (out.size() > 1) out += ",";
        out += "{\"buffer\":\"" + buf + "\",\"access\":\"" + what + "\",\"lo\":" + std::to_string(lo) +
               ",\"hi\":" + std::to_string(hi) + ",\"allowed_lo\":" + std::to_string(alo) +
               ",\"allowed_hi\":" + std::to_string(ahi) + "}";
    };
    for (const auto& [name, c] : cand) {
        if ((!c.read() && !c.wrote()) || own_layout.count(name)) continue;
        // findBuffer's fallback windows are static memory (a kernel's own constant
        // tables); an overrun past a registered buffer throws BufferOOB instead.
        if (name.rfind("concrete@", 0) == 0) continue;
        // A local array of the kernel's own (salt SaltScratch): its stack memory, not the caller's; an access
        // past the array throws BufferOOB like any registered buffer.
        if (name.rfind("scratch@", 0) == 0) continue;
        auto pi = pairs.find(name);
        auto ri = ref.find(pi != pairs.end() ? pi->second : name);
        Footprint r = ri != ref.end() ? ri->second : Footprint{};   // untouched by the reference
        if (c.read()) {
            long long alo = r.read() ? std::min(0LL, r.rlo) : 0;
            long long top = std::max(r.read() ? r.rhi : -1, r.wrote() ? r.whi : -1);
            long long ahi = (r.read() || r.wrote()) ? top + slack : -1;
            if (auto li = read_layout.find(name); li != read_layout.end())   // the caller-owned extent
                ahi = std::max(ahi, li->second - 1 + slack);
            if (c.rlo < alo) add(err, name, "read-before-start", c.rlo, c.rhi, alo, ahi);
            else if (c.rhi > ahi) add(warn, name, "read", c.rlo, c.rhi, alo, ahi);
        }
        if (c.wrote()) {
            if (inputs.count(name)) add(err, name, "write-to-input", c.wlo, c.whi, 0, -1);
            else if (auto li = layout.find(name); li != layout.end() && c.wlo >= 0 && c.whi < li->second)
                ;                                       // inside the caller-owned layout
            else if (!r.wrote() || c.wlo < r.wlo || c.whi > r.whi)
                add(err, name, "write", c.wlo, c.whi, r.wrote() ? r.wlo : 0, r.wrote() ? r.whi : -1);
        }
    }
    return {err + "]", warn + "]"};
}

} // namespace salt_cvc5
