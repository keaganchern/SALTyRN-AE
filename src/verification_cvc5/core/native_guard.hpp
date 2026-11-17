#pragma once
// Native-access guard.
//
// A kernel runs natively: only the intrinsic models and the generator's rewritten accesses (salt_scalar,
// salt_deref, salt_param) reach the symbolic model. Three native operations would bypass it silently, so each
// kernel call is checked for them, and a call that performs one is refused (a refusal can make a proof fail,
// never pass):
//  - a store into a registered buffer's backing memory: the model never reads that memory, so the stored value
//    would be invisible (an output element both kernels wrote that way would even compare equal, untested).
//    native_guard_begin/_end snapshot every registered buffer around the call and refuse any changed byte,
//    except a side channel the harness reads instead (rsum's `*out += sum`, see below).
//  - float arithmetic on a NaN-payload sentinel (a symbolic input element read natively, or a float extracted
//    from a vector): the sentinel is a signaling NaN, which every x86 float operation quiets and a negation
//    flips the sign of; the decoder (mk_fp32_from_float) refuses a sentinel changed either way. The bitwise
//    operations that keep a positive sentinel (fabsf, a bit-cast) and the C tests on one are refused by the
//    generator's static check (emit.py, _c_float_tests).
//  - native use of an integer extracted from a vector (rsum's `vaddvq_s32`, `__riscv_vmv_x_s`): the extract
//    returns kExtractMarker and stashes the symbolic value; the only native use the harness models is adding it
//    to an output element (`*out += s`), so after the call that element must hold exactly its value before plus
//    the marker (native_guard_sum). A pure reduction must also extract exactly once (the harness's count check),
//    so a sum handed back to an intrinsic and extracted again is refused too.
//  - a native read of a symbolic buffer that has no sentinel (an integer input read through a type-punned pointer,
//    `*(const uint64_t*)a0`, which the generator's rewrites do not see, or an output read back): the read would see
//    the backing array's zeros. The harness's arrays live on pages of their own (salt_vector), and during a kernel
//    call NativeProtect makes the pages of every such buffer inaccessible: a native access faults, and the handler
//    reports NATIVE_ACCESS and ends the shape's process. The model itself never touches those pages (it reads the
//    symbolic bytes; only concrete buffers and sentinel-bound float buffers are read natively, and stay readable).
#include "symbolic_buffer.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <tuple>
#include <unistd.h>
#include <vector>

namespace salt_cvc5 {

struct NativeAccess : std::runtime_error { using std::runtime_error::runtime_error; };
[[noreturn]] inline void salt_tail_read(size_t i, size_t n) {
    throw NativeAccess("a read of vector element " + std::to_string(i) + " of a vector produced with vl " +
                       std::to_string(n) + ": a tail element, whose value RVV leaves unspecified (tail-agnostic)");
}

// ---- page-backed harness arrays ------------------------------------------------------------------------------
inline std::map<uintptr_t, size_t>& salt_page_regions() { static std::map<uintptr_t, size_t> r; return r; }
inline size_t salt_page_size() { static size_t p = static_cast<size_t>(sysconf(_SC_PAGESIZE)); return p; }

template<class T>
struct SaltPageAllocator {
    using value_type = T;
    SaltPageAllocator() = default;
    template<class U> SaltPageAllocator(const SaltPageAllocator<U>&) {}
    T* allocate(size_t n) {
        size_t pg = salt_page_size(), len = (std::max<size_t>(n * sizeof(T), 1) + pg - 1) / pg * pg;
        void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) throw std::bad_alloc();
        salt_page_regions()[reinterpret_cast<uintptr_t>(p)] = len;
        return static_cast<T*>(p);
    }
    void deallocate(T* p, size_t) {
        auto& r = salt_page_regions();
        auto it = r.find(reinterpret_cast<uintptr_t>(p));
        if (it != r.end()) { munmap(p, it->second); r.erase(it); }
    }
    template<class U> bool operator==(const SaltPageAllocator<U>&) const { return true; }
    template<class U> bool operator!=(const SaltPageAllocator<U>&) const { return false; }
};
template<class T> using salt_vector = std::vector<T, SaltPageAllocator<T>>;

// ---- protection during a kernel call --------------------------------------------------------------------------
inline char salt_fault_line[640];                  // the NATIVE_ACCESS status line, formatted before each call
inline constexpr size_t kMaxProtected = 64;
inline volatile sig_atomic_t salt_protected_n = 0;
inline uintptr_t salt_protected_lo[kMaxProtected], salt_protected_hi[kMaxProtected];

extern "C" inline void salt_fault_handler(int sig, siginfo_t* si, void*) {
    uintptr_t a = reinterpret_cast<uintptr_t>(si->si_addr);
    for (sig_atomic_t i = 0; i < salt_protected_n; i++) {
        if (a >= salt_protected_lo[i] && a < salt_protected_hi[i]) {
            static const char msg[] = "NATIVE_ACCESS: a native C access to a symbolic buffer's memory "
                                      "(a read the symbolic model cannot see)\n";
            ssize_t w1 = write(1, salt_fault_line, strnlen(salt_fault_line, sizeof salt_fault_line));
            ssize_t w2 = write(2, msg, sizeof msg - 1);
            (void)w1; (void)w2;
            _exit(1);
        }
    }
    signal(sig, SIG_DFL);                           // not ours: the default action when the access repeats
}

inline void salt_install_fault_handler() {
    static bool done = false;
    if (done) return;
    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = salt_fault_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
    done = true;
}

// Makes the pages of every symbolic buffer without sentinels inaccessible for its lifetime (or until release()),
// except the buffers named in `exempt` (a side channel the kernel writes natively).
struct NativeProtect {
    std::vector<std::pair<uintptr_t, size_t>> regions;
    NativeProtect(VerificationContext& ctx, const std::vector<std::string>& exempt) {
        salt_install_fault_handler();
        auto& pages = salt_page_regions();
        for (auto* b : ctx.registered_buffers) {
            if (!b || b->isFallback() || b->isConcrete() || b->isScalarRef() || !b->baseAddr() || !b->numBytes())
                continue;
            if (std::find(exempt.begin(), exempt.end(), b->name()) != exempt.end()) continue;
            auto it = pages.upper_bound(b->baseAddr());
            if (it == pages.begin()) continue;
            --it;                                   // the page allocation holding the buffer, if any
            if (b->baseAddr() + b->numBytes() > it->first + it->second) continue;
            std::pair<uintptr_t, size_t> reg(it->first, it->second);
            if (std::find(regions.begin(), regions.end(), reg) != regions.end()) continue;
            if (regions.size() >= kMaxProtected) break;
            regions.push_back(reg);
        }
        for (size_t i = 0; i < regions.size(); i++) {
            salt_protected_lo[i] = regions[i].first;
            salt_protected_hi[i] = regions[i].first + regions[i].second;
        }
        salt_protected_n = static_cast<sig_atomic_t>(regions.size());
        for (auto& [p, n] : regions) mprotect(reinterpret_cast<void*>(p), n, PROT_NONE);
    }
    void release() {
        for (auto& [p, n] : regions) mprotect(reinterpret_cast<void*>(p), n, PROT_READ | PROT_WRITE);
        regions.clear();
        salt_protected_n = 0;
    }
    ~NativeProtect() { release(); }
    NativeProtect(const NativeProtect&) = delete;
    NativeProtect& operator=(const NativeProtect&) = delete;
};

// What an integer extract returns natively (any other value would do; this one is unlikely as a kernel constant).
inline constexpr uint32_t kExtractMarker = 0x2A5A17C3u;

struct NativeSnapshot {
    std::vector<std::pair<const SymbolicBuffer*, std::vector<uint8_t>>> bufs;
};

inline NativeSnapshot native_guard_begin(VerificationContext& ctx) {
    NativeSnapshot s;
    for (auto* b : ctx.registered_buffers) {
        if (!b || b->isFallback() || !b->baseAddr() || !b->numBytes()) continue;
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b->baseAddr());
        s.bufs.emplace_back(b, std::vector<uint8_t>(p, p + b->numBytes()));
    }
    return s;
}

// A kernel calling an intrinsic the library has no model for: refused (a stub returning a constant would make the
// model compute with that constant, whatever the kernel's real value is).
[[noreturn]] inline void salt_unmodeled(const char* name) {
    throw NativeAccess(std::string(name) + ": this intrinsic has no symbolic model");
}

// The term a native float stands for. A sentinel (exponent all ones, marker bit 21) names a symbolic element (a
// buffer lane, or an extracted vector lane for buffer id 0) and decodes to it; one native code changed (quieted by
// arithmetic, sign set by a negation) is refused; any other float is that constant.
inline Term salt_decode_fp32(TermManager& tm, float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    Op op = tm.mkOp(Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV, {8, 24});
    if ((bits & 0x7FA00000u) != 0x7FA00000u) return tm.mkTerm(op, {tm.mkBitVector(32, static_cast<uint64_t>(bits))});
    char hex[16];
    std::snprintf(hex, sizeof hex, "0x%08x", bits);
    if ((bits & 0xFFE00000u) != kScalarRefSentinel)
        throw NativeAccess(std::string("native float arithmetic on a symbolic value (its NaN-payload sentinel became ") +
                           hex + "): the symbolic model cannot follow it");
    if (!g_ctx) throw NativeAccess(std::string("sentinel ") + hex + " outside a verification context");
    uint8_t id = static_cast<uint8_t>((bits >> 16) & 0x1Fu);
    uint32_t lane = bits & 0xFFFFu;
    if (id == 0) {
        if (lane >= g_ctx->extracted_fp_.size()) throw NativeAccess(std::string("sentinel ") + hex + " names no extracted lane");
        return tm.mkTerm(op, {g_ctx->extracted_fp_[lane]});
    }
    SymbolicBuffer* buf = g_ctx->findBufferById(id);
    if (!buf) throw NativeAccess(std::string("sentinel ") + hex + " names no unique scalar-ref buffer");
    return tm.mkTerm(op, {buf->loadScalar(static_cast<size_t>(lane) * 4, 32)});
}

// A float extracted from a vector lane (`bv32`, the lane's IEEE bits): a sentinel for an extracted_fp_ entry.
inline float salt_extract_fp32(const Term& bv32) {
    auto& v = g_ctx->extracted_fp_;
    if (v.size() >= 0x10000u) throw NativeAccess("more extracted floats than a sentinel can address");
    uint32_t bits = kScalarRefSentinel | static_cast<uint32_t>(v.size());
    v.push_back(bv32);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// The bytes a kernel may write natively: (buffer name, first byte, length).
using NativeAllow = std::vector<std::tuple<std::string, size_t, size_t>>;

inline void native_guard_end(VerificationContext& ctx, const NativeSnapshot& s, const char* side,
                             const NativeAllow& allow = {}) {
    for (auto& [b, before] : s.bufs) {
        // A buffer unregistered during the call (none is, today) no longer has memory to compare.
        if (std::find(ctx.registered_buffers.begin(), ctx.registered_buffers.end(), b) == ctx.registered_buffers.end())
            continue;
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b->baseAddr());
        for (size_t i = 0; i < before.size(); i++) {
            if (p[i] == before[i]) continue;
            bool allowed = false;
            for (auto& [n, off, len] : allow)
                if (n == b->name() && i >= off && i < off + len) allowed = true;
            if (allowed) continue;
            throw NativeAccess(std::string(side) + " stored natively into " + b->name() + " at byte " +
                               std::to_string(i) + ": a C store the symbolic model cannot see");
        }
    }
}

// The element at `off` of `buf` (4 bytes) after a reduction's native store of its extracted sum: 1 if it holds its
// value before plus the marker (`*out += s`: the model's result is the prior value plus the sum), 2 if it holds the
// marker (`*out = s`, allowed with `allow_set`: the result is the sum, which then includes the prior value if the
// kernel read it, symbolically). Anything else is refused.
inline int native_guard_sum(const NativeSnapshot& s, const SymbolicBuffer& buf, size_t off, const char* side,
                            bool allow_set = false) {
    for (auto& [b, before] : s.bufs) {
        if (b != &buf) continue;
        if (off + 4 > before.size()) break;
        uint32_t was, now;
        std::memcpy(&was, before.data() + off, 4);
        std::memcpy(&now, reinterpret_cast<const uint8_t*>(buf.baseAddr()) + off, 4);
        if (now == was + kExtractMarker) return 1;
        if (allow_set && now == kExtractMarker && was != 0) return 2;
        throw NativeAccess(std::string(side) + " used an extracted sum natively other than `*out += sum`" +
                           (allow_set ? " or `*out = sum`" : "") + " (" + buf.name() + " at byte " +
                           std::to_string(off) + ")");
    }
    throw NativeAccess(std::string(side) + ": the reduction output " + buf.name() + " was not checked");
}

} // namespace salt_cvc5
