#pragma once
// Byte-addressed symbolic buffer with typed load/store views.
// cvc5 port — mirrors src/verification/core/symbolic_buffer.hpp.

#include "context.hpp"
#include "symbolic_vector.hpp"

#include <cvc5/cvc5.h>

#include <cstddef>
#include <climits>
#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace salt_cvc5 {

// A scalar-ref sentinel: a signaling NaN, marker bit 21 set, then a 5-bit buffer id and a 16-bit lane
// (bindScalarRef). Buffer id 0 marks a float extracted from a vector (VerificationContext::extract_fp).
inline constexpr uint32_t kScalarRefSentinel = 0x7FA00000u;

// An access outside a buffer's capacity. The harness tells a fixture that is too
// small (the reference overran) from a candidate overrun.
struct BufferOOB : std::runtime_error {
    std::string buffer, op;
    long long offset;
    BufferOOB(const std::string& b, const std::string& o, long long off, const std::string& msg)
        : std::runtime_error(msg), buffer(b), op(o), offset(off) {}
};

// Byte offsets a kernel touched in one buffer, inclusive; negative offsets are
// before the buffer start. Recorded only while tracking is on (kernel calls), so
// the harness's own snapshots and comparisons do not count.
struct Footprint {
    long long rlo = LLONG_MAX, rhi = LLONG_MIN, wlo = LLONG_MAX, whi = LLONG_MIN;
    bool read() const { return rlo <= rhi; }
    bool wrote() const { return wlo <= whi; }
};

class SymbolicBuffer {
    mutable std::vector<Term> bytes_;  // lazily minted on first touch; empty when concrete_ptr_
    mutable std::vector<Term> init_;   // each byte's minted (initial) term, for a counterexample's inputs (in-out buffers)
    mutable std::map<size_t, Term> pre_bytes_;  // bytes just before the buffer, by distance (see getByte)
    size_t num_bytes_;
    uintptr_t base_addr_ = 0;
    TermManager* tm_;
    std::string name_;
    const uint8_t* concrete_ptr_ = nullptr;  // non-null → read-only concrete
    uint8_t buffer_id_ = 0;         // unique [1..31] for scalar-ref sentinel decoding
    bool scalar_ref_ = false;       // bound with bindScalarRef: its backing memory holds sentinels
    static inline thread_local uint8_t s_next_id_ = 1;
    mutable Footprint fp_;
    static inline thread_local bool s_tracking_ = false;
    bool fallback_ = false;         // findBuffer's window onto unregistered (static) memory
    void noteRead(long long off, size_t n) const {
        if (!s_tracking_) return;
        fp_.rlo = std::min(fp_.rlo, off);
        fp_.rhi = std::max(fp_.rhi, off + (long long)n - 1);
    }
    void noteWrite(long long off, size_t n) {
        if (!s_tracking_) return;
        fp_.wlo = std::min(fp_.wlo, off);
        fp_.whi = std::max(fp_.whi, off + (long long)n - 1);
    }
public:
    static void setTracking(bool on) { s_tracking_ = on; }
    void markFallback() { fallback_ = true; }
    bool isFallback() const { return fallback_; }
    void footprintReset() { fp_ = Footprint{}; }
    const Footprint& footprint() const { return fp_; }
    // How far before a symbolic buffer a load may start (e.g. a masked left-padding
    // pixel at row - 12 bytes) and still resolve to that buffer.
    static constexpr size_t kPreBytes = 256;

    const std::string& name() const { return name_; }
    uint8_t bufferId() const { return buffer_id_; }
    bool isScalarRef() const { return scalar_ref_; }
    // The byte's value before any kernel stored to it (null if it was never read before being written).
    Term initialByte(size_t offset) const { return offset < init_.size() ? init_[offset] : Term(); }
    SymbolicBuffer(TermManager& tm, size_t num_bytes, const std::string& name)
        : num_bytes_(num_bytes), tm_(&tm), name_(name) {
        buffer_id_ = s_next_id_;
        s_next_id_ = (s_next_id_ >= 31) ? 1 : (s_next_id_ + 1);
        bytes_.resize(num_bytes);  // null Terms; each byte minted on first getByte()
        init_.resize(num_bytes);
    }

    // Concrete read-only ctor: skips per-byte symbolic constants entirely;
    // loads synthesize a single BV constant from the bound bytes.  Stores
    // throw — kernels must not write to a buffer registered as concrete.
    SymbolicBuffer(TermManager& tm, size_t num_bytes, const std::string& name,
                   const void* concrete_data)
        : num_bytes_(num_bytes),
          base_addr_(reinterpret_cast<uintptr_t>(concrete_data)),
          tm_(&tm), name_(name),
          concrete_ptr_(reinterpret_cast<const uint8_t*>(concrete_data)) {}

    void bind(const void* ptr) { base_addr_ = reinterpret_cast<uintptr_t>(ptr); }

    // Bind buffer and write NaN-payload sentinels at each fp32 lane so that
    // scalar reads (`float x = *p++;`) carry buffer_id + offset through the
    // bits.  mk_fp32_from_float decodes the sentinel and recovers the
    // symbolic Term via loadScalar.  Use when the kernel reads floats from
    // this buffer scalar-wise and passes them to `_vf_` vector intrinsics.
    // The sentinel is a signaling NaN (quiet bit 22 clear, marker bit 21 set):
    // a copy keeps it, any native float operation on it quiets it (and a
    // negation sets its sign), which the decoder refuses (native_guard.hpp).
    void bindScalarRef(void* ptr) {
        scalar_ref_ = true;
        base_addr_ = reinterpret_cast<uintptr_t>(ptr);
        uint32_t* fp = reinterpret_cast<uint32_t*>(ptr);
        size_t n_lanes = num_bytes_ / 4;
        if (n_lanes > 0x10000u)
            throw std::runtime_error("bindScalarRef: " + name_ + " has more lanes than a sentinel can address");
        for (size_t i = 0; i < n_lanes; i++) {
            uint32_t off = static_cast<uint32_t>(i) & 0xFFFFu;
            fp[i] = kScalarRefSentinel | (static_cast<uint32_t>(buffer_id_) << 16) | off;
        }
    }
    uintptr_t baseAddr() const { return base_addr_; }
    size_t numBytes() const { return num_bytes_; }
    bool isConcrete() const { return concrete_ptr_ != nullptr; }

    size_t ptrToByteOffset(const void* ptr) const {
        return static_cast<size_t>(reinterpret_cast<uintptr_t>(ptr) - base_addr_);
    }
    bool contains(const void* ptr) const {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        return addr >= base_addr_ && addr < base_addr_ + num_bytes_;
    }
    // Bytes from `ptr` up to the buffer start, or 0 if `ptr` is not before it.
    size_t bytesBefore(const void* ptr) const {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        return addr < base_addr_ ? static_cast<size_t>(base_addr_ - addr) : 0;
    }

    void _oob(size_t offset, const char* op) const {
        throw BufferOOB(name_, op, (long long)offset,
                        "buffer '" + name_ + "' OOB " + op + " at " +
                        std::to_string((long long)offset) + " (capacity " +
                        std::to_string(num_bytes_) + ") — fixture too small");
    }
    Term getByte(size_t offset) const {
        noteRead((long long)offset, 1);   // a wrapped "negative" offset records as before-start
        // A fallback window reads the real memory it covers (a static table may
        // be longer than the window); a harness buffer's bytes stop at its capacity.
        if (concrete_ptr_ && (fallback_ || offset < num_bytes_))
            return tm_->mkBitVector(8, (uint64_t)concrete_ptr_[offset]);
        if (offset >= num_bytes_) {
            // A load that starts shortly before the buffer wraps to a "negative"
            // offset. XNN_OOB_READS lets kernels over-read but not depend on those
            // bytes, so they are fixed, unconstrained values shared by both kernels.
            size_t before = static_cast<size_t>(0) - offset;
            if (before > kPreBytes) _oob(offset, "read");
            Term& t = pre_bytes_[before];
            if (t.isNull())
                t = tm_->mkConst(tm_->mkBitVectorSort(8), name_ + "_pre" + std::to_string(before));
            return t;
        }
        if (bytes_[offset].isNull()) {  // lazily mint the symbolic byte on first touch
            bytes_[offset] = tm_->mkConst(tm_->mkBitVectorSort(8),
                                          name_ + "_b" + std::to_string(offset));
            init_[offset] = bytes_[offset];
        }
        return bytes_[offset];
    }
    void setByte(size_t offset, Term val) {
        if (concrete_ptr_) throw std::runtime_error("setByte on concrete buffer: " + name_);
        if (offset >= num_bytes_) _oob(offset, "setByte");
        noteWrite((long long)offset, 1);
        bytes_[offset] = val;
    }

    // little-endian: byte[off] is LSB
    Term loadScalar(size_t byte_offset, size_t bits) const {
        size_t n_bytes = bits / 8;
        if (concrete_ptr_ && (fallback_ || byte_offset + n_bytes <= num_bytes_)) {
            noteRead((long long)byte_offset, n_bytes);
            if (bits <= 64) {
                uint64_t value = 0;
                for (size_t i = 0; i < n_bytes; i++) {
                    value |= (uint64_t)concrete_ptr_[byte_offset + i] << (8 * i);
                }
                return tm_->mkBitVector(static_cast<uint32_t>(bits), value);
            }
            Term result = tm_->mkBitVector(8, (uint64_t)concrete_ptr_[byte_offset]);
            for (size_t i = 1; i < n_bytes; i++) {
                Term hi = tm_->mkBitVector(8, (uint64_t)concrete_ptr_[byte_offset + i]);
                result = tm_->mkTerm(Kind::BITVECTOR_CONCAT, {hi, result});
            }
            return result;
        }
        Term result = getByte(byte_offset);   // lazy: mints on first touch
        // A scalar stored whole and read back whole: its bytes are extract[8i+7:8i]
        // of one term; return that term, so a load yields exactly what was stored
        // (and both kernels see one shared leaf, not two concat-of-extract trees).
        Term whole;
        bool same = n_bytes > 1;
        for (size_t i = 0; i < n_bytes && same; i++) {
            Term b = i ? getByte(byte_offset + i) : result;
            if (b.getKind() != Kind::BITVECTOR_EXTRACT) { same = false; break; }
            Op op = b.getOp();
            Term src = b[0];
            if (op[0].getUInt32Value() != 8 * i + 7 || op[1].getUInt32Value() != 8 * i ||
                (i && src != whole) || src.getSort().getBitVectorSize() != bits)
                same = false;
            whole = src;
        }
        if (same) return whole;
        for (size_t i = 1; i < n_bytes; i++) {
            result = tm_->mkTerm(Kind::BITVECTOR_CONCAT,
                                 {getByte(byte_offset + i), result});
        }
        return result;
    }

    void storeScalar(size_t byte_offset, Term val, size_t bits) {
        if (concrete_ptr_) throw std::runtime_error("storeScalar on concrete buffer: " + name_);
        size_t n_bytes = bits / 8;
        noteWrite((long long)byte_offset, n_bytes);
        for (size_t i = 0; i < n_bytes; i++) {
            if (byte_offset + i >= num_bytes_) _oob(byte_offset + i, "write");
            uint32_t lo = static_cast<uint32_t>(i * 8);
            uint32_t hi = lo + 7;
            Op op = tm_->mkOp(Kind::BITVECTOR_EXTRACT, {hi, lo});
            bytes_[byte_offset + i] = tm_->mkTerm(op, {val});
        }
    }

    template <size_t ElemBits, size_t NumLanes>
    NeonVector<ElemBits, NumLanes> loadNeon(size_t byte_offset) const {
        std::array<Term, NumLanes> lanes;
        size_t elem_bytes = ElemBits / 8;
        for (size_t i = 0; i < NumLanes; i++) {
            if constexpr (ElemBits == 8) {
                lanes[i] = getByte(byte_offset + i);
            } else {
                lanes[i] = loadScalar(byte_offset + i * elem_bytes, ElemBits);
            }
        }
        return NeonVector<ElemBits, NumLanes>(*tm_, lanes);
    }

    template <size_t ElemBits, size_t NumLanes>
    void storeNeon(size_t byte_offset, const NeonVector<ElemBits, NumLanes>& vec) {
        if (concrete_ptr_) throw std::runtime_error("storeNeon on concrete buffer: " + name_);
        size_t elem_bytes = ElemBits / 8;
        for (size_t i = 0; i < NumLanes; i++) {
            if constexpr (ElemBits == 8) {
                if (byte_offset + i >= num_bytes_) _oob(byte_offset + i, "write");
                noteWrite((long long)(byte_offset + i), 1);
                bytes_[byte_offset + i] = vec.getLane(i);
            } else {
                storeScalar(byte_offset + i * elem_bytes, vec.getLane(i), ElemBits);
            }
        }
    }

    RVVVector loadRVV(size_t byte_offset, size_t vl, size_t elem_bits) const {
        std::vector<Term> elems;
        elems.reserve(vl);
        size_t elem_bytes = elem_bits / 8;
        for (size_t i = 0; i < vl; i++) {
            if (elem_bits == 8) elems.push_back(getByte(byte_offset + i));
            else elems.push_back(loadScalar(byte_offset + i * elem_bytes, elem_bits));
        }
        return RVVVector(*tm_, elem_bits, elems);
    }

    void storeRVV(size_t byte_offset, const RVVVector& vec) {
        if (concrete_ptr_) throw std::runtime_error("storeRVV on concrete buffer: " + name_);
        size_t elem_bytes = vec.elemBits() / 8;
        for (size_t i = 0; i < vec.getVL(); i++) {
            if (vec.elemBits() == 8) {
                if (byte_offset + i >= num_bytes_) _oob(byte_offset + i, "write");
                noteWrite((long long)(byte_offset + i), 1);
                bytes_[byte_offset + i] = vec.getElement(i);
            } else storeScalar(byte_offset + i * elem_bytes, vec.getElement(i), vec.elemBits());
        }
    }
};

inline SymbolicBuffer& VerificationContext::registerBuffer(
    const std::string& name, size_t num_bytes) {
    auto [it, inserted] = buffers.emplace(
        name, std::make_unique<SymbolicBuffer>(tm, num_bytes, name));
    if (!inserted) throw std::runtime_error("Buffer already registered: " + name);
    registered_buffers.push_back(it->second.get());
    return *it->second;
}

inline void VerificationContext::dump_model_json(const std::string& shape) {
    const char* dir = std::getenv("SALT_CEX_JSON");
    if (!dir || !*dir) return;
    std::string tag = shape;
    for (char& ch : tag) if (ch == ',') ch = '-';
    FILE* f = std::fopen((std::string(dir) + "/cex-" + tag + ".json").c_str(), "w");
    if (!f) return;
    std::fprintf(f, "{\"shape\": \"%s\", \"buffers\": {", shape.c_str());
    bool first = true;
    for (auto& [name, buf] : buffers) {
        if (!buf || buf->isFallback()) continue;
        std::fprintf(f, "%s\n  \"%s\": \"", first ? "" : ",", name.c_str());
        first = false;
        for (size_t i = 0; i < buf->numBytes(); i++) {
            Term v = solver->getValue(buf->getByte(i));
            std::fprintf(f, "%02lx", std::strtoul(v.getBitVectorValue(10).c_str(), nullptr, 10) & 0xffUL);
        }
        std::fprintf(f, "\"");
    }
    // In-out buffers (an accumulating output): their bytes before the kernels ran.
    std::fprintf(f, "\n}, \"initial\": {");
    first = true;
    for (auto& [name, buf] : buffers) {
        if (!buf || buf->isFallback() || buf->isConcrete()) continue;
        bool changed = false;
        for (size_t i = 0; i < buf->numBytes() && !changed; i++) {
            Term t = buf->initialByte(i);
            changed = !t.isNull() && t != buf->getByte(i);
        }
        if (!changed) continue;
        std::fprintf(f, "%s\n  \"%s\": \"", first ? "" : ",", name.c_str());
        first = false;
        for (size_t i = 0; i < buf->numBytes(); i++) {
            Term t = buf->initialByte(i);
            unsigned long v = t.isNull() ? 0UL : std::strtoul(solver->getValue(t).getBitVectorValue(10).c_str(), nullptr, 10) & 0xffUL;
            std::fprintf(f, "%02lx", v);
        }
        std::fprintf(f, "\"");
    }
    std::fprintf(f, "\n}}\n");
    std::fclose(f);
}

inline void VerificationContext::unregisterBuffer(SymbolicBuffer* b) {
    registered_buffers.erase(std::remove(registered_buffers.begin(), registered_buffers.end(), b),
                             registered_buffers.end());
    buffers.erase(b->name());
}

inline SymbolicBuffer& VerificationContext::registerConcreteBuffer(
    const std::string& name, const void* data, size_t num_bytes) {
    auto [it, inserted] = buffers.emplace(
        name, std::make_unique<SymbolicBuffer>(tm, num_bytes, name, data));
    if (!inserted) throw std::runtime_error("Buffer already registered: " + name);
    registered_buffers.push_back(it->second.get());
    return *it->second;
}

inline SymbolicBuffer& VerificationContext::findBuffer(const void* ptr) {
    for (auto* buf : registered_buffers) {
        if (buf->contains(ptr)) return *buf;
    }
    // A load starting shortly before a symbolic buffer (a masked padding lane)
    // belongs to that buffer; its getByte models the leading bytes. Without this,
    // the concrete fallback below would read the buffer's backing C array, i.e.
    // scalar-ref NaN sentinels or zeros instead of the symbolic values.
    SymbolicBuffer* nearest = nullptr;
    for (auto* buf : registered_buffers) {
        size_t gap = buf->bytesBefore(ptr);
        if (!buf->isFallback() && gap > 0 && gap <= SymbolicBuffer::kPreBytes &&
            (!nearest || gap < nearest->bytesBefore(ptr)))
            nearest = buf;
    }
    if (nearest) return *nearest;
    // Static memory (a kernel's own constant table) lies in a loaded image. Any
    // other pointer just past a buffer's capacity overruns the nearest buffer that
    // ends before it (e.g. an unmasked strided load whose inactive lanes run off
    // the end of a row).
    Dl_info img;
    if (ptr && !dladdr(ptr, &img)) {
        uintptr_t a = reinterpret_cast<uintptr_t>(ptr);
        SymbolicBuffer* over = nullptr;
        for (auto* buf : registered_buffers) {
            uintptr_t end = buf->baseAddr() + buf->numBytes();
            if (buf->baseAddr() && a >= end && a - end < 4096 &&
                (!over || end > over->baseAddr() + over->numBytes()))
                over = buf;
        }
        if (over) {
            std::string near;
            for (auto* buf : registered_buffers) {
                char line[160];
                std::snprintf(line, sizeof(line), " %s=[%#lx,+%zu)", buf->name().c_str(),
                              (unsigned long)buf->baseAddr(), buf->numBytes());
                near += line;
            }
            std::fprintf(stderr, "findBuffer: %p is past every buffer; nearest end: %s. Buffers:%s\n",
                         ptr, over->name().c_str(), near.c_str());
            over->_oob(a - over->baseAddr(), "read");
        }
    }
    // Concrete-memory fallback: a miss means `ptr` is not a registered symbolic
    // buffer — it's real concrete memory the kernel reads directly, i.e. a
    // `static const` lookup table (e.g. rsum's onemask_table). Symbolic
    // inputs/outputs/params are always registered, so they never reach here;
    // reading the table's actual bytes is exactly what the real load returns,
    // which is what the verifier should model. Anchor a small concrete window at
    // `ptr` (over-allocated for a vector load; only touched bytes are read).
    if (ptr) {
        char nm[48];
        std::snprintf(nm, sizeof(nm), "concrete@%p", ptr);
        std::fprintf(stderr, "findBuffer: concrete-data fallback for %p (static const?)\n", ptr);
        SymbolicBuffer& w = registerConcreteBuffer(nm, ptr, 64);
        w.markFallback();
        return w;
    }
    std::fprintf(stderr, "findBuffer FAIL: null pointer\n");
    throw std::runtime_error("No buffer found for pointer");
}

inline SymbolicBuffer* VerificationContext::findBufferSafe(const void* ptr) noexcept {
    for (auto* buf : registered_buffers) {
        if (buf->contains(ptr)) return buf;
    }
    return nullptr;
}

// The scalar-ref buffer a sentinel names. Ids come from a counter that wraps at 31, so two live buffers may share
// one; only sentinel-bound buffers count, and an id two of them share decodes to neither (nullptr: the caller
// refuses it) rather than to the first.
inline SymbolicBuffer* VerificationContext::findBufferById(uint8_t id) noexcept {
    SymbolicBuffer* hit = nullptr;
    for (auto* buf : registered_buffers) {
        if (buf->bufferId() != id || !buf->isScalarRef()) continue;
        if (hit) return nullptr;
        hit = buf;
    }
    return hit;
}

} // namespace salt_cvc5

// salt_float16 copy-ctor / copy-assign / dtor — out-of-line so they can use the
// full SymbolicBuffer definition.  See context.hpp for the declaration.
inline salt_float16::salt_float16(const salt_float16& other) : value(other.value) {
    if (!salt_cvc5::g_ctx) return;
    auto it = salt_cvc5::g_scalar_terms.find(&other);
    if (it != salt_cvc5::g_scalar_terms.end()) {
        salt_cvc5::g_scalar_terms.insert_or_assign(this, it->second);
        return;
    }
    if (auto* buf = salt_cvc5::g_ctx->findBufferSafe(&other)) {
        cvc5::Term bv = buf->loadScalar(buf->ptrToByteOffset(&other), 16);
        cvc5::Op op = salt_cvc5::g_ctx->tm.mkOp(
            cvc5::Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV, {5, 11});
        cvc5::Term fp = salt_cvc5::g_ctx->tm.mkTerm(op, {bv});
        salt_cvc5::g_scalar_terms.insert_or_assign(this, fp);
    }
}

inline salt_float16& salt_float16::operator=(const salt_float16& other) {
    if (this == &other) return *this;
    value = other.value;
    if (!salt_cvc5::g_ctx) return *this;
    auto it = salt_cvc5::g_scalar_terms.find(&other);
    if (it != salt_cvc5::g_scalar_terms.end()) {
        salt_cvc5::g_scalar_terms.insert_or_assign(this, it->second);
        return *this;
    }
    if (auto* buf = salt_cvc5::g_ctx->findBufferSafe(&other)) {
        cvc5::Term bv = buf->loadScalar(buf->ptrToByteOffset(&other), 16);
        cvc5::Op op = salt_cvc5::g_ctx->tm.mkOp(
            cvc5::Kind::FLOATINGPOINT_TO_FP_FROM_IEEE_BV, {5, 11});
        cvc5::Term fp = salt_cvc5::g_ctx->tm.mkTerm(op, {bv});
        salt_cvc5::g_scalar_terms.insert_or_assign(this, fp);
    } else {
        salt_cvc5::g_scalar_terms.erase(this);
    }
    return *this;
}

inline salt_float16::~salt_float16() {
    if (!salt_cvc5::g_scalar_terms.empty()) salt_cvc5::g_scalar_terms.erase(this);
}
