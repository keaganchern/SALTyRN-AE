The following optimized RVV kernel was produced by a performance search starting from a verified reference RVV kernel. It compiled, passed the search's concrete tests and is faster, but formal verification found inputs on which its outputs differ from the reference's.

Reference RVV kernel (verified against the original Neon kernel; this is the ground truth):
```c
{reference_code}
```

Optimized RVV kernel (fails verification):
```c
{candidate_code}
```

Verification counterexample:
```
{counterexample}
```

This is a semantic mismatch, not a syntax or runtime issue. The reference kernel defines the correct outputs. If the counterexample says the kernel was refused for a native access instead, the verifier could not follow it: compute on input values with RVV intrinsics (a scalar read from an input may only be passed to an intrinsic, not tested, converted or used in C arithmetic), and write outputs with vector stores, not C assignments.

Task:
- Repair the optimized kernel so it computes exactly the reference's outputs for every input: all bit patterns (NaN, infinity, signed zero, subnormals), every size, every valid parameter value, with the same rounding, saturation and overflow behavior.
- Keep its optimizations (LMUL and vector length choices, unrolling, instruction scheduling, loop structure) wherever they are not the cause of the mismatch, and make the smallest change that fixes it.
- Use the counterexample to localize the first wrong operation. Do not special-case the counterexample's values or restrict the kernel to particular input ranges.

Constraints:
- The function definition must be exactly `{signature}`.
- Keep the same weight and buffer layouts the kernel reads (and its `pack_weights` function, if it has one).
- Use only valid RVV 1.0 intrinsics from `<riscv_vector.h>`.

Return the complete repaired kernel in a single ```c code block.
