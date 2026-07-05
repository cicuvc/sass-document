# ATOMG — Atomic Operation on Global Memory

**Opcode mnemonic:** `ATOMG`  
**Pipe:** `mio_pipe` (MIO — memory I/O pipe, MIO_SLOW_OPS subset)  
**INSTRUCTION_TYPE:** `INST_TYPE_DECOUPLED_RD_WR_SCBD`  
**VIRTUAL_QUEUE:** `$VQ_AGU`

## Semantics

Performs an atomic read-modify-write on global device memory via memory
descriptor. Returns the old value in `Rd`.

`Rd = atomic_op(*global(URc, Ra + offset), Rb)`

Format: `ATOMG{.E}{.COP}{.SEM.SCO.PRIVATE} Pu, Rd, [addr], Rb` (and `Rc` for CAS).

ATOMG is the global-memory counterpart to ATOMS (shared memory). Uses the
same memory-descriptor mechanism as LDG/STG.

## Variant overview

14 encoding variants across 6 opcode slots:

| Group | Opcodes | Operation | Operands |
|-------|---------|-----------|:---:|
| atomg_int | `0x3a8` / `0x19a8` | ADD, MIN, MAX, INC, DEC, AND, OR, XOR, EXCH, SAFEADD | Rd, addr, Rb |
| atomg_fp | `0x3a3` / `0x19a3` | ADD, MIN, MAX (float) | Rd, addr, Rb |
| atomg_cas | `0x3a9` | CAS (compare-and-swap) | Rd, addr, Rb, Rc |

Each group: RaNonRZ, RaRZ, uniform_Ra32, uniform_Ra64, uniform_RaRZ, memdesc.

## ATOMG vs REDG — the return-value distinction

```
ATOMG.E.ADD.STRONG.GPU PT, R11, desc[UR4][R2.64], R11
REDG.E.ADD.STRONG.GPU      desc[UR4][R2.64], R7
```

| Property | ATOMG | REDG |
|----------|-------|------|
| Returns old value | **Yes** — Rd is meaningful | **No** — fire-and-forget |
| Pu (write predicate) | Yes (control result writeback) | No (always discarded) |
| Scoreboard | `INST_TYPE_DECOUPLED_RD_WR_SCBD` | `INST_TYPE_DECOUPLED_RD_SCBD` |
| Pipeline behavior | Must wait for memory read response | Can retire immediately after issue |
| EXCH/SWAP | Yes | No (reduction only) |
| CAS | Yes (separate sub-opcode) | No |
| SAFEADD | Yes | No |
| Int opcodes | `0x3a8` / `0x19a8` | `0x98e` / `0x198e` |
| FP opcodes | `0x3a3` / `0x19a3` | `0x9a6` / `0x19a6` |
| Same ops? | ADD/MIN/MAX/INC/DEC/AND/OR/XOR | ADD/MIN/MAX/INC/DEC/AND/OR/XOR (same set minus EXCH) |

**Hardware rationale:** REDG can issue the store and immediately retire
without stalling for the read response. ATOMG must wait for the old value to
arrive and write it into Rd. If the program doesn't need the old value
(`Rd = atomic_op(addr, val)` but `Rd` is dead), the compiler should use REDG
for better throughput. On sm_90, ptxas automatically selects REDG when the
old value is unused:

```c
old = atomicAdd(p, 1);    // → ATOMG (needs Rd)
     atomicAdd(p, 1);     // → REDG  (fire-and-forget)
```

## Verified encodings

| Lo64 | Disassembly |
|------|-------------|
| `0x0000000b020b09a8` | `@P0 ATOMG.E.ADD.STRONG.GPU PT, R11, desc[UR4][R2.64], R11` |
| `0x00000008020973a9` | `ATOMG.E.CAS.STRONG.GPU PT, R9, [R2], R8, R9` |
| `0x0000000d020b79a8` | `ATOMG.E.EXCH.STRONG.GPU PT, R11, desc[UR4][R2.64], R13` |

### PTX to SASS mapping

| PTX | SASS (sm_90) |
|-----|-------------|
| `atom.global.add.u32 %r, [%ptr], %val` | `ATOMG.E.ADD desc[UR][Ra.64], Rb` |
| `atom.global.cas.b32 %r, [%ptr], %cmp, %val` | `ATOMG.E.CAS [Ra], Rb, Rc` |
| `atom.global.exch.b32 %r, [%ptr], %val` | `ATOMG.E.EXCH desc[UR][Ra.64], Rb` |
| `red.global.add.u32 [%ptr], %val` (result unused) | `REDG.E.ADD desc[UR][Ra.64], Rb` |
