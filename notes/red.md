# RED — Reduction (shared memory, fire-and-forget)

**Opcode mnemonic:** `RED`  
**Pipe:** `mio_pipe` (MIO — memory I/O pipe, MIO_SLOW_OPS subset)  
**INSTRUCTION_TYPE:** `INST_TYPE_DECOUPLED_RD_SCBD` (decoupled read-only scoreboard)  
**VIRTUAL_QUEUE:** `$VQ_AGU`

## Semantics

Performs an atomic reduction (read-modify-write with fire-and-forget semantics)
on shared memory. Stores the old value to `Rd` and applies the operation to
`*smem[addr]`.

`Rd, *smem[addr] = reduction_op(*smem[addr], Rb)`

Unlike ATOMS (which returns the old value), RED is fire-and-forget — the
result is written back to memory but the return value is optional/dead.

## Status on sm_90

**RED is effectively dead on sm_90.** PTX `red.shared.*` is lowered to
`ATOMS.POPC.INC` (the `atoms_arrive__popcinc` variant) rather than RED. The
12 RED encoding variants appear in the spec but are not emitted by ptxas.

The corresponding global-memory reduction instruction **REDG** is actively used:
`red.global` → `REDG.E.ADD.STRONG.GPU desc[UR][Ra.64], Rb`.

## RED vs REDG

| Property | RED | REDG |
|----------|-----|------|
| Memory space | Shared (on-chip) | Global (device) |
| Opcodes | `0x9a6`/`0x19a6` (fp), `0x98e`/`0x198e` (int) | same |
| memdesc | No | Yes (`desc[URc][Ra.64]`) |
| Shader | CS only | All shaders |
| E/SCO/COP | No | Yes (like LDG/STG) |
| Active | Dead (→ ATOMS) | Active |
| Operands | Rd, [Ra], Rb | Rd, desc[URc][Ra], Rb |

RED and REDG share the same opcodes — differentiated by `memdesc` bit and
shader-type constraints. This is analogous to LDS/LDG or STS/STG pairs.

## Variant overview (spec-only)

12 encoding variants across 4 opcode slots:

| Group | Opcodes | Data type | Ops (enum) | Sizes |
|-------|---------|-----------|------------|-------|
| red_fp | 0x9a6 / 0x19a6 | Float (F16x2/F16x4/F16x8/BF16/F32/F64) | ADD(0), MIN(1), MAX(2) | Various |
| red_int | 0x98e / 0x198e | Integer | ADD(0),MIN(1),MAX(2),INC(3),DEC(4),AND(5),OR(6),XOR(7) | U32/S32/U64/S64 |

Each group has 6 variants: RaNonRZ, RaRZ, uniform_Ra32, uniform_Ra64, uniform_RaRZ, memdesc.

## PTX to SASS mapping

| PTX | SASS (sm_90) |
|-----|-------------|
| `red.shared.add.u32 [%smem], %val` | **ATOMS.POPC.INC** (not RED!) |
| `red.global.add.u32 [%gmem], %val` | **REDG.E.ADD** desc[...], Rb |
