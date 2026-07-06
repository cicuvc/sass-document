# JMX / JMXU — Absolute register-indirect jump (GPR / uniform-GPR target)

**Opcode mnemonics:** `JMX` = `0b100101001100` = **0x94c**; `JMXU` = `0b1100101011001` = **0x1959** | **Pipe:** `cbu_pipe` (Branch Unit) | **INSTRUCTION_TYPE:** `INST_TYPE_DECOUPLED_BRU_DEPBAR_RD_SCBD`

The absolute-indirect twins of `BRX`/`BRXU` — same "target = register + immediate offset"
mechanism, but the *absolute* jump family (as `JMP` is to `BRA`). `JMX` reads a **GPR
pair**; `JMXU` reads a **uniform GPR pair**.

## Semantics
`@Pg JMX{.INC/.DEC} {Pp,} Ra [, off]` jumps (for lanes where `Pg` holds) to a target
formed from the 64-bit register `Ra` (`ISRC_A_SIZE = 64`, even-aligned pair,
`Ra != R254`) and the signed immediate `off = sImm*4`. `JMXU` is identical but the base is
a uniform register pair `URa` and it carries a `cond` modifier. `Pp` is the divergence
predicate; `depth` adjusts the call-depth counter.

The **field layout and disassembly rendering are byte-for-byte identical to BRX/BRXU**
(verified); only the mnemonic differs. Semantically the BR* pair is the relative-indirect
family and the JM* pair is the absolute-indirect family (mirroring BRA vs JMP), a
distinction in runtime target interpretation not visible in the static text.

## Variant overview
| mnem | opcode `{b91,[11:0]}` | target register | cond field | alt |
|------|-----------------------|-----------------|------------|-----|
| `jmx_`  | 0x094c | `Ra` [31:24] (GPR) | — | `jmx_rel_` |
| `jmxu_` | 0x1959 | `URa` [29:24] (UGPR) | [33:32] `COND` | `jmxu_rel_` |

## Operands / fields (128-bit)
| bits | field | JMX | JMXU |
|------|-------|-----|------|
| [91]∥[11:0] | opcode | 0x94c | 0x1959 |
| [14:12]/[15] | `Pg`/`Pg_not` | guard | guard |
| [89:87]/[90] | `Pp`(`Pnz`)/`Pp_not` | divergence pred (≠PT → printed) | same |
| [86:85] | `depth` `DEPTH` | `.INC`/`.DEC` | `.INC`/`.DEC` |
| [33:32] | `cond` `COND` | — | 0=none,1=`.U`,2=`.DIV`,3=`.CONV` |
| [31:24] | `Ra` | GPR (RZ=255) | — |
| [29:24] | `URa` | — | UGPR (URZ=63) |
| [81:34]∥[23:16] | `sImm` | 56-bit signed, offset = `sImm*4` | same |

Offset rendering (same as BRX/BRXU): `off = (sImm*4) & 0xffffffffff`, **omitted when
`sImm==0`** (`JMX R6`, `JMXU UR4`); negatives print masked (`-16 → 0xfffffffff0`).

## Cross-comparison
| | BRA | JMP | BRX | **JMX** | BRXU | **JMXU** |
|--|-----|-----|-----|---------|------|----------|
| target | rel imm | abs imm/const | GPR+off (rel) | **GPR+off (abs)** | UGPR+off (rel) | **UGPR+off (abs)** |
| opcode | 0x947 | 0x94a/0xb4a | 0x949 | **0x94c** | 0x1958 | **0x1959** |
| `RPC_WRITERS` | y | y | y | **y** | y | **y** |
| `CBU_OPS_WITH_REQ` | BRA y | JMP n | y | **y** | y | **y** |

## Latency
`cbu_pipe` = `BRU_OPS`. Both `RPC_WRITERS` → **9-cycle** RPC true-dependency
(`sm_90_latencies.txt:411,414`) and `CBU_OPS_WITH_REQ` (line 219, honor `&req=`).
`DECOUPLED_BRU`, `MIN_WAIT_NEEDED=1`.

## Verified encodings (decoder: `tools/decode_jmx.py`, shared core in `decode_brx.py`)
Not emitted by ptxas (see `brx.md`). Ground truth via **cubin-patching + nvdisasm**:
self-test 7/7, plus a **randomized battery of 300 patched encodings decoded 100%**.

| Lo64 | Hi64 | Disassembly |
|------|------|-------------|
| 0x000000040424094c | 0x000fea0003800000 | `@P0 JMX R4 0x490` |
| 0x000000000600094c | 0x000fea0003800000 | `@P0 JMX R6` (off=0 omitted) |
| 0xfffffffc06fc094c | 0x000fea0003800000 | `@P0 JMX R6 0xfffffffff0` (off = -16) |
| 0x0000000004000959 | 0x000fea000b800000 | `@P0 JMXU UR4` |
| 0x0000000204000959 | 0x000fea000b800000 | `@P0 JMXU.DIV UR4` |
| 0x0000000304000959 | 0x000fea000b800000 | `@P0 JMXU.CONV UR4` |

Hand-check `JMX R4 0x490`: opcode 0x94c; `Ra`[31:24]=4→R4; `sImm=(1<<8)|0x24=0x124`,
`0x124*4=0x490`.

## Open questions
- The BR*/JM* runtime distinction (relative-indirect vs absolute-indirect target) is not
  observable statically; it mirrors the confirmed BRA(rel)/JMP(abs) split.
- Like BRX/BRXU, real jump-table usage is unobserved because ptxas never emitted these in
  the sampled code; register-population idiom and 40-bit offset edge cases are unverified.
