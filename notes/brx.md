# BRX / BRXU — Register-indirect branch (GPR / uniform-GPR target)

**Opcode mnemonics:** `BRX` = `0b100101001001` = **0x949**; `BRXU` = `0b1100101011000` = **0x1958** | **Pipe:** `cbu_pipe` (Branch Unit) | **INSTRUCTION_TYPE:** `INST_TYPE_DECOUPLED_BRU_DEPBAR_RD_SCBD`

The register-indirect relatives of `BRA`/`JMP`: the branch target comes from a **register
value plus an immediate offset**, rather than from an immediate/const alone. Used for
compiler-built jump tables and computed branches. `BRX` reads a **GPR pair**; `BRXU` reads
a **uniform GPR pair**.

## Semantics
`@Pg BRX{.INC/.DEC} {Pp,} Ra [, off]` branches (for lanes where `Pg` holds) to a target
formed from the 64-bit register `Ra` and the signed immediate `off` (`= sImm*4`). `Ra` is
`ISRC_A_SIZE = 64` and is range/alignment-checked as an **even-aligned register pair**
(`Ra%2==0`, `Ra != R254`), i.e. it holds a 64-bit target/base address; the encoded `off`
is added to it. `BRXU` is identical but the base comes from a **uniform** register pair
`URa` (`URa%2==0`), and it additionally carries a `cond` modifier.

The optional `Pp` is the divergence predicate (same role as in `BRA`); `depth`
(`.INC`/`.DEC`) adjusts the call-depth counter.

## Variant overview
| mnem | opcode `{b91,[11:0]}` | target register | cond field | alt |
|------|-----------------------|-----------------|------------|-----|
| `brx_`  | 0x0949 | `Ra` [31:24] (GPR) | — | `brx_rel_` |
| `brxu_` | 0x1958 | `URa` [29:24] (UGPR) | [33:32] `COND` | `brxu_rel_` |

`_rel_` alternates only change how the assembler is given the offset (label vs explicit
relative), same bits.

## Operands / fields (128-bit)
| bits | field | BRX | BRXU |
|------|-------|-----|------|
| [91]∥[11:0] | opcode | 0x949 | 0x1958 |
| [14:12]/[15] | `Pg`/`Pg_not` | guard | guard |
| [89:87]/[90] | `Pp`(`Pnz`)/`Pp_not` | divergence pred (≠PT → printed) | same |
| [86:85] | `depth` `DEPTH` | `.INC`/`.DEC` | `.INC`/`.DEC` |
| [33:32] | `cond` `COND` | — | 0=none,1=`.U`,2=`.DIV`,3=`.CONV` |
| [31:24] | `Ra` | GPR (RZ=255) | — |
| [29:24] | `URa` | — | UGPR (URZ=63) |
| [81:34]∥[23:16] | `sImm` | 56-bit signed, offset = `sImm*4` | same |

### Offset rendering
`off = (sImm*4) & 0xffffffffff` (40-bit address mask). **Omitted entirely when `sImm==0`**
(`BRXU UR4`, `BRX R4`). Negatives print masked: `sImm=-1 → 0xfffffffffc`.

## Cross-comparison
| | BRA/JMP | **BRX** | **BRXU** | CALL |
|--|---------|---------|----------|------|
| target | imm/const | GPR pair + off | UGPR pair + off | reg/const/imm |
| `RPC_WRITERS` | yes | yes | yes | yes |
| `CBU_OPS_WITH_REQ` (`&req=`) | BRA yes / JMP no | **yes** | **yes** | yes |

BRX is to BRA what JMX is to JMP: the indirect form. BRXU/JMXU are the uniform-register
indirect forms (BRXU shares the numeric opcode 0x1958 with the ref-memo `BRXU` entry).

## Latency
`cbu_pipe` = `BRU_OPS`. Both are `RPC_WRITERS` → **9-cycle** RPC true-dependency
(`sm_90_latencies.txt:411,414`) and `CBU_OPS_WITH_REQ` (line 219, honor `&req=`).
`DECOUPLED_BRU`, `MIN_WAIT_NEEDED=1`.

## Verified encodings (decoder: `tools/decode_brx.py`)
Neither is emitted by ptxas on sm_90/CUDA 13.1: computed goto is rejected in device code,
dense switches lower to `BRA` trees, and libcublas has 0 BRX/BRXU. Ground truth via
**cubin-patching + nvdisasm**: self-test 7/7, plus a **randomized battery of 300 patched
encodings decoded 100%** (both ops, all cond/depth/predicate/register/offset combos).

| Lo64 | Hi64 | Disassembly |
|------|------|-------------|
| 0x0000000404240949 | 0x000fea0003800000 | `@P0 BRX R4 0x490` |
| 0xfffffffc04fc0949 | 0x000fea0003800000 | `@P0 BRX R4 0xfffffffff0` (off = -16) |
| 0x0000000006400949 | 0x000fea0001800000 | `@P0 BRX P3, R6 0x100` (Pp=P3) |
| 0x0000000006400949 | 0x000fea0003a00000 | `@P0 BRX.INC R6 0x100` (depth) |
| 0x0000000004000958 | 0x000fea000b800000 | `@P0 BRXU UR4` (off=0 omitted) |
| 0x0000000204000958 | 0x000fea000b800000 | `@P0 BRXU.DIV UR4` |
| 0x0000000304000958 | 0x000fea000b800000 | `@P0 BRXU.CONV UR4` |

Hand-check `BRX R4 0x490`: opcode 0x949; `Ra`[31:24]=4→R4; `sImm=(1<<8)|0x24=0x124`,
`0x124*4=0x490` (raw offset, not PC-relative — a `BRA` with the same bits renders `0x580`).

## Open questions
- Exact runtime target formula (`Ra + off` absolute vs. relative-to-anchor) can't be
  pinned from static disasm alone since `Ra` is a runtime value; the encoding + 64-bit
  even-aligned `Ra` strongly imply `target = Ra(64-bit) + sImm*4`.
- The 40-bit offset print mask matches sampled cases; behavior for offsets exceeding the
  GPU virtual-address width is unverified.
- Real-world jump-table idiom (which instruction populates `Ra`, table layout) is
  unobserved because ptxas never emitted these in the sampled code.
