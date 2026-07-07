# PREEXIT / ACQBULK — Programmatic Dependent Launch (PDL) control

**Opcode mnemonics:** `PREEXIT` = `0b100000101101` = **0x82d**; `ACQBULK` = `0b100000101110` = **0x82e** | **Pipe:** `cbu_pipe` | compute-only (`SHADER_TYPE==CS`)

The two ends of Hopper **Programmatic Dependent Launch (PDL)** — the SASS lowering of PTX
`griddepcontrol`, which lets a grid signal that its dependent grids may start early
(overlapping tail of one grid with the head of the next).

## Semantics (verified PTX→SASS)
| PTX | SASS | role |
|-----|------|------|
| `griddepcontrol.launch_dependents` | **`PREEXIT`** | **producer**: this grid has advanced enough that dependent grids may launch |
| `griddepcontrol.wait` | **`ACQBULK`** | **consumer**: wait/acquire until prerequisite grids have signaled |

- **`PREEXIT`** ("pre-exit"): announces the grid is near its productive end so the driver can
  begin launching dependents. `INST_TYPE_DECOUPLED_BRU_DEPBAR_RD_SCBD`, `VQ_UNORDERED` — it
  signals and continues (decoupled); the compiler hoists it early so dependents launch ASAP.
- **`ACQBULK`** ("acquire bulk"): blocks until the prerequisite grid's data is guaranteed
  visible. `INST_TYPE_COUPLED_MATH`, `VIRTUAL_QUEUE=None` — a fixed-latency *acquire* (same
  coupled-on-cbu pattern as `ELECT`/`ENDCOLLECTIVE`), i.e. it must complete before dependent
  work reads shared results.

## Variant overview
Each is a single CLASS / opcode, **operand-less** (all `ISRC_*`/`IDEST_*` = 0) — only a
guard predicate.

## Operands / fields (128-bit)
| bits | field | notes |
|------|-------|-------|
| [91]∥[11:0] | opcode | 0x82d PREEXIT / 0x82e ACQBULK |
| [14:12]/[15] | `Pg`/`Pg_not` | guard predicate (7=PT hidden) |

## Cross-comparison
| | **PREEXIT** | **ACQBULK** |
|--|-------------|-------------|
| PTX | `griddepcontrol.launch_dependents` | `griddepcontrol.wait` |
| side | producer (signal) | consumer (wait/acquire) |
| INSTRUCTION_TYPE | DECOUPLED_BRU | COUPLED_MATH |
| VIRTUAL_QUEUE | VQ_UNORDERED | None |
| blocks? | no (signal + continue) | yes (acquire) |

Neither is in `RPC_WRITERS` or `CBU_OPS_WITH_REQ`. Distinct from the CGA cluster barrier
(`UCGABAR_*`, cluster-scope arrive/wait) and from `EXIT` (thread termination) — PDL is a
**grid-to-grid** launch-overlap mechanism.

## Verified encodings (decoder: `tools/decode_preexit.py`)
Self-test 3/3; `tests/griddep2.cu` (inline `griddepcontrol`) 2/2 per dump.

| Lo64 | Hi64 | Disassembly | src |
|------|------|-------------|-----|
| 0x000000000000782d | 0x000ff00000000000 | `PREEXIT` | `griddepcontrol.launch_dependents` |
| 0x000000000000782e | 0x000fcc0000000000 | `ACQBULK` | `griddepcontrol.wait` |
| 0x000000000000182d | 0x000ff00000000000 | `@P1 PREEXIT` | guard (spec-inferred) |

### PTX→SASS mapping
Determined by diffing a baseline kernel against kernels with each `griddepcontrol` form
(sm_90/CUDA 13.1): `launch_dependents` adds exactly one `PREEXIT`, `wait` adds exactly one
`ACQBULK`; the baseline has neither. Emitted for kernels launched with the PDL attribute
(`cudaLaunchAttributeProgrammaticStreamSerialization` / `cudaGridDependencySynchronize()`).

## Open questions
- Whether `PREEXIT` interacts with the at-exit state (`ATEXIT_PC`/`MATEXIT`) beyond the PDL
  signal, and the exact scope of `ACQBULK`'s acquire (grid vs cluster), is not spec-stated.
- Non-PT guard forms are inferred from the shared cbu encoding, not sampled from emitted code.
