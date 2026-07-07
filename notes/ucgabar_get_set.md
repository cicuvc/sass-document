# UCGABAR_GET / UCGABAR_SET — CGA cluster-barrier query / set

**Opcode mnemonics:** `UCGABAR_GET` = `0b1010111000111` = **0x15c7**; `UCGABAR_SET` = `0b1001111000111` = **0x13c7** | **Pipe:** `udp_pipe` (uniform datapath) | **INSTRUCTION_TYPE:** `INST_TYPE_DECOUPLED_BRU_DEPBAR_RD_SCBD` | compute-only (`SHADER_TYPE==CS`)

The query/set members of the thread-block-cluster (CGA) barrier family (see
`ucgabar_arv.md` for `ARV`/`WAIT` and the `CGABARRIER` resource). `GET` reads the barrier
state/token into a uniform register; `SET` initializes/sets the barrier from a uniform
register.

## Semantics
- **`@UPg UCGABAR_GET URd`** — read the cluster barrier's current token/phase into uniform
  register `URd` [21:16]. A **`CGABAR_READERS`** member.
- **`@UPg UCGABAR_SET URb`** — set/initialize the cluster barrier from `URb` [37:32]. A
  **`CGABAR_WRITERS`** member.

Both are uniform ops (uniform-predicate `UPg` guard), no other operands.

## Operands / fields (128-bit)
| bits | field | GET | SET |
|------|-------|-----|-----|
| [91]∥[11:0] | opcode | 0x15c7 (b91=1) | 0x13c7 (b91=1) |
| [14:12]/[15] | `UPg`/`UPg_not` | uniform guard (7=UPT hidden) | uniform guard |
| [21:16] | `URd` | barrier token dest | — |
| [37:32] | `URb` | — | source uniform reg |

## Latency (CGABARRIER resource)
`GET` reads the `CGABARRIER` resource (writer→reader true-dep of **6 cycles** from
`ARV`/`SET`); `SET` writes it (`sm_90_latencies.txt:369-373`). `udp_pipe`, `DECOUPLED_BRU`,
`VQ_UNORDERED`.

## Not emitted / not rendered (CUDA 13.1)
Neither is produced by the sampled toolchain: cooperative-groups `cluster_group`
arrive/wait lower to `UCGABAR_ARV`/`_WAIT`, and `cuda::barrier` (mbarrier) lowers to the
`SYNCS.*` shared-memory-barrier family — **not** GET/SET. Moreover, **nvdisasm (CUDA 13.1)
does not render these opcodes**: hand-patching an instruction to `0x15c7`/`0x13c7` produces
**headerless raw bytes** (the disassembler skips the address label and prints only the
128-bit hex), e.g.
```
/*0070*/  CGAERRBAR ;
          /* 0x00000000000575c7 */   <- patched UCGABAR_GET UR5, 0x0080 label omitted
          /* 0x000fe20008000000 */
/*0090*/  IMAD ...
```
So this is a **spec-vs-tool gap**: the mnemonics exist in the sm_90 ISA DB (with the fields
above) but the shipped disassembler omits them. The `URd`/`URb` renderings below are
therefore **spec-inferred**, validated only at the field-extraction level (the patched
bytes carry `URd=5` / `URb=7`), not against nvdisasm output.

## Verified encodings (decoder: `tools/decode_ucgabar.py`)
Field-level (spec-inferred, nvdisasm does not render):
| Lo64 | Hi64 | Decoder output |
|------|------|----------------|
| 0x00000000000575c7 | 0x000fe20008000000 | `UCGABAR_GET UR5` |
| 0x00000007000073c7 | 0x000fe20008000000 | `UCGABAR_SET UR7` |

(The decoder's `UCGABAR_ARV`/`_WAIT` outputs are byte-exact against real
`cluster.barrier_arrive/wait`; see `ucgabar_arv.md`.)

## Open questions
- Actual mnemonic spelling/operand rendering nvdisasm *would* use for GET/SET is unknown
  (unrendered); `URd`/`URb` placement is taken from the ISA DB encoding only.
- Which host construct (if any) emits GET/SET — possibly driver/runtime cluster-launch
  setup or a future CG API; not reproduced here.
- Exact `CGABARRIER` token layout that GET reads / SET writes is not spec-exposed.
