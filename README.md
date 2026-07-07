# sass-dec

Reverse-engineering the **NVIDIA Hopper (sm_90) SASS** instruction set from
`nvdisasm`-dumped ISA description files. The goal is to reconstruct how SASS
instructions decode — encoding bit layout, functional-unit grouping, and
scoreboard/latency behavior — and to write a per-instruction reference doc for
every compute instruction.

There is no build system: this is a *reading and interpreting* project on top of
two raw ISA dumps, supported by a Python extractor, per-instruction decoders,
CUDA test kernels, and research notes.

## Layout

| Path | What it is |
| --- | --- |
| `sm_90_instructions.txt` | Full instruction/encoding spec (~159k lines). Grep-first; never read whole. |
| `sm_90_latencies.txt` | Pipe grouping + scoreboard/latency tables (~441 lines). |
| `sm_75_instructions.txt`, `sm_80_instructions.txt` | Older-arch dumps for cross-arch comparison. |
| `tools/` | stdlib-only extractor + query CLI + per-instruction decoders. |
| `notes/` | Per-instruction and per-topic reference docs (the deliverable). |
| `tests/` | CUDA (`.cu`) kernels that force specific SASS encodings for verification. |
| `TODO.md` | Master checklist of instructions to document (207 in scope). |
| `ref_memo.txt` | Curated sm_70..sm_90 opcode roster (source of the checklist). |
| `sm90.json` | Generated queryable DB (~21 MB, gitignored/regenerable). |

## Tooling

The spec is parsed into a queryable JSON DB — prefer it over ad-hoc `grep`.

```bash
# Parse both .txt files -> sm90.json (has a built-in validation gate)
python3 tools/parse_sm90.py

# Query the DB
python3 tools/query_sm90.py mnem <NAME>       # variants, opcodes, format, pipe
python3 tools/query_sm90.py class <name> -v   # full CLASS block
python3 tools/query_sm90.py layout <class>    # 128-bit field map
python3 tools/query_sm90.py opcode <hex|0b|int>
python3 tools/query_sm90.py enum <Name>       # modifier value map
python3 tools/query_sm90.py table <Name>      # decode table
python3 tools/query_sm90.py pipe <MNEMONIC>   # functional-unit membership
python3 tools/query_sm90.py stats
```

A clean parse prints `validation OK`: **1589 variants** (1168 `CLASS` +
421 `ALTERNATE CLASS`), **238 mnemonics**, 414 enums, 84 tables, 2309 FUNIT
fields, 277 pipe entries.

`tools/decode_<mnem>.py` are minimal per-instruction decoders: they extract
fields from a 128-bit encoding (lo64 + hi64) and reconstruct the SASS assembly,
validated against real cuobjdump vectors.

## Key facts about the ISA

- Each sm_90 SASS instruction is **128 bits / 16 bytes** = hi64 `[127:64]` + lo64 `[63:0]`.
  (The file header says `WORD_SIZE 64` — ignore it; trust the 128-bit width.)
- Opcode is a **13-bit** field: `{bit[91], bits[11:0]}`.
- Registers: 8-bit GPR (`0xFF` = `RZ`), 6-bit uniform (`UR0`–`UR63`).
- Predicates: 3-bit (`PT` = 7) plus a 1-bit negate flag.
- Field names encode bit position: `BITS_<width>_<hi>_<lo>_<name>` (MSB:LSB).

See `AGENTS.md` for the full spec-layout guide and the per-instruction
documentation recipe.

## Status

- Instructions in scope: **207** (compute; texture/surface/graphics and
  pseudo/lowered opcodes excluded).
- Notes written: 168 · decoders: 96 · test kernels: 103.
- Progress is tracked in `TODO.md`.
