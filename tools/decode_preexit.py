#!/usr/bin/env python3
"""Decoder for the sm_90 Programmatic Dependent Launch (PDL) control ops
PREEXIT (0x82d) and ACQBULK (0x82e).

These implement PTX `griddepcontrol`:
  griddepcontrol.launch_dependents  ->  PREEXIT   (producer: dependents may launch)
  griddepcontrol.wait               ->  ACQBULK   (consumer: wait/acquire prerequisite)

Both are operand-less (guard predicate only), compute-only (SHADER_TYPE==CS),
dispatched on cbu_pipe. PREEXIT is DECOUPLED_BRU (signals and continues);
ACQBULK is COUPLED_MATH / VQ=None (a fixed-latency acquire, like ELECT/ENDCOLLECTIVE).

Fields (128-bit):
  opcode = {bit[91], bits[11:0]}   0x82d PREEXIT | 0x82e ACQBULK
  Pg=[14:12]/[15]   guard predicate

Usage: python3 decode_preexit.py            (self-test)
       python3 decode_preexit.py <sass.txt>  (validate every PREEXIT/ACQBULK)
"""
import re
import sys

OPC = {0x82d: "PREEXIT", 0x82e: "ACQBULK"}


def bits(v, hi, lo):
    return (v >> lo) & ((1 << (hi - lo + 1)) - 1)


def pred(idx, neg):
    return ("!" if neg else "") + ("PT" if idx == 7 else "P%d" % idx)


def decode(lo64, hi64, pc=0):
    inst = lo64 | (hi64 << 64)
    opcode = (bits(inst, 91, 91) << 12) | bits(inst, 11, 0)
    if opcode not in OPC:
        return "?opcode 0x%x" % opcode
    pg, pg_not = bits(inst, 14, 12), bits(inst, 15, 15)
    guard = "" if (pg == 7 and pg_not == 0) else "@%s " % pred(pg, pg_not)
    return "%s%s" % (guard, OPC[opcode])


# (lo64, hi64, expected) — real griddepcontrol lowering
VECTORS = [
    (0x000000000000782d, 0x000ff00000000000, "PREEXIT"),   # griddepcontrol.launch_dependents
    (0x000000000000782e, 0x000fcc0000000000, "ACQBULK"),   # griddepcontrol.wait
    (0x000000000000182d, 0x000ff00000000000, "@P1 PREEXIT"),   # patch (Pg=1)
]


def run_vectors():
    ok = 0
    for lo, hi, exp in VECTORS:
        got = decode(lo, hi)
        ok += got == exp
        print("%s %-14s (exp %s)" % ("OK " if got == exp else "XX ", got, exp))
    print("\n%d/%d vectors matched" % (ok, len(VECTORS)))


LINE = re.compile(r"/\*([0-9a-f]+)\*/\s+(.*?);\s*/\*\s*([0-9a-fx]+)\s*\*/")
HEX = re.compile(r"/\*\s*([0-9a-fx]+)\s*\*/")


def validate_dump(path):
    lines = open(path).readlines()
    total = ok = 0
    for i, ln in enumerate(lines):
        m = LINE.search(ln)
        if not m or not re.search(r"\b(PREEXIT|ACQBULK)\b", m.group(2)):
            continue
        text, lo = m.group(2).strip(), int(m.group(3), 16)
        hm = HEX.search(lines[i + 1]) if i + 1 < len(lines) else None
        if not hm:
            continue
        got = decode(lo, int(hm.group(1), 16))
        total += 1
        ok += got == text
        if got != text:
            print("XX got %-14s exp %-14s [%016x]" % (got, text, lo))
    print("%s: %d/%d PREEXIT/ACQBULK matched" % (path, ok, total))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            validate_dump(p)
    else:
        run_vectors()
