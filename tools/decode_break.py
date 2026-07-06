#!/usr/bin/env python3
"""Decoder / validator for the sm_90 BREAK instruction (opcode 0x942).

BREAK peels the guarded lanes out of convergence-barrier register Bi's
participant set. It shares the CBU base layout with BSSY/BSYNC (guard Pg,
second predicate Pp, barReg [19:16]) but carries no branch target, so the
shared decoder in decode_bssy.py handles it directly.

Usage: python3 decode_break.py   (runs the built-in self-test vectors)
"""

from decode_bssy import decode

# (pc, lo64, hi64, expected cuobjdump rendering)
VECTORS = [
    # tests/break_test.cu  (kB: triple-nested, break-all peels inner barrier B1)
    (0x01f0, 0x0000000000018942, 0x000fea0003800000, "@!P0 BREAK B1"),
    (0x02a0, 0x0000000000017941, 0x000fea0003800000, "BSYNC B1"),
    (0x0300, 0x0000000000007941, 0x000fea0003800000, "BSYNC B0"),
    # tests/bssy_break_test.cu  (nested loops + goto)
    (0x0180, 0x0000000000018942, 0x000fea0003800000, "@!P0 BREAK B1"),
]


def main():
    ok = 0
    for pc, lo, hi, exp in VECTORS:
        got = decode(lo, hi, pc)
        flag = "OK " if got == exp else "XX "
        if got == exp:
            ok += 1
        print("%s pc=0x%04x  %-18s  (expected %s)" % (flag, pc, got, exp))
    print("\n%d/%d vectors matched" % (ok, len(VECTORS)))


if __name__ == "__main__":
    main()
