#!/usr/bin/env python3
"""Decoder / validator for the sm_90 BSYNC instruction (opcode 0x941).

BSYNC waits for all participating lanes of convergence-barrier register Bi and
reconverges them at the recorded reconvergence PC. It shares the CBU base layout
with BSSY/BREAK (guard Pg, second predicate Pp, barReg [19:16]) and carries no
branch target, so the shared decoder in decode_bssy.py handles it directly.

Usage: python3 decode_bsync.py   (runs the built-in self-test vectors)
"""

from decode_bssy import decode

# (pc, lo64, hi64, expected cuobjdump rendering)
VECTORS = [
    # tests/bssy_test.cu (switch kernel)
    (0x0160, 0x0000000000007941, 0x000fea0003800000, "BSYNC B0"),
    # tests/bssy_break_test.cu (nested loops + goto)
    (0x01f0, 0x0000000000017941, 0x000fea0003800000, "BSYNC B1"),
    (0x0250, 0x0000000000007941, 0x000fea0003800000, "BSYNC B0"),
    # tests/break_test.cu (kB: triple-nested)
    (0x02a0, 0x0000000000017941, 0x000fea0003800000, "BSYNC B1"),
    (0x0300, 0x0000000000007941, 0x000fea0003800000, "BSYNC B0"),
    # libcublas.so
    (0x0b90, 0x0000000000007941, 0x000fea0003800000, "BSYNC B0"),
]


def main():
    ok = 0
    for pc, lo, hi, exp in VECTORS:
        got = decode(lo, hi, pc)
        flag = "OK " if got == exp else "XX "
        if got == exp:
            ok += 1
        print("%s pc=0x%04x  %-12s  (expected %s)" % (flag, pc, got, exp))
    print("\n%d/%d vectors matched" % (ok, len(VECTORS)))


if __name__ == "__main__":
    main()
