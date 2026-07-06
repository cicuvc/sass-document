#!/usr/bin/env python3
"""Decoder / validator for the sm_90 absolute-indirect branches JMX (0x94c) and
JMXU (0x1959).

They are the absolute-indirect twins of BRX/BRXU and share the identical field
layout, so the shared decoder in decode_brx.py handles them directly (only the
mnemonic differs). See notes/jmx.md.

Usage: python3 decode_jmx.py   (built-in self-test vectors)
"""
from decode_brx import decode

# (lo64, hi64, expected) — ground truth from cubin-patch + nvdisasm
VECTORS = [
    (0x000000040424094c, 0x000fea0003800000, "@P0 JMX R4 0x490"),
    (0x000000000600094c, 0x000fea0003800000, "@P0 JMX R6"),            # off=0 omitted
    (0xfffffffc06fc094c, 0x000fea0003800000, "@P0 JMX R6 0xfffffffff0"),
    (0x0000000004000959, 0x000fea000b800000, "@P0 JMXU UR4"),
    (0x0000000104000959, 0x000fea000b800000, "@P0 JMXU.U UR4"),
    (0x0000000204000959, 0x000fea000b800000, "@P0 JMXU.DIV UR4"),
    (0x0000000304000959, 0x000fea000b800000, "@P0 JMXU.CONV UR4"),
]


def main():
    ok = 0
    for lo, hi, exp in VECTORS:
        got = decode(lo, hi)
        ok += got == exp
        print("%s %-24s (exp %s)" % ("OK " if got == exp else "XX ", got, exp))
    print("\n%d/%d vectors matched" % (ok, len(VECTORS)))


if __name__ == "__main__":
    main()
