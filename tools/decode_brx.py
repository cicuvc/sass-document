#!/usr/bin/env python3
"""Decoder for the sm_90 indirect branches BRX/JMX (GPR) and BRXU/JMXU (uniform GPR).

  BRX  0x0949  @Pg BRX{.INC/.DEC} {Pp,} R<Ra> [0x<off>]     Ra=[31:24] (GPR pair)
  JMX  0x094c  @Pg JMX{.INC/.DEC} {Pp,} R<Ra> [0x<off>]     (absolute-indirect twin of BRX)
  BRXU 0x1958  @Pg BRXU{cond} {Pp,} UR<URa> [0x<off>]       URa=[29:24], cond=[33:32]
  JMXU 0x1959  @Pg JMXU{cond} {Pp,} UR<URa> [0x<off>]       (absolute-indirect twin of BRXU)

Common fields (128-bit):
  opcode = {bit[91], bits[11:0]}
  Pg=[14:12] Pg_not=[15]         guard predicate
  Pp/Pnz=[89:87] Pp_not=[90]     divergence predicate (printed if != PT)
  depth=[86:85] DEPTH            0=none,1=.INC,2=.DEC
  cond=[33:32]  COND (U forms)   0=none,1=.U,2=.DIV,3=.CONV
  sImm = bits[81:34]∥bits[23:16] (56-bit signed), SCALE 4
         printed offset = (sImm*4) & 0xffffffffff, OMITTED when sImm==0

None of these is emitted by ptxas for the sampled workloads; ground-truth vectors
were produced by patching a real cubin instruction and reading nvdisasm.
Usage: python3 decode_brx.py            (self-test)
       python3 decode_brx.py <sass.txt> (validate every BRX/JMX/BRXU/JMXU in a dump)
"""
import re
import sys

ADDR_MASK = (1 << 40) - 1
DEPTH = {0: "", 1: ".INC", 2: ".DEC", 3: ".???3"}
COND = {0: "", 1: ".U", 2: ".DIV", 3: ".CONV"}
# opcode -> (mnemonic, uses_uniform_reg)
OPC = {0x949: ("BRX", False), 0x94c: ("JMX", False),
       0x1958: ("BRXU", True), 0x1959: ("JMXU", True)}


def bits(v, hi, lo):
    return (v >> lo) & ((1 << (hi - lo + 1)) - 1)


def sx(v, w):
    return v - (1 << w) if v & (1 << (w - 1)) else v


def pred(idx, neg):
    return ("!" if neg else "") + ("PT" if idx == 7 else "P%d" % idx)


def decode(lo64, hi64, pc=0):
    inst = lo64 | (hi64 << 64)
    opcode = (bits(inst, 91, 91) << 12) | bits(inst, 11, 0)
    if opcode not in OPC:
        return "?opcode 0x%x" % opcode
    mnem, uniform = OPC[opcode]

    pg, pg_not = bits(inst, 14, 12), bits(inst, 15, 15)
    pp, pp_not = bits(inst, 89, 87), bits(inst, 90, 90)
    depth = bits(inst, 86, 85)
    simm = sx((bits(inst, 81, 34) << 8) | bits(inst, 23, 16), 56)
    off = (simm * 4) & ADDR_MASK

    guard = "" if (pg == 7 and pg_not == 0) else "@%s " % pred(pg, pg_not)
    pp_s = "" if (pp == 7 and pp_not == 0) else "%s, " % pred(pp, pp_not)
    off_s = "" if simm == 0 else " 0x%x" % off

    if uniform:
        ura = bits(inst, 29, 24)
        reg = "URZ" if ura == 63 else "UR%d" % ura
        name = mnem + DEPTH[depth] + COND[bits(inst, 33, 32)]
    else:
        ra = bits(inst, 31, 24)
        reg = "RZ" if ra == 0xff else "R%d" % ra
        name = mnem + DEPTH[depth]

    return "%s%s %s%s%s" % (guard, name, pp_s, reg, off_s)


# (lo64, hi64, expected) — ground truth from cubin-patch + nvdisasm
VECTORS = [
    (0x0000000404240949, 0x000fea0003800000, "@P0 BRX R4 0x490"),
    (0xfffffffc04fc0949, 0x000fea0003800000, "@P0 BRX R4 0xfffffffff0"),
    (0x0000000006400949, 0x000fea0001800000, "@P0 BRX P3, R6 0x100"),
    (0x0000000006400949, 0x000fea0003a00000, "@P0 BRX.INC R6 0x100"),
    (0x0000000004000958, 0x000fea000b800000, "@P0 BRXU UR4"),
    (0x0000000204000958, 0x000fea000b800000, "@P0 BRXU.DIV UR4"),
    (0x0000000304000958, 0x000fea000b800000, "@P0 BRXU.CONV UR4"),
]


def run_vectors():
    ok = 0
    for lo, hi, exp in VECTORS:
        got = decode(lo & 0xffffffffffffffff, hi)
        ok += got == exp
        print("%s %-26s (exp %s)" % ("OK " if got == exp else "XX ", got, exp))
    print("\n%d/%d vectors matched" % (ok, len(VECTORS)))


LINE = re.compile(r"/\*([0-9a-f]+)\*/\s+(.*?);\s*/\*\s*([0-9a-fx]+)\s*\*/")
HEX = re.compile(r"/\*\s*([0-9a-fx]+)\s*\*/")


def validate_dump(path):
    lines = open(path).readlines()
    total = ok = 0
    for i, ln in enumerate(lines):
        m = LINE.search(ln)
        if not m:
            continue
        text = m.group(2).strip()
        if not re.search(r"\b(BRXU?|JMXU?)\b", text):
            continue
        lo = int(m.group(3), 16)
        hm = HEX.search(lines[i + 1]) if i + 1 < len(lines) else None
        if not hm:
            continue
        hi = int(hm.group(1), 16)
        got = decode(lo, hi, int(m.group(1), 16))
        total += 1
        ok += got == text
        if got != text:
            print("XX got %-26s exp %-26s [%016x %016x]" % (got, text, lo, hi))
    print("%s: %d/%d BRX/BRXU matched" % (path, ok, total))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            validate_dump(p)
    else:
        run_vectors()
