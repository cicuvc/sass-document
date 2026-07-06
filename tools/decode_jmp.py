#!/usr/bin/env python3
"""Decoder for the sm_90 JMP instruction (absolute / constant-bank jump, opcode *0x?4a).

JMP has two target families x three uniform forms (12 CLASSes, 6 distinct opcodes):
  imm  base            0x094a   JMP{cond} {Pp,} 0x<TARGET>          (absolute = Sa*4)
  imm  uniform         0x194a   JMP{cond} {Pp,} [~]URb, 0x<TARGET>
  imm  uniform_pred    0x154a   JMP{cond} {Pp,} [!]UPq, 0x<TARGET>
  const base           0x0b4a   JMP{cond} {Pp,} c[bank][off*4]
  const uniform        0x1b4a   JMP{cond} {Pp,} [~]URb, c[bank][off*4]
  const uniform_pred   0x174a   JMP{cond} {Pp,} [!]UPq, c[bank][off*4]

Fields (128-bit):
  opcode = {bit[91], bits[11:0]}
  Pg=[14:12] Pg_not=[15]      guard predicate
  Pp/Pnz=[89:87] Pp_not=[90]  divergence predicate (printed if != PT)
  cond=[33:32]                enum depends on variant (see COND_MAPS)
  imm target Sa = bits[80:34] (hi 47) ∥ bits[23:16] (lo 8); ABSOLUTE addr = Sa*4
  const  bank = bits[58:54];  byte offset = bits[53:40] * 4
  uniform reg  URb=[29:24] invert=[30];  uniform pred UPq=[26:24] not=[27]

JMP is not emitted by ptxas for typical workloads; ground-truth vectors below were
produced by patching a real cubin instruction and reading nvdisasm's rendering.
Usage: python3 decode_jmp.py            (self-test)
       python3 decode_jmp.py <sass.txt> (validate every JMP in a dump)
"""
import re
import sys


def bits(v, hi, lo):
    return (v >> lo) & ((1 << (hi - lo + 1)) - 1)


def pred(idx, neg, t="P"):
    name = ("%sT" % t) if idx == 7 else "%s%d" % (t, idx)
    return ("!" if neg else "") + name


# [33:32] cond enum differs per variant family. At the bit91=0 opcodes both the
# __CONV_DIV and __U classes coexist, so the field decodes across both enums:
#   0->nocond, 1->UONLY "U", 2->DIV, 3->CONV
COND_BASE = {0: "", 1: ".U", 2: ".DIV", 3: ".CONV"}
COND_UREG = {2: ".DIV", 3: ".CONV"}                  # COND_DIV_CONV_jmp (0,1 invalid)
COND_UPRED = {1: ".U"}                               # UONLY (uniform_pred only)


def cond_str(cond, table):
    return table.get(cond, ".???%d" % cond)


def decode(lo64, hi64, pc):
    inst = lo64 | (hi64 << 64)
    opcode = (bits(inst, 91, 91) << 12) | bits(inst, 11, 0)

    pg, pg_not = bits(inst, 14, 12), bits(inst, 15, 15)
    pp, pp_not = bits(inst, 89, 87), bits(inst, 90, 90)
    cond = bits(inst, 33, 32)

    is_const = bits(inst, 9, 9) == 1          # 0xb4a vs 0x94a differ in bit9
    guard = "" if (pg == 7 and pg_not == 0) else "@%s " % pred(pg, pg_not)
    pp_s = "" if (pp == 7 and pp_not == 0) else "%s, " % pred(pp, pp_not)

    # target string
    if is_const:
        bank = bits(inst, 58, 54)
        off = bits(inst, 53, 40)
        if off & (1 << 13):                   # 14-bit signed field, byte = field*4
            off -= (1 << 14)
        offb = off * 4
        tgt = "c[%#x][%s%#x]" % (bank, "-" if offb < 0 else "", abs(offb))
    else:
        sa = (bits(inst, 80, 34) << 8) | bits(inst, 23, 16)
        tgt = "0x%x" % ((sa * 4) & 0xffffffffffff)

    # variant-specific extra operand + cond enum
    b91 = bits(inst, 91, 91)
    low = bits(inst, 11, 0)
    extra = ""
    ctab = COND_BASE
    if b91 == 1 and low in (0x94a, 0xb4a):        # *_uniform_ (URb)
        ctab = COND_UREG
        urb = bits(inst, 29, 24)
        inv = "~" if bits(inst, 30, 30) else ""
        extra = "%s%s, " % (inv, "URZ" if urb == 63 else "UR%d" % urb)
    elif b91 == 1 and low in (0x54a, 0x74a):      # *_uniform_pred_ (UPq)
        ctab = COND_UPRED
        upq = bits(inst, 26, 24)
        neg = "!" if bits(inst, 27, 27) else ""
        extra = "%s%s, " % (neg, "UPT" if upq == 7 else "UP%d" % upq)

    return "%sJMP%s %s%s%s" % (guard, cond_str(cond, ctab), pp_s, extra, tgt)


# (pc, lo64, hi64, expected) — ground truth from cubin-patch + nvdisasm
VECTORS = [
    (0xe0, 0x000000040024094a, 0x000fea0003800000, "@P0 JMP 0x490"),
    (0xe0, 0x0000000400240b4a, 0x000fea0003800000, "@P0 JMP c[0x0][0x0]"),
    (0xe0, 0x000000040024094a, 0x000fea000b800000, "@P0 JMP.???0 UR0, 0x490"),
    (0xe0, 0x000000060024094a, 0x000fea000b800000, "@P0 JMP.DIV UR0, 0x490"),
    (0xe0, 0x000000060024054a, 0x000fea000b800000, "@P0 JMP.???2 UP0, 0x490"),
    (0xe0, 0x0000000600240b4a, 0x000fea000b800000, "@P0 JMP.DIV UR0, c[0x0][0x0]"),
    (0xe0, 0x0000010000000b4a, 0x000fea0003800000, "@P0 JMP c[0x0][0x4]"),
    (0xe0, 0x0000400000000b4a, 0x000fea0003800000, "@P0 JMP c[0x0][0x100]"),
    (0xe0, 0x0000410000000b4a, 0x000fea0003800000, "@P0 JMP c[0x0][0x104]"),
    (0xe0, 0x0080800000000b4a, 0x000fea0003800000, "@P0 JMP c[0x2][0x200]"),
]


def run_vectors():
    ok = 0
    for pc, lo, hi, exp in VECTORS:
        got = decode(lo, hi, pc)
        ok += got == exp
        print("%s pc=0x%02x  %-30s (exp %s)"
              % ("OK " if got == exp else "XX ", pc, got, exp))
    print("\n%d/%d vectors matched" % (ok, len(VECTORS)))


LINE = re.compile(r"/\*([0-9a-f]+)\*/\s+(.*?);\s*/\*\s*([0-9a-fx]+)\s*\*/")
HEX = re.compile(r"/\*\s*([0-9a-fx]+)\s*\*/")


def validate_dump(path):
    lines = open(path).readlines()
    total = ok = 0
    for i, ln in enumerate(lines):
        m = LINE.search(ln)
        if not m or " JMP" not in (" " + m.group(2)):
            continue
        pc, text, lo = int(m.group(1), 16), m.group(2).strip(), int(m.group(3), 16)
        hm = HEX.search(lines[i + 1]) if i + 1 < len(lines) else None
        if not hm:
            continue
        hi = int(hm.group(1), 16)
        got = decode(lo, hi, pc)
        total += 1
        ok += got == text
        if got != text:
            print("XX pc=0x%x got %-28s exp %s [%016x %016x]" % (pc, got, text, lo, hi))
    print("%s: %d/%d JMP matched" % (path, ok, total))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            validate_dump(p)
    else:
        run_vectors()
