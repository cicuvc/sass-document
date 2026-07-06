#!/usr/bin/env python3
"""Decoder for the sm_90 BRA instruction (relative branch, opcode 0x947).

Main CLASS bra__CONV_DIV layout (128-bit):
  opcode  = {bit[91], bits[11:0]}           (13-bit); base BRA bit91=0
  Pg      = bits[14:12], Pg_not = bit[15]   guard predicate  ('@!Px')
  Pp/Pnz  = bits[89:87], Pp_not = bit[90]   divergence predicate operand (printed if != PT)
  cond    = bits[33:32]  COND__DIV_CONV     0=none, 2=.DIV, 3=.CONV
  depth   = bits[86:85]  DEPTH              0=none, 1=.INC, 2=.DEC (call-depth adjust)
  sImm    = bits[81:34] (hi 48) ∥ bits[23:16] (lo 8) = 56-bit signed, SCALE 4
            target = PC + 0x10 + sImm*4     (relative to the next instruction)

Usage: python3 decode_bra.py            (built-in self-test vectors)
       python3 decode_bra.py <sass.txt> (validate against every BRA in a dump)
"""
import re
import sys


def bits(val, hi, lo):
    return (val >> lo) & ((1 << (hi - lo + 1)) - 1)


def sx(val, width):
    return val - (1 << width) if val & (1 << (width - 1)) else val


def pred(idx, neg):
    name = "PT" if idx == 7 else "P%d" % idx
    return ("!" if neg else "") + name


def decode(lo64, hi64, pc):
    inst = lo64 | (hi64 << 64)
    opcode = (bits(inst, 91, 91) << 12) | bits(inst, 11, 0)

    pg, pg_not = bits(inst, 14, 12), bits(inst, 15, 15)
    pp, pp_not = bits(inst, 89, 87), bits(inst, 90, 90)
    cond = bits(inst, 33, 32)
    depth = bits(inst, 86, 85)

    sa = sx((bits(inst, 81, 34) << 8) | bits(inst, 23, 16), 56)
    target = (pc + 0x10) + sa * 4

    guard = "" if (pg == 7 and pg_not == 0) else "@%s " % pred(pg, pg_not)
    depth_s = {0: "", 1: ".INC", 2: ".DEC", 3: ".INVALID3"}[depth]
    cond_s = {0: "", 1: "", 2: ".DIV", 3: ".CONV"}[cond]
    pp_s = "" if (pp == 7 and pp_not == 0) else "%s, " % pred(pp, pp_not)

    # extra operand by variant (distinguished by full 13-bit opcode)
    extra = ""
    if opcode == 0x1947:            # bra_uniform_: [~]URb at [30]/[29:24]
        urb = bits(inst, 29, 24)
        inv = "~" if bits(inst, 30, 30) else ""
        urb_s = "URZ" if urb == 63 else "UR%d" % urb
        extra = "%s%s, " % (inv, urb_s)
    elif opcode == 0x1547:          # bra_uniform_pred_: [!]UPq at [27]/[26:24]
        upq = bits(inst, 26, 24)
        neg = "!" if bits(inst, 27, 27) else ""
        upq_s = "UPT" if upq == 7 else "UP%d" % upq
        extra = "%s%s, " % (neg, upq_s)

    return "%sBRA%s%s %s%s0x%x" % (
        guard, cond_s, depth_s, pp_s, extra, target & 0xffffffffffff)


# (pc, lo64, hi64, expected) — from tests/*_test.cu dumps + libcublas
VECTORS = [
    (0x00c0, 0x0000000000208947, 0x000fea0003800000, "@!P0 BRA 0x150"),
    (0x0120, 0x00000000000c7947, 0x000fec0003800000, "BRA 0x160"),
    (0x0170, 0xfffffffc00e08947, 0x000fea000083ffff, "@!P0 BRA P1, 0x100"),
    (0x01e0, 0xfffffffc00d08947, 0x000fea000083ffff, "@!P0 BRA P1, 0x130"),
    (0x0240, 0xfffffffc00a08947, 0x000fea000083ffff, "@!P0 BRA P1, 0xd0"),
    (0x0190, 0x00000000002c8947, 0x000fea0003800000, "@!P0 BRA 0x250"),
    (0x01c0, 0xfffffffc00fc7947, 0x000fc0000383ffff, "BRA 0x1c0"),
    # bra_uniform_ (0x1947): BRA.DIV URb, target  (from libcublas)
    (0x14b0, 0x0000001207e87947, 0x000fea000b800000, "BRA.DIV UR7, 0x2860"),
    (0x1100, 0x0000001e05887947, 0x000fea000b800000, "BRA.DIV UR5, 0x2f30"),
]


def run_vectors():
    ok = 0
    for pc, lo, hi, exp in VECTORS:
        got = decode(lo, hi, pc)
        flag = "OK " if got == exp else "XX "
        ok += got == exp
        print("%s pc=0x%04x  %-22s  (expected %s)" % (flag, pc, got, exp))
    print("\n%d/%d vectors matched" % (ok, len(VECTORS)))


# Parse a cuobjdump -sass dump: mnemonic line has /*PC*/ ... ; /* lo64 */
# and the following line has /* hi64 */.
LINE = re.compile(r"/\*([0-9a-f]+)\*/\s+(.*?);\s*/\*\s*([0-9a-fx]+)\s*\*/")
HEX = re.compile(r"/\*\s*([0-9a-fx]+)\s*\*/")


def validate_dump(path):
    with open(path) as f:
        lines = f.readlines()
    total = ok = 0
    for i, ln in enumerate(lines):
        m = LINE.search(ln)
        if not m or " BRA" not in (" " + m.group(2)):
            continue
        pc = int(m.group(1), 16)
        text = m.group(2).strip()
        lo = int(m.group(3), 16)
        hm = HEX.search(lines[i + 1]) if i + 1 < len(lines) else None
        if not hm:
            continue
        hi = int(hm.group(1), 16)
        got = decode(lo, hi, pc)
        total += 1
        if got == text:
            ok += 1
        else:
            print("XX pc=0x%04x got %-24s exp %s  [lo=%016x hi=%016x]"
                  % (pc, got, text, lo, hi))
    print("%s: %d/%d BRA matched" % (path, ok, total))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            validate_dump(p)
    else:
        run_vectors()
