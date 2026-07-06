#!/usr/bin/env python3
"""Decoder for the sm_90 convergence-barrier control instructions BSSY / BSYNC / BREAK.

All three share the CBU base layout:
  opcode  = {bit[91], bits[11:0]}   (13-bit)
  Pg      = bits[14:12]  guard predicate  (7 = PT)
  Pg_not  = bit[15]      guard negate  ('@!Px')
  Pp      = bits[89:87]  second predicate operand (7 = PT, printed only if != PT)
  Pp_not  = bit[90]
  barReg  = bits[19:16]  convergence-barrier register B0..B15
BSSY additionally carries a branch target:
  Sa      = bits[63:34]  (30-bit signed), SCALE 4
            target = PC_next + Sa*4   (PC_next = pc + 0x10)

Usage: python3 decode_bssy.py    (runs the built-in self-test vectors)
"""

OPC = {0x945: "BSSY", 0x941: "BSYNC", 0x942: "BREAK"}


def bits(val, hi, lo):
    return (val >> lo) & ((1 << (hi - lo + 1)) - 1)


def sign_extend(val, width):
    if val & (1 << (width - 1)):
        val -= (1 << width)
    return val


def decode(lo64, hi64, pc):
    inst = lo64 | (hi64 << 64)

    opcode = (bits(inst, 91, 91) << 12) | bits(inst, 11, 0)
    mnem = OPC.get(opcode, "?0x%03x" % opcode)

    pg = bits(inst, 14, 12)
    pg_not = bits(inst, 15, 15)
    pp = bits(inst, 89, 87)
    pp_not = bits(inst, 90, 90)
    barreg = bits(inst, 19, 16)

    # guard predicate prefix
    if pg == 7 and pg_not == 0:
        guard = ""
    else:
        guard = "@%s%s " % ("!" if pg_not else "", "PT" if pg == 7 else "P%d" % pg)

    # Pp operand (printed only when not the default PT)
    pp_str = ""
    if not (pp == 7 and pp_not == 0):
        pp_str = "%s%s, " % ("!" if pp_not else "", "PT" if pp == 7 else "P%d" % pp)

    text = "%s%s " % (guard, mnem)

    if mnem == "BSSY":
        sa = sign_extend(bits(inst, 63, 34), 30)
        target = (pc + 0x10) + sa * 4
        text += "%sB%d, 0x%x" % (pp_str, barreg, target & 0xffffffffffff)
    else:  # BSYNC / BREAK
        text += "%sB%d" % (pp_str, barreg)

    return text.rstrip()


# (pc, lo64, hi64, expected cuobjdump rendering)
VECTORS = [
    # cublas libcublas.so
    (0x0960, 0x0000023000007945, 0x000fe20003800000, "BSSY B0, 0xba0"),
    (0x0d00, 0x0000012000007945, 0x000ff20003800000, "BSSY B0, 0xe30"),
    # tests/bssy_test.cu (switch kernel)
    (0x0090, 0x000000d000007945, 0x000fe20003800000, "BSSY B0, 0x170"),
    (0x0160, 0x0000000000007941, 0x000fea0003800000, "BSYNC B0"),
    # tests/bssy_break_test.cu (nested loops + break)
    (0x00b0, 0x000001a000007945, 0x000fe20003800000, "BSSY B0, 0x260"),
    (0x00f0, 0x0000010000017945, 0x000fe20003800000, "BSSY B1, 0x200"),
    (0x0180, 0x0000000000018942, 0x000fea0003800000, "@!P0 BREAK B1"),
    (0x01f0, 0x0000000000017941, 0x000fea0003800000, "BSYNC B1"),
    (0x0250, 0x0000000000007941, 0x000fea0003800000, "BSYNC B0"),
]


def main():
    ok = 0
    for pc, lo, hi, exp in VECTORS:
        got = decode(lo, hi, pc)
        flag = "OK " if got == exp else "XX "
        if got == exp:
            ok += 1
        print("%s pc=0x%04x  %-22s  (expected %s)" % (flag, pc, got, exp))
    print("\n%d/%d vectors matched" % (ok, len(VECTORS)))


if __name__ == "__main__":
    main()
