#!/usr/bin/env python3
"""STL full decoder — 4 variants, local memory (stack) store."""

from typing import Optional

SZ = {0:"U8",1:"S8",2:"U16",3:"S16",4:"32",5:"64",6:"128",7:"INVAL"}
COP_N = {0:"EF",1:"EN",2:"EL",3:"LU",4:"EU",5:"NA"}


def extract(lo64, hi64, bits):
    val = 0
    for bit in bits:
        bv = ((hi64 >> (bit - 64)) if bit >= 64 else (lo64 >> bit)) & 1
        val = (val << 1) | bv
    return val


def get_opcode(lo64, hi64):
    return extract(lo64, hi64, [91] + list(range(11, -1, -1)))


def s24(val):
    if val & 0x800000:
        val -= 0x1000000
    return val


def decode_stl(lo64: int, hi64: int) -> Optional[str]:
    opc = get_opcode(lo64, hi64)
    if opc not in (0x387, 0x1987):
        return None

    pg = extract(lo64, hi64, [14, 13, 12])
    pg_not = extract(lo64, hi64, [15])
    sz = extract(lo64, hi64, [75, 74, 73])
    cop = extract(lo64, hi64, [86, 85, 84])
    memdesc_flag = extract(lo64, hi64, [76])
    ra = extract(lo64, hi64, [31, 30, 29, 28, 27, 26, 25, 24])
    rb = extract(lo64, hi64, [39, 38, 37, 36, 35, 34, 33, 32])
    raw_off = extract(lo64, hi64, list(range(63, 39, -1)))
    offset = s24(raw_off)

    urb = None
    if opc == 0x1987:
        urb = extract(lo64, hi64, [37, 36, 35, 34, 33, 32])

    parts = []
    if pg != 7:
        parts.append(f"@{'!' if pg_not else ''}P{pg}")

    mnem = "STL"
    if cop != 1:
        mnem += f".{COP_N[cop]}"
    if sz != 4:
        mnem += f".{SZ[sz]}"
    parts.append(mnem)

    rb_s = f"R{rb}" if rb != 0xff else "RZ"

    if memdesc_flag == 1 and urb is not None:
        off_s = ""
        if offset > 0:
            off_s = f"+{offset:#x}"
        elif offset < 0:
            off_s = f"{offset:#x}"
        parts.append(f"desc[UR{urb}][R{ra}{off_s}], {rb_s}")
    elif urb is not None:
        if ra == 0xff:
            off_s = ""
            if offset > 0:
                off_s = f"+{offset:#x}"
            elif offset < 0:
                off_s = f"{offset:#x}"
            parts.append(f"[UR{urb}{off_s}], {rb_s}")
        else:
            off_s = ""
            if offset > 0:
                off_s = f"+{offset:#x}"
            elif offset < 0:
                off_s = f"{offset:#x}"
            parts.append(f"[R{ra} + UR{urb}{off_s}], {rb_s}")
    else:
        ra_s = f"R{ra}" if ra != 0xff else "RZ"
        if offset > 0:
            parts.append(f"[{ra_s}+{offset:#x}], {rb_s}")
        elif offset < 0:
            parts.append(f"[{ra_s}{offset:#x}], {rb_s}")
        else:
            parts.append(f"[{ra_s}], {rb_s}")

    return " ".join(parts)


if __name__ == "__main__":
    tests = [
        (0x0000000201007387, 0x002fee0000100a00, "STL.64 [R1], R2"),
        (0x0000080401007387, 0x0011ee0000100a00, "STL.64 [R1+0x8], R4"),
        (0x0000100601007387, 0x0045ee0000100a00, "STL.64 [R1+0x10], R6"),
        (0x0000180801007387, 0x0085ee0000100a00, "STL.64 [R1+0x18], R8"),
        (0x0000200a01007387, 0x0025e80000100a00, "STL.64 [R1+0x20], R10"),
    ]
    ok = 0
    for lo, hi, exp in tests:
        r = decode_stl(lo, hi)
        s = "OK" if r == exp else "MISMATCH"
        if r == exp: ok += 1
        print(f"{r}")
        if s != "OK":
            print(f"  expected: {exp}")
    print(f"\n{ok}/{len(tests)} PASS")
