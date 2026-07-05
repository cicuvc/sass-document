#!/usr/bin/env python3
"""LDL full decoder — 4 variants, local memory (stack) load."""

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


def decode_ldl(lo64: int, hi64: int) -> Optional[str]:
    opc = get_opcode(lo64, hi64)
    if opc not in (0x983, 0x1983):
        return None

    pg = extract(lo64, hi64, [14, 13, 12])
    pg_not = extract(lo64, hi64, [15])
    rd = extract(lo64, hi64, [23, 22, 21, 20, 19, 18, 17, 16])
    sz = extract(lo64, hi64, [75, 74, 73])
    cop = extract(lo64, hi64, [86, 85, 84])
    memdesc_flag = extract(lo64, hi64, [76])
    ra = extract(lo64, hi64, [31, 30, 29, 28, 27, 26, 25, 24])
    raw_off = extract(lo64, hi64, list(range(63, 39, -1)))
    offset = s24(raw_off)

    urb = None
    if opc == 0x1983:
        urb = extract(lo64, hi64, [37, 36, 35, 34, 33, 32])

    parts = []
    if pg != 7:
        parts.append(f"@{'!' if pg_not else ''}P{pg}")

    mnem = "LDL"
    if cop != 1:
        mnem += f".{COP_N[cop]}"
    if sz != 4:
        mnem += f".{SZ[sz]}"
    parts.append(mnem)

    rd_s = f"R{rd}" if rd != 0xff else "RZ"

    if memdesc_flag == 1 and urb is not None:
        off_s = ""
        if offset > 0:
            off_s = f"+{offset:#x}"
        elif offset < 0:
            off_s = f"{offset:#x}"
        parts.append(f"{rd_s}, desc[UR{urb}][R{ra}{off_s}]")
    elif urb is not None:
        if ra == 0xff:
            off_s = ""
            if offset > 0:
                off_s = f"+{offset:#x}"
            elif offset < 0:
                off_s = f"{offset:#x}"
            parts.append(f"{rd_s}, [UR{urb}{off_s}]")
        else:
            off_s = ""
            if offset > 0:
                off_s = f"+{offset:#x}"
            elif offset < 0:
                off_s = f"{offset:#x}"
            parts.append(f"{rd_s}, [R{ra} + UR{urb}{off_s}]")
    else:
        ra_s = f"R{ra}" if ra != 0xff else "RZ"
        if offset > 0:
            parts.append(f"{rd_s}, [{ra_s}+{offset:#x}]")
        elif offset < 0:
            parts.append(f"{rd_s}, [{ra_s}{offset:#x}]")
        else:
            parts.append(f"{rd_s}, [{ra_s}]")

    return " ".join(parts)


if __name__ == "__main__":
    tests = [
        (0x0000000006007983, 0x000ea80000100800, "LDL R0, [R6]"),
        (0x0000040006037983, 0x000ea80000100800, "LDL R3, [R6+0x4]"),
        (0x0000000001367983, 0x000ea20000300a00, "LDL.LU.64 R54, [R1]"),
        (0x0000000001367983, 0x0001620000100a00, "LDL.64 R54, [R1]"),
        (0x0000000001527983, 0x000ea80000100a00, "LDL.64 R82, [R1]"),
        (0x0000080001507983, 0x000ee80000100a00, "LDL.64 R80, [R1+0x8]"),
    ]
    ok = 0
    for lo, hi, exp in tests:
        r = decode_ldl(lo, hi)
        s = "OK" if r == exp else "MISMATCH"
        if r == exp: ok += 1
        print(f"{r}")
        if s != "OK":
            print(f"  expected: {exp}")
    print(f"\n{ok}/{len(tests)} PASS")
