#!/usr/bin/env python3
"""LDC full decoder — all 4 variants, verifying encoding vs disassembly."""

from typing import Optional

SZ_NAMES = {0: "U8", 1: "S8", 2: "U16", 3: "S16", 4: "32", 5: "64"}
AD_NAMES = {0: "IA", 1: "IL", 2: "IS", 3: "ISL"}


def extract(lo64: int, hi64: int, bits: list[int]) -> int:
    val = 0
    for bit in bits:
        bv = ((hi64 >> (bit - 64)) if bit >= 64 else (lo64 >> bit)) & 1
        val = (val << 1) | bv
    return val


def get_opcode(lo64: int, hi64: int) -> int:
    return extract(lo64, hi64, [91] + list(range(11, -1, -1)))


def decode_ldc(lo64: int, hi64: int) -> Optional[str]:
    opc = get_opcode(lo64, hi64)
    if opc not in (0xb82, 0x1582):
        return None

    pg = extract(lo64, hi64, [14, 13, 12])
    pg_not = extract(lo64, hi64, [15])
    rd = extract(lo64, hi64, [23, 22, 21, 20, 19, 18, 17, 16])
    sz = extract(lo64, hi64, [75, 74, 73])
    ad = extract(lo64, hi64, [79, 78])

    parts = []
    if pg != 7:
        parts.append(f"@{'!' if pg_not else ''}P{pg}")

    mnem = "LDC"
    if sz != 4:
        mnem += f".{SZ_NAMES[sz]}"
    if ad != 0:
        mnem += f".{AD_NAMES[ad]}"
    parts.append(mnem)

    bindless = (opc == 0x1582)

    if bindless:
        ur_addr = extract(lo64, hi64, [29, 28, 27, 26, 25, 24])
        offset = extract(lo64, hi64, [53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38])
        rb = extract(lo64, hi64, [71, 70, 69, 68, 67, 66, 65, 64])

        rb_s = f"R{rb}" if rb != 0xff else "RZ"
        offset_s = f"{offset:#x}"
        if rb == 0xff:
            addr_s = f"c[UR{ur_addr}][{offset_s}]"
        else:
            addr_s = f"c[UR{ur_addr}][{rb_s}+{offset_s}]"
    else:
        bank = extract(lo64, hi64, [58, 57, 56, 55, 54])
        offset = extract(lo64, hi64, [53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38])
        ra = extract(lo64, hi64, [31, 30, 29, 28, 27, 26, 25, 24])

        if ra == 0xff and offset == 0:
            addr_s = f"c[{bank:#x}][RZ]"
        elif ra == 0xff:
            addr_s = f"c[{bank:#x}][{offset:#x}]"
        elif offset == 0:
            addr_s = f"c[{bank:#x}][R{ra}]"
        else:
            addr_s = f"c[{bank:#x}][R{ra}+{offset:#x}]"

    rd_s = f"R{rd}" if rd != 0xff else "RZ"
    parts.append(f"{rd_s}, {addr_s}")

    return " ".join(parts)


if __name__ == "__main__":
    test_vectors = [
        (0x00000a00ff017b82, 0x000fe20000000800, "LDC R1, c[0x0][0x28]"),
        (0x00008600ff027b82, 0x000e620000000a00, "LDC.64 R2, c[0x0][0x218]"),
        (0x00008800ff047b82, 0x000e300000000a00, "LDC.64 R4, c[0x0][0x220]"),
        (0x00008a00ff067b82, 0x000eb00000000a00, "LDC.64 R6, c[0x0][0x228]"),
        (0x00008c00ff087b82, 0x000ee20000000a00, "LDC.64 R8, c[0x0][0x230]"),
        (0x00008e00ff0a7b82, 0x000e620000000a00, "LDC.64 R10, c[0x0][0x238]"),
        (0x00009000ff0c7b82, 0x000f220000000a00, "LDC.64 R12, c[0x0][0x240]"),
        (0x00009200ff0e7b82, 0x000f620000000a00, "LDC.64 R14, c[0x0][0x248]"),
        (0x00009400ff107b82, 0x000ee20000000a00, "LDC.64 R16, c[0x0][0x250]"),
        (0x00008400ff027b82, 0x000e240000000a00, "LDC.64 R2, c[0x0][0x210]"),
        (0x00000800ff007b82, 0x000e240000000800, "LDC R0, c[0x0][0x20]"),
        (0x00009400ff163b82, 0x000e640000000a00, "@P3 LDC.64 R22, c[0x0][0x250]"),
        (0x00c00000ff057b82, 0x000e300000000800, "LDC R5, c[0x3][RZ]"),
    ]

    all_ok = True
    for lo, hi, expected in test_vectors:
        result = decode_ldc(lo, hi)
        status = "OK" if result == expected else f"MISMATCH"
        if status != "OK":
            all_ok = False
        print(f"{lo:#018x} {hi:#018x}  =>  {result}")
        print(f"  expected: {expected}  [{status}]")

    print()
    if all_ok:
        print("ALL PASS")
    else:
        print("SOME FAILED")
