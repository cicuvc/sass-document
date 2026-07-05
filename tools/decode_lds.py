#!/usr/bin/env python3
"""LDS full decoder — all 3 variants, verifying encoding vs disassembly."""

from typing import Optional

SZ_NAMES = {0: "U8", 1: "S8", 2: "U16", 3: "S16", 4: "32", 5: "64", 6: "128"}
STRIDE_NAMES = {0: "X1", 1: "X4", 2: "X8", 3: "X16"}


def extract(lo64: int, hi64: int, bits: list[int]) -> int:
    val = 0
    for bit in bits:
        bv = ((hi64 >> (bit - 64)) if bit >= 64 else (lo64 >> bit)) & 1
        val = (val << 1) | bv
    return val


def get_opcode(lo64: int, hi64: int) -> int:
    return extract(lo64, hi64, [91] + list(range(11, -1, -1)))


def s24(val: int) -> int:
    if val & 0x800000:
        val -= 0x1000000
    return val


def decode_lds(lo64: int, hi64: int) -> Optional[str]:
    opc = get_opcode(lo64, hi64)
    if opc not in (0x984, 0x1984):
        return None

    uniform = (opc == 0x1984)

    pg = extract(lo64, hi64, [14, 13, 12])
    pg_not = extract(lo64, hi64, [15])
    rd = extract(lo64, hi64, [23, 22, 21, 20, 19, 18, 17, 16])
    sz = extract(lo64, hi64, [75, 74, 73])
    raw_off = extract(lo64, hi64, list(range(63, 39, -1)))
    offset = s24(raw_off)
    ra = extract(lo64, hi64, [31, 30, 29, 28, 27, 26, 25, 24])

    stride = 0
    urb = None
    if not uniform:
        if ra != 0xff:
            stride = extract(lo64, hi64, [79, 78])
        else:
            stride = 0
    else:
        stride = extract(lo64, hi64, [79, 78])
        urb = extract(lo64, hi64, [37, 36, 35, 34, 33, 32])

    parts = []
    if pg != 7:
        parts.append(f"@{'!' if pg_not else ''}P{pg}")

    mnem = "LDS"
    if sz != 4:
        mnem += f".{SZ_NAMES[sz]}"
    if stride != 0:
        mnem += f".{STRIDE_NAMES[stride]}"
    parts.append(mnem)

    rd_s = f"R{rd}" if rd != 0xff else "RZ"
    ra_s = f"R{ra}" if ra != 0xff else "RZ"

    if uniform and urb is not None:
        base = f"UR{urb}"
    else:
        base = ra_s

    if offset > 0:
        addr_s = f"[{base}+{offset:#x}]"
    elif offset < 0:
        addr_s = f"[{base}+{offset:#x}]"
    else:
        addr_s = f"[{base}]"

    parts.append(f"{rd_s}, {addr_s}")
    return " ".join(parts)


if __name__ == "__main__":
    test_vectors = [
        (0x000000001d100984, 0x000e640000000800, "@P0 LDS R16, [R29]"),
        (0x000100001d1b1984, 0x000e280000000800, "@P1 LDS R27, [R29+0x100]"),
        (0x000080001d190984, 0x000e620000000800, "@P0 LDS R25, [R29+0x80]"),
        (0xffdf80001b065984, 0x000e260000000800, "@P5 LDS R6, [R27+-0x2080]"),
        (0xffe080001b0c0984, 0x000e620000000800, "@P0 LDS R12, [R27+-0x1f80]"),
        (0xffe000001b121984, 0x000e660000000800, "@P1 LDS R18, [R27+-0x2000]"),
        (0x0000000014120984, 0x000eb00000000a00, "@P0 LDS.64 R18, [R20]"),
        (0x0000800014120984, 0x000e240000000a00, "@P0 LDS.64 R18, [R20+0x80]"),
        (0x0001000014120984, 0x000e240000000a00, "@P0 LDS.64 R18, [R20+0x100]"),
        (0x0001800014120984, 0x000e220000000a00, "@P0 LDS.64 R18, [R20+0x180]"),
        (0x0002000014141984, 0x000e280000000a00, "@P1 LDS.64 R20, [R20+0x200]"),
        (0xffdf800000140984, 0x000e280000000a00, "@P0 LDS.64 R20, [R0+-0x2080]"),
        (0xffe0000000121984, 0x000e680000000a00, "@P1 LDS.64 R18, [R0+-0x2000]"),
        (0xffe08000000c2984, 0x000ea80000000a00, "@P2 LDS.64 R12, [R0+-0x1f80]"),
        (0x0000000017100984, 0x000ea20000000c00, "@P0 LDS.128 R16, [R23]"),
        (0x00010000170c1984, 0x000e620000000c00, "@P1 LDS.128 R12, [R23+0x100]"),
        (0x0002000017100984, 0x000e240000000c00, "@P0 LDS.128 R16, [R23+0x200]"),
        (0xffdf000000100984, 0x000e620000000c00, "@P0 LDS.128 R16, [R0+-0x2100]"),
        (0xffe00000000c1984, 0x000e260000000c00, "@P1 LDS.128 R12, [R0+-0x2000]"),
        (0xffe1000000047984, 0x000e300000000c00, "LDS.128 R4, [R0+-0x1f00]"),
    ]

    all_ok = True
    for lo, hi, expected in test_vectors:
        result = decode_lds(lo, hi)
        status = "OK" if result == expected else "MISMATCH"
        if status != "OK":
            all_ok = False
        print(f"{lo:#018x} {hi:#018x}  =>  {result}")
        print(f"  expected: {expected}  [{status}]")

    print()
    if all_ok:
        print("ALL PASS")
    else:
        print("SOME FAILED")
