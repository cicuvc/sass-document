#!/usr/bin/env python3
"""STS full decoder — all 3 variants, verifying encoding vs disassembly."""

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


def decode_sts(lo64: int, hi64: int) -> Optional[str]:
    opc = get_opcode(lo64, hi64)
    if opc not in (0x388, 0x1988):
        return None

    uniform = (opc == 0x1988)

    pg = extract(lo64, hi64, [14, 13, 12])
    pg_not = extract(lo64, hi64, [15])
    rb = extract(lo64, hi64, [39, 38, 37, 36, 35, 34, 33, 32])
    sz = extract(lo64, hi64, [75, 74, 73])
    raw_off = extract(lo64, hi64, list(range(63, 39, -1)))
    offset = s24(raw_off)
    ra = extract(lo64, hi64, [31, 30, 29, 28, 27, 26, 25, 24])

    stride = 0
    ur = None
    if not uniform:
        if ra != 0xff:
            stride = extract(lo64, hi64, [79, 78])
    else:
        stride = extract(lo64, hi64, [79, 78])
        ur = extract(lo64, hi64, [69, 68, 67, 66, 65, 64])

    parts = []
    if pg != 7:
        parts.append(f"@{'!' if pg_not else ''}P{pg}")

    mnem = "STS"
    if sz != 4:
        mnem += f".{SZ_NAMES[sz]}"
    if stride != 0:
        mnem += f".{STRIDE_NAMES[stride]}"
    parts.append(mnem)

    ra_s = f"R{ra}" if ra != 0xff else "RZ"
    rb_s = f"R{rb}" if rb != 0xff else "RZ"

    if uniform and ur is not None:
        base = f"UR{ur}"
    else:
        base = ra_s

    if offset > 0:
        addr_s = f"[{base}+{offset:#x}]"
    elif offset < 0:
        addr_s = f"[{base}+{offset:#x}]"
    else:
        addr_s = f"[{base}]"

    parts.append(f"{addr_s}, {rb_s}")
    return " ".join(parts)


if __name__ == "__main__":
    test_vectors = [
        (0x002080061a00c388, 0x0041e80000000800, "@!P4 STS [R26+0x2080], R6"),
        (0x000000151a009388, 0x0081e60000000800, "@!P1 STS [R26], R21"),
        (0x0020a0081400a388, 0x0101e80000000800, "@!P2 STS [R20+0x20a0], R8"),
        (0x000020191400b388, 0x0003e20000000800, "@!P3 STS [R20+0x20], R25"),
        (0x0020c0091800d388, 0x0041e80000000800, "@!P5 STS [R24+0x20c0], R9"),
        (0x0000401718008388, 0x0083e20000000800, "@!P0 STS [R24+0x40], R23"),
        (0x000060191c009388, 0x010fe20000000800, "@!P1 STS [R28+0x60], R25"),
        (0x0020e00b1c00b388, 0x0001e80000000800, "@!P3 STS [R28+0x20e0], R11"),
        (0x0021000c1700c388, 0x0041f20000000800, "@!P4 STS [R23+0x2100], R12"),
        (0x0000801317008388, 0x0081e80000000800, "@!P0 STS [R23+0x80], R19"),
        (0x0000a0161900b388, 0x0103e20000000800, "@!P3 STS [R25+0xa0], R22"),
        (0x0021200d16007388, 0x0201e40000000800, "STS [R22+0x2120], R13"),
        (0x0000c0191b009388, 0x0041e20000000800, "@!P1 STS [R27+0xc0], R25"),
        (0x0021401518007388, 0x0201e40000000800, "STS [R24+0x2140], R21"),
    ]

    all_ok = True
    for lo, hi, expected in test_vectors:
        result = decode_sts(lo, hi)
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
