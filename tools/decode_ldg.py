#!/usr/bin/env python3
"""LDG full decoder — 6 variants, verifying encoding vs disassembly."""

from typing import Optional

SZ = {0:"U8",1:"S8",2:"U16",3:"S16",4:"32",5:"64",6:"128",7:"INVALID"}
COP = {0:"EF",1:"EN",2:"EL",3:"LU",4:"EU",5:"NA",6:"INV6",7:"INV7"}
SP2 = {0:"nosp2",1:"LTC64B",2:"LTC128B",3:"LTC256B"}

MEM_SEM = {0: "CONSTANT", 1: "WEAK", 2: "STRONG", 3: "MMIO"}
MEM_SCO = {0: "nosco", 1: "CTA", 2: "SM", 3: "VC", 4: "GPU", 5: "SYS"}
MEM_PRIV = {0: "noprivate", 1: "PRIVATE"}

MEM_REV = {
    0:  (1,0,0),  4: (0,0,0),  1: (0,0,1),  2: (0,1,0),  3: (0,1,1),
    5:  (2,2,0),  9: (0,2,0), 10: (2,5,0), 11: (0,2,1), 12: (3,5,0),
    6:  (2,4,1),  7: (2,4,0),  8: (3,4,0), 13: (0,3,0), 14: (0,3,1),
}


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


def decode_ldg(lo64: int, hi64: int) -> Optional[str]:
    opc = get_opcode(lo64, hi64)
    if opc not in (0x381, 0x1981):
        return None

    pg = extract(lo64, hi64, [14, 13, 12])
    pg_not = extract(lo64, hi64, [15])
    rd = extract(lo64, hi64, [23, 22, 21, 20, 19, 18, 17, 16])
    sz = extract(lo64, hi64, [75, 74, 73])
    e = extract(lo64, hi64, [72])
    cop = extract(lo64, hi64, [86, 85, 84])
    sp2 = extract(lo64, hi64, [69, 68])
    mem = extract(lo64, hi64, [80, 79, 78, 77])
    memdesc = extract(lo64, hi64, [76])
    pu = extract(lo64, hi64, [83, 82, 81])
    pnz_enc = extract(lo64, hi64, [67, 66, 65, 64])
    ra = extract(lo64, hi64, [31, 30, 29, 28, 27, 26, 25, 24])
    raw_off = extract(lo64, hi64, list(range(63, 39, -1)))
    offset = s24(raw_off)

    has_urb = (opc == 0x1981)
    urb = None
    if has_urb:
        urb = extract(lo64, hi64, [37, 36, 35, 34, 33, 32])

    pnz_not_bit = pnz_enc >> 3
    if pnz_not_bit == 0:
        pnz_pred = 7 - (pnz_enc & 0x7)
    else:
        pnz_pred = 15 - pnz_enc

    parts = []
    if pg != 7:
        parts.append(f"@{'!' if pg_not else ''}P{pg}")

    mnem = "LDG"
    if e == 1:
        mnem += ".E"
    if cop != 1:
        mnem += f".{COP[cop]}"
    if sp2 != 0:
        mnem += f".{SP2[sp2]}"
    if sz != 4:
        mnem += f".{SZ[sz]}"

    mem_info = MEM_REV.get(mem)
    if mem_info:
        ms, mc, mp = mem_info
        if ms != 1:
            mnem += f".{MEM_SEM[ms]}"
        if mc != 0:
            mnem += f".{MEM_SCO[mc]}"
        if mp != 0:
            mnem += f".{MEM_PRIV[mp]}"

    parts.append(mnem)

    if pu != 7:
        parts.append(f"P{pu},")
        parts.append(f"R{rd},")
    else:
        parts.append(f"R{rd},")

    if memdesc == 1:
        off_s = ""
        if offset > 0:
            off_s = f"+{offset:#x}"
        elif offset < 0:
            off_s = f"{offset:#x}"
        e64 = ".64" if e == 1 else ""
        parts.append(f"desc[UR{urb}][R{ra}{e64}{off_s}]")
    elif has_urb and urb is not None:
        e64 = ".64" if e == 1 else ""
        if ra == 0xff:
            off_s = ""
            if offset > 0:
                off_s = f"+{offset:#x}"
            elif offset < 0:
                off_s = f"{offset:#x}"
            parts.append(f"[UR{urb}{off_s}]")
        else:
            off_s = ""
            if offset > 0:
                off_s = f"+{offset:#x}"
            elif offset < 0:
                off_s = f"{offset:#x}"
            parts.append(f"[R{ra}{e64} + UR{urb}{off_s}]")
    else:
        ra_s = f"R{ra}" if ra != 0xff else "RZ"
        if offset > 0:
            parts.append(f"[{ra_s}+{offset:#x}]")
        elif offset < 0:
            parts.append(f"[{ra_s}{offset:#x}]")
        else:
            parts.append(f"[{ra_s}]")

    if pnz_pred != 7:
        parts.append(f"{'!' if pnz_not_bit else ''}P{pnz_pred}")

    return " ".join(parts)


if __name__ == "__main__":
    test_vectors = [
        (0x0000800812068981, 0x000ea2000c1e9900,
         "@!P0 LDG.E.CONSTANT R6, desc[UR8][R18.64+0x80]"),
        (0x0000000812159981, 0x000ee2000c1e9900,
         "@!P1 LDG.E.CONSTANT R21, desc[UR8][R18.64]"),
        (0x000080081019b981, 0x000f22000c1e9900,
         "@!P3 LDG.E.CONSTANT R25, desc[UR8][R16.64+0x80]"),
        (0x0001000812068981, 0x001162000c1e9900,
         "@!P0 LDG.E.CONSTANT R6, desc[UR8][R18.64+0x100]"),
        (0x000080081209a981, 0x000ea2000c1e9900,
         "@!P2 LDG.E.CONSTANT R9, desc[UR8][R18.64+0x80]"),
        (0x0000000812178981, 0x000ee2000c1e9900,
         "@!P0 LDG.E.CONSTANT R23, desc[UR8][R18.64]"),
        (0x0001000810089981, 0x001162000c1e9900,
         "@!P1 LDG.E.CONSTANT R8, desc[UR8][R16.64+0x100]"),
        (0x0000000810199981, 0x002f2c000c1e9900,
         "@!P1 LDG.E.CONSTANT R25, desc[UR8][R16.64]"),
        (0x00008008100bc981, 0x000f22000c1e9900,
         "@!P4 LDG.E.CONSTANT R11, desc[UR8][R16.64+0x80]"),
        (0x0000000402037981, 0x000ea2000c1e1900,
         "LDG.E R3, desc[UR4][R2.64]"),
    ]

    all_ok = True
    for lo, hi, expected in test_vectors:
        result = decode_ldg(lo, hi)
        status = "OK" if result == expected else "MISMATCH"
        if status != "OK":
            all_ok = False
        print(f"{result}")
        print(f"  expected: {expected}  [{status}]")

    print()
    if all_ok:
        print(f"ALL PASS ({len(test_vectors)}/{len(test_vectors)})")
    else:
        print("SOME FAILED")
