#!/usr/bin/env python3
"""STG full decoder — 6 variants, memdesc-based global store."""

from typing import Optional

SZ = {0:"U8",1:"S8",2:"U16",3:"S16",4:"32",5:"64",6:"128",7:"INVAL"}
COP = {0:"EF",1:"EN",2:"EL",3:"LU",4:"EU",5:"NA"}
SEM_n = {0:"CONSTANT",1:"WEAK",2:"STRONG",3:"MMIO"}
SCO_n = {0:"nosco",1:"CTA",2:"SM",3:"VC",4:"GPU",5:"SYS"}
PRIV_n = {0:"noprivate",1:"PRIVATE"}

MEM0_REV = {}
for sem_k, sem_v in SEM_n.items():
    for sco_k, sco_v in SCO_n.items():
        for prv_k, prv_v in PRIV_n.items():
            key = (sem_k << 8) | (sco_k << 4) | prv_k
            idx = None
            if key == 0x100: idx = 0    # WEAK,nosco,noprivate
            elif key == 0x001: idx = 1  # CONSTANT,nosco,PRIVATE
            elif key == 0x010: idx = 2  # CONSTANT,CTA,noprivate
            elif key == 0x011: idx = 3  # CONSTANT,CTA,PRIVATE
            elif key == 0x221: idx = 4  # STRONG,CTA,PRIVATE
            elif key == 0x220: idx = 5  # STRONG,CTA,noprivate
            elif key == 0x241: idx = 6  # STRONG,GPU,PRIVATE
            elif key == 0x240: idx = 7  # STRONG,GPU,noprivate
            elif key == 0x340: idx = 8  # MMIO,GPU,noprivate
            elif key == 0x020: idx = 9  # CONSTANT,SM,noprivate
            elif key == 0x250: idx = 10 # STRONG,SYS,noprivate
            elif key == 0x021: idx = 11 # CONSTANT,SM,PRIVATE
            elif key == 0x350: idx = 12 # MMIO,SYS,noprivate
            elif key == 0x030: idx = 13 # CONSTANT,VC,noprivate
            elif key == 0x031: idx = 14 # CONSTANT,VC,PRIVATE
            elif key == 0x040: idx = 15 # CONSTANT,GPU,noprivate
            if idx is not None:
                MEM0_REV[idx] = (sem_k, sco_k, prv_k)


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


def decode_stg(lo64: int, hi64: int) -> Optional[str]:
    opc = get_opcode(lo64, hi64)
    if opc not in (0x386, 0x1986):
        return None

    pg = extract(lo64, hi64, [14, 13, 12])
    pg_not = extract(lo64, hi64, [15])
    e = extract(lo64, hi64, [72])
    cop = extract(lo64, hi64, [86, 85, 84])
    sz = extract(lo64, hi64, [75, 74, 73])
    mem = extract(lo64, hi64, [80, 79, 78, 77])
    memdesc_flag = extract(lo64, hi64, [76])
    ra = extract(lo64, hi64, [31, 30, 29, 28, 27, 26, 25, 24])
    rb = extract(lo64, hi64, [39, 38, 37, 36, 35, 34, 33, 32])
    raw_off = extract(lo64, hi64, list(range(63, 39, -1)))
    offset = s24(raw_off)

    urc = None
    if opc == 0x1986:
        urc = extract(lo64, hi64, [69, 68, 67, 66, 65, 64])

    parts = []
    if pg != 7:
        parts.append(f"@{'!' if pg_not else ''}P{pg}")

    mnem = "STG"
    if e == 1:
        mnem += ".E"
    if cop != 1:
        mnem += f".{COP[cop]}"
    if sz != 4:
        mnem += f".{SZ[sz]}"

    mem_info = MEM0_REV.get(mem)
    if mem_info:
        ms, mc, mp = mem_info
        if ms != 1:
            mnem += f".{SEM_n[ms]}"
        if mc != 0:
            mnem += f".{SCO_n[mc]}"
        if mp != 0:
            mnem += f".{PRIV_n[mp]}"

    parts.append(mnem)

    if memdesc_flag == 1 and urc is not None:
        off_s = ""
        if offset > 0:
            off_s = f"+{offset:#x}"
        elif offset < 0:
            off_s = f"{offset:#x}"
        e64 = ".64" if e == 1 else ""
        parts.append(f"desc[UR{urc}][R{ra}{e64}{off_s}],")
    elif urc is not None:
        e64 = ".64" if e == 1 else ""
        if ra == 0xff:
            off_s = ""
            if offset > 0:
                off_s = f"+{offset:#x}"
            elif offset < 0:
                off_s = f"{offset:#x}"
            parts.append(f"[UR{urc}{off_s}],")
        else:
            off_s = ""
            if offset > 0:
                off_s = f"+{offset:#x}"
            elif offset < 0:
                off_s = f"{offset:#x}"
            parts.append(f"[R{ra}{e64} + UR{urc}{off_s}],")
    else:
        ra_s = f"R{ra}" if ra != 0xff else "RZ"
        if offset > 0:
            parts.append(f"[{ra_s}+{offset:#x}],")
        elif offset < 0:
            parts.append(f"[{ra_s}{offset:#x}],")
        else:
            parts.append(f"[{ra_s}],")

    parts.append(f"R{rb}" if rb != 0xff else "RZ")
    return " ".join(parts)


if __name__ == "__main__":
    tests = [
        (0x0000001c10000986, 0x000ea2000c101908,
         "@P0 STG.E desc[UR8][R16.64], R28"),
        (0x0001001b10001986, 0x000ea2000c101908,
         "@P1 STG.E desc[UR8][R16.64+0x100], R27"),
        (0x0000801910000986, 0x000ea2000c101908,
         "@P0 STG.E desc[UR8][R16.64+0x80], R25"),
        (0x0000001d10000986, 0x000ea2000c101908,
         "@P0 STG.E desc[UR8][R16.64], R29"),
        (0x0001001c10001986, 0x000ea2000c101908,
         "@P1 STG.E desc[UR8][R16.64+0x100], R28"),
        (0x0000001d10000986, 0x000ea2000c101908,
         "@P0 STG.E desc[UR8][R16.64], R29"),
        (0x0000000d02005986, 0x000ea2000c101908,
         "@P5 STG.E desc[UR8][R2.64], R13"),
        (0x0000801102006986, 0x000ea2000c101908,
         "@P6 STG.E desc[UR8][R2.64+0x80], R17"),
        (0x0001001502000986, 0x000ea2000c101908,
         "@P0 STG.E desc[UR8][R2.64+0x100], R21"),
        (0x0000000304007986, 0x001fe2000c101904,
         "STG.E desc[UR4][R4.64], R3"),
    ]
    ok = 0
    for lo, hi, exp in tests:
        r = decode_stg(lo, hi)
        s = "OK" if r == exp else "MISMATCH"
        if r == exp: ok += 1
        print(f"{r}")
        if s != "OK":
            print(f"  expected: {exp}")
    print(f"\n{ok}/{len(tests)} PASS")
