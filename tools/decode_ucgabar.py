#!/usr/bin/env python3
"""Decoder for the sm_90 CGA (thread-block cluster) barrier ops on the uniform datapath:
UCGABAR_ARV / UCGABAR_WAIT / UCGABAR_GET.

These implement Hopper cluster-barrier arrive/wait/query (cooperative-groups
`cluster_group::barrier_arrive()` / `barrier_wait()`). Uniform-datapath ops
(udp_pipe), uniform-predicate guarded, compute-only.

  0x19c7 UCGABAR_ARV [.SYNCALL]   arrive at the cluster barrier
  0x1dc7 UCGABAR_WAIT             wait on the cluster barrier
  0x15c7 UCGABAR_GET  URd         read the barrier token into a uniform reg (URd[21:16])
  0x13c7 UCGABAR_SET  URb         set/init the barrier from a uniform reg (URb[37:32])

Note: UCGABAR_GET/SET are defined in the ISA DB but nvdisasm (CUDA 13.1) does NOT
render them (hand-patched 0x15c7/0x13c7 disassemble to headerless raw bytes), and
ptxas does not emit them; their operand renderings below are spec-inferred.

Fields (128-bit):
  opcode = {bit[91], bits[11:0]}   (b91=1 for all)
  UPg=[14:12] UPg_not=[15]   uniform-predicate guard (7=UPT hidden -> "@UP<n>")
  syncall=[72]               ARV only -> .SYNCALL
  URd=[21:16]                GET only -> destination uniform reg

Usage: python3 decode_ucgabar.py            (self-test)
       python3 decode_ucgabar.py <sass.txt>  (validate every UCGABAR* in a dump)
"""
import re
import sys

OPC = {0x19c7: "UCGABAR_ARV", 0x1dc7: "UCGABAR_WAIT",
       0x15c7: "UCGABAR_GET", 0x13c7: "UCGABAR_SET"}


def bits(v, hi, lo):
    return (v >> lo) & ((1 << (hi - lo + 1)) - 1)


def upred(idx, neg):
    return ("!" if neg else "") + ("UPT" if idx == 7 else "UP%d" % idx)


def decode(lo64, hi64, pc=0):
    inst = lo64 | (hi64 << 64)
    opcode = (bits(inst, 91, 91) << 12) | bits(inst, 11, 0)
    if opcode not in OPC:
        return "?opcode 0x%x" % opcode
    mnem = OPC[opcode]

    upg, upg_not = bits(inst, 14, 12), bits(inst, 15, 15)
    guard = "" if (upg == 7 and upg_not == 0) else "@%s " % upred(upg, upg_not)

    if opcode == 0x19c7 and bits(inst, 72, 72):
        mnem += ".SYNCALL"
    if opcode == 0x15c7:                       # GET: URd[21:16] (spec-inferred)
        urd = bits(inst, 21, 16)
        mnem += " %s" % ("URZ" if urd == 63 else "UR%d" % urd)
    if opcode == 0x13c7:                       # SET: URb[37:32] (spec-inferred)
        urb = bits(inst, 37, 32)
        mnem += " %s" % ("URZ" if urb == 63 else "UR%d" % urb)
    return (guard + mnem).rstrip()


# (lo64, hi64, expected) — real cluster.barrier_arrive/wait + cubin-patch guards
VECTORS = [
    (0x00000000000079c7, 0x000fe20008000000, "UCGABAR_ARV"),          # cluster.barrier_arrive()
    (0x0000000000007dc7, 0x000fe20008000000, "UCGABAR_WAIT"),         # cluster.barrier_wait()
    (0x00000000000019c7, 0x000fe20008000000, "@UP1 UCGABAR_ARV"),     # patch (UPg=1)
    (0x00000000000099c7, 0x000fe20008000000, "@!UP1 UCGABAR_ARV"),    # patch (UPg=1, not)
]


def run_vectors():
    ok = 0
    for lo, hi, exp in VECTORS:
        got = decode(lo, hi)
        ok += got == exp
        print("%s %-22s (exp %s)" % ("OK " if got == exp else "XX ", got, exp))
    print("\n%d/%d vectors matched" % (ok, len(VECTORS)))


LINE = re.compile(r"/\*([0-9a-f]+)\*/\s+(.*?);\s*/\*\s*([0-9a-fx]+)\s*\*/")
HEX = re.compile(r"/\*\s*([0-9a-fx]+)\s*\*/")


def validate_dump(path):
    lines = open(path).readlines()
    total = ok = 0
    for i, ln in enumerate(lines):
        m = LINE.search(ln)
        if not m or not re.search(r"\bUCGABAR_", m.group(2)):
            continue
        text, lo = m.group(2).strip(), int(m.group(3), 16)
        hm = HEX.search(lines[i + 1]) if i + 1 < len(lines) else None
        if not hm:
            continue
        got = decode(lo, int(hm.group(1), 16))
        total += 1
        ok += got == text
        if got != text:
            print("XX got %-22s exp %-22s [%016x]" % (got, text, lo))
    print("%s: %d/%d UCGABAR* matched" % (path, ok, total))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            validate_dump(p)
    else:
        run_vectors()
