#!/usr/bin/env python3
"""Correlate sm_90 usched_info (stall count + group/yield bit) with the
dependency latencies in sm_90_latencies.txt.

Reads cuobjdump `-sass` text (128-bit sm_90: instruction line carries lo64 in
the first /*..*/, the following line carries hi64). Extracts per-instruction:
  * mnemonic (base, no modifiers) and pipe
  * GPR/PRED defs & uses (best-effort operand parse)
  * control fields: usched_info[109:105], req_bit_set[121:116],
    src_rel_sb[115:113], dst_wr_sb[112:110]

usched decode:  eff_stall = usched & 0xF ; group_bit = usched>>4 ;
                usched==0 -> DRAIN (yield).  1..15 = WnEG, 17..27 = Wn(trans).

Correlation: for each adjacent producer->consumer pair (distance 0) with a true
RAW dep on a fixed-latency producer, compare producer eff_stall to the
TABLE_TRUE latency predicted by parse_latencies.

Usage:
    usched_probe.py <sass.txt> [--limit N]
"""
import re, sys, collections
import parse_latencies as PL

LO_RE = re.compile(r'/\*([0-9a-f]{4})\*/\s+(.*?);\s*/\* (0x[0-9a-f]{16}) \*/')
HI_RE = re.compile(r'^\s*/\* (0x[0-9a-f]{16}) \*/\s*$')
REG_RE = re.compile(r'\bR(\d+)\b')
PRED_RE = re.compile(r'\bP(\d+)\b')

FIXED_PIPES = {"int_pipe", "fmalighter_pipe", "fp16_pipe",
               "fma64lite_pipe", "fma64heavy_pipe", "udp_pipe"}
BRANCHISH = {"BRA","BRX","JMP","JMX","CALL","RET","EXIT","BSYNC","WARPSYNC",
             "BSSY","KILL","BREAK","NANOSLEEP","YIELD","BPT","RTT","JMXU","BRXU"}


def fld(v, hi, lo):
    return (v >> lo) & ((1 << (hi - lo + 1)) - 1)


def split_operands(s):
    out, buf, depth = [], "", 0
    for c in s:
        if c in "[{(":
            depth += 1; buf += c
        elif c in "]})":
            depth -= 1; buf += c
        elif c == "," and depth == 0:
            out.append(buf); buf = ""
        else:
            buf += c
    if buf.strip():
        out.append(buf)
    return [o.strip() for o in out]


class Insn:
    __slots__ = ("addr","mnem","pipe","gpr_def","gpr_use","pred_def","pred_use",
                 "usched","stall","group","req","rd_sb","wr_sb","text","is_store",
                 "is_branch")

    def __init__(self, text, lo64, hi64, lat):
        full = (hi64 << 64) | lo64
        self.text = text
        toks = text.split()
        # strip predicate guard
        idx = 0
        if toks and toks[0].startswith("@"):
            idx = 1
        mfull = toks[idx] if idx < len(toks) else ""
        self.mnem = mfull.split(".")[0]
        self.pipe = (lat.pipe_of(self.mnem) or ["?"])[0]
        self.usched = fld(full, 109, 105)
        self.stall = self.usched & 0xF
        self.group = (self.usched >> 4) & 1
        self.req = fld(full, 121, 116)
        self.rd_sb = fld(full, 115, 113)
        self.wr_sb = fld(full, 112, 110)
        self.is_branch = self.mnem in BRANCHISH
        self.is_store = bool(re.match(r"^(ST|RED|ATOM|SUST|SURED|SUATOM|CCTL|MEMBAR|FENCE|DEPBAR|BAR|ERRBAR)", self.mnem))
        # operand parse
        opstr = text.split(None, idx + 1)
        opstr = opstr[-1] if len(opstr) > idx + 1 else ""
        ops = split_operands(opstr)
        self.gpr_def = set(); self.gpr_use = set()
        self.pred_def = set(); self.pred_use = set()
        wide = (".64" in mfull) or (".WIDE" in mfull)

        def regs_of(op):
            return [int(x) for x in REG_RE.findall(op) if int(x) != 255]

        def preds_of(op):
            return [int(x) for x in PRED_RE.findall(op) if int(x) != 7]

        # Destination operands: op0 is a dest (unless store), plus any
        # immediately-following bare PREDICATE operands (carry-out / dual pred).
        dest_end = 0
        if not self.is_store and ops:
            dest_end = 1
            for oi in range(1, len(ops)):
                o = ops[oi].strip()
                if re.fullmatch(r"!?(P\d+|PT|UP\d+|UPT)", o):
                    dest_end = oi + 1
                else:
                    break
        for oi, op in enumerate(ops):
            r = regs_of(op); p = preds_of(op)
            if oi < dest_end and "[" not in op:
                self.gpr_def.update(r)
                if wide:
                    self.gpr_def.update(x + 1 for x in r)
                self.pred_def.update(p)
            else:
                self.gpr_use.update(r)
                self.pred_use.update(p)


def parse_stream(fp, lat):
    """Yield lists of Insn per function block."""
    block = []
    pending = None
    for line in fp:
        if "Function :" in line or ".headerflags" in line:
            if block:
                yield block
            block = []
            pending = None
            continue
        m = LO_RE.search(line)
        if m:
            pending = (m.group(2).strip(), int(m.group(3), 16))
            continue
        h = HI_RE.match(line)
        if h and pending:
            text, lo64 = pending
            hi64 = int(h.group(1), 16)
            pending = None
            try:
                block.append(Insn(text, lo64, hi64, lat))
            except Exception:
                pass
    if block:
        yield block


def analyze_overlap(path, lat, limit=None):
    """Full RAW forwarding survey: for every producer->consumer op-pair, the
    table latency L vs the minimum pure cumulative gap minG. overlap = L - minG
    is the forwarding/bypass the hardware provides (0 = no forwarding).
    Aggregated per (producer_pipe -> consumer_pipe)."""
    # per (pmnem,cmnem): [L, minG, n]
    pair = collections.defaultdict(lambda: [-1, 10 ** 9, 0])  # [L, minG, n]
    n = 0
    with open(path) as fp:
        for block in parse_stream(fp, lat):
            pos, pure = reconstruct_pos(block)
            for i, p in enumerate(block):
                if p.pipe not in FIXED_PIPES or p.is_branch or not p.gpr_def:
                    continue
                for r in p.gpr_def:
                    pure_seg = True
                    for j in range(i + 1, min(len(block), i + 40)):
                        q = block[j]; pure_seg = pure_seg and pure[j]
                        if r in q.gpr_use:
                            vals = [int(v) for v, *_ in lat.lookup("TRUE", "GPR", p.mnem, q.mnem) if v.isdigit()]
                            if vals:
                                L = min(vals); G = pos[j] - pos[i]; n += 1
                                e = pair[(p.mnem, p.pipe, q.mnem, q.pipe)]
                                e[0] = L
                                if pure_seg:
                                    e[1] = min(e[1], G)
                                e[2] += 1
                            break
                        if r in q.gpr_def or q.is_branch:
                            break
            if limit and n > limit:
                break

    # aggregate to pipe-pair: overlap distribution weighted by n (only pairs with
    # a valid pure minG and enough samples)
    pipe_ov = collections.defaultdict(collections.Counter)  # (pp,cp)->Counter(overlap->n)
    examples = collections.defaultdict(list)
    for (pm, pp, cm, cp), (L, mg, k) in pair.items():
        if L == -1 or mg == 10 ** 9 or k < 200:
            continue
        ov = L - mg
        pipe_ov[(pp, cp)][ov] += k
        examples[(pp, cp)].append((k, pm, cm, L, mg, ov))

    print("== RAW forwarding overlap (L_table - minG) per producer->consumer pipe ==")
    print("   overlap>0 => hardware forwarding/bypass; 0 => table latency is exact\n")
    print(f"  {'producer pipe':16} {'consumer pipe':16} {'wt.ovlp':>8} {'range':>7} {'n':>10}")
    consumer_roll = collections.defaultdict(collections.Counter)
    for (pp, cp) in sorted(pipe_ov):
        c = pipe_ov[(pp, cp)]
        tot = sum(c.values())
        wov = sum(o * k for o, k in c.items()) / tot
        lo = min(c); hi = max(c)
        print(f"  {pp:16} {cp:16} {wov:8.2f} {str(lo)+'..'+str(hi):>7} {tot:>10}")
        consumer_roll[cp].update(c)

    print("\n== rolled up by CONSUMER pipe (forwarding is a consumer-read property) ==")
    print(f"  {'consumer pipe':16} {'wt.mean overlap':>16} {'n':>10}")
    for cp in sorted(consumer_roll):
        c = consumer_roll[cp]; tot = sum(c.values())
        wov = sum(o * k for o, k in c.items()) / tot
        print(f"  {cp:16} {wov:16.2f} {tot:>10}")

    print("\n== representative op-pairs with non-zero overlap ==")
    flat = []
    for key, lst in examples.items():
        for (k, pm, cm, L, mg, ov) in lst:
            if ov != 0:
                flat.append((k, pm, cm, key[0], key[1], L, mg, ov))
    for k, pm, cm, pp, cp, L, mg, ov in sorted(flat, reverse=True)[:20]:
        print(f"  {pm+'->'+cm:20} ({pp}->{cp})  L={L} minG={mg} overlap={ov}  n={k}")


def reconstruct_pos(block):
    """Assign issue cycle to each insn: pos[i+1]=pos[i]+eff_stall[i].
    Returns (pos list, pure list) where pure[i] flags that the gap from i-1 to i
    is exactly known (no DRAIN, no wait-mask on i, not after a branch)."""
    pos = [0] * len(block)
    pure = [True] * len(block)
    for i in range(1, len(block)):
        prev = block[i - 1]
        gap = prev.stall if prev.usched != 0 else 1  # DRAIN gap unknown -> assume >=1
        pos[i] = pos[i - 1] + max(gap, 1)
        # gap into i is impure if prev drained, i waits on a scoreboard, or prev branch
        pure[i] = (prev.usched != 0) and (block[i].req == 0) and (not prev.is_branch)
    return pos, pure


def analyze_cumulative(path, lat, limit=None):
    """For each RAW GPR edge (nearest downstream reader), measure cumulative
    issue gap G = pos_C - pos_P vs table latency, across distances."""
    # (p_pipe,c_pipe) -> Counter over (L, G)
    gap_stats = collections.defaultdict(collections.Counter)
    # (p_mnem,c_mnem) -> list stats: min G on pure edges, table L
    tight = collections.defaultdict(lambda: [10 ** 9, -1, 0])  # minG, L, n
    feas = collections.Counter()  # (G>=L) bool count
    n = 0
    with open(path) as fp:
        for block in parse_stream(fp, lat):
            pos, pure = reconstruct_pos(block)
            for i, p in enumerate(block):
                if p.pipe not in FIXED_PIPES or not p.gpr_def or p.is_branch:
                    continue
                for r in p.gpr_def:
                    # nearest downstream reader of r, no intervening redef
                    seg_pure = True
                    for j in range(i + 1, min(len(block), i + 40)):
                        q = block[j]
                        if not pure[j]:
                            seg_pure = False
                        if r in q.gpr_use:
                            pl = lat.lookup("TRUE", "GPR", p.mnem, q.mnem)
                            vals = [int(v) for v, *_ in pl if v.isdigit()]
                            if vals:
                                L = min(vals)
                                G = pos[j] - pos[i]
                                n += 1
                                gap_stats[(p.pipe, q.pipe)][(L, G)] += 1
                                feas[G >= L] += 1
                                if seg_pure:
                                    t = tight[(p.mnem, q.mnem)]
                                    if G < t[0]:
                                        t[0] = G
                                    t[1] = L
                                    t[2] += 1
                            break
                        if r in q.gpr_def or q.is_branch:
                            break
                if limit and n > limit:
                    break
    print(f"RAW GPR edges analysed: {n}")
    tot = sum(feas.values())
    print(f"feasibility G>=L_table: {100*feas[True]/tot:.1f}%   G<L_table: {100*feas[False]/tot:.1f}%")
    print("\n== cumulative gap G vs table L by pipe pair (min G = hw-effective latency) ==")
    for key in sorted(gap_stats):
        c = gap_stats[key]
        total = sum(c.values())
        if total < 200:
            continue
        Ls = set(L for (L, G) in c)
        minG_byL = {}
        for (L, G), k in c.items():
            minG_byL.setdefault(L, 10 ** 9)
            minG_byL[L] = min(minG_byL[L], G)
        summary = ", ".join(f"L={L}:minG={minG_byL[L]}" for L in sorted(Ls))
        print(f"  {key[0]:15}->{key[1]:15} n={total:8}  {summary}")
    print("\n== tight edges (pure timing): table L vs observed minimum cumulative gap ==")
    rows = sorted(tight.items(), key=lambda kv: -kv[1][2])[:24]
    print(f"  {'pair':20} {'L_table':>7} {'minG':>5} {'overlap(L-minG)':>15} {'n':>9}")
    for (pm, cm), (mg, L, k) in rows:
        if L == -1 or mg == 10 ** 9:
            continue
        print(f"  {pm+'->'+cm:20} {L:>7} {mg:>5} {L-mg:>15} {k:>9}")


def solve_trifamily(path, lat, limit=None):
    """One-pass collection of pure-edge minimum cumulative gaps for RAW, WAW,
    WAR, then solve per-pipe w[pipe] and r[pipe] from all three families.
    Anchor: r[fmalighter_pipe] = 0 (assumes FMAI consumer reads at issue)."""
    # min observed pure gap per (dep, prod_pipe, cons_pipe)
    obs = collections.defaultdict(lambda: 10 ** 9)  # (dep,Ppipe,Cpipe)->minG
    edges = 0
    with open(path) as fp:
        for block in parse_stream(fp, lat):
            pos, pure = reconstruct_pos(block)
            for i, pi in enumerate(block):
                p = pi
                if p.is_branch or p.pipe not in FIXED_PIPES:
                    continue
                # nearest downstream consumer (RAW), writer (WAW), writer-after-read (WAR)
                # per def register
                for (typ, search_set) in [("RAW", p.gpr_def), ("WAW", p.gpr_def), ("WAR", p.gpr_use)]:
                    for r in search_set:
                        pure_seg = True
                        for j in range(i + 1, min(len(block), i + 40)):
                            q = block[j]; pure_seg = pure_seg and pure[j]  # pure[j] means gap i→j pure
                            hit = None; want = None
                            if typ == "RAW" and r in q.gpr_use:
                                hit = "TRUE"; want = "RAW"
                            elif typ == "WAW" and r in q.gpr_def:
                                hit = "OUTPUT"; want = "WAW"
                            elif typ == "WAR" and r in q.gpr_def:
                                # WAR: p reads r, q writes r (anti-dep)
                                hit = "ANTI"; want = "WAR"
                            if hit:
                                vals = [int(v) for v, *_ in lat.lookup(hit, "GPR", p.mnem, q.mnem) if v.isdigit()]
                                if vals:
                                    G = pos[j] - pos[i]
                                    edges += 1
                                    key = (want, p.pipe, q.pipe)
                                    if pure_seg and G < obs[key]:
                                        obs[key] = G
                                break
                            # stop search if redefined or branch
                            if r in q.gpr_def and typ == "RAW":
                                break
                            if q.is_branch:
                                break
                if limit and edges > limit:
                    break
            if limit and edges > limit:
                break
    print(f"edges scanned: {edges}  tri-family pure-edge minima observed:\n")
    pipes = sorted(FIXED_PIPES)
    for dep in ["RAW","WAW","WAR"]:
        print(f"  {dep}:")
        for pp in pipes:
            for cp in pipes:
                k = (dep, pp, cp)
                if k in obs:
                    print(f"    {pp:18} -> {cp:18}  minG={obs[k]:>4}")
    # Solve w[pipe], r[pipe] by least-squares over the difference constraints
    #   RAW: w[P]-r[C]=G ; WAW: w[P]-w[C]=G ; WAR: r[P]-w[C]=G
    # via Gauss-Seidel averaging (anchor r[fmalighter]=0), then report residuals.
    print("\n----- w/r least-squares solve (anchor r[fmalighter_pipe]=0) -----")
    anchor = "fmalighter_pipe"
    var = {}
    for p in pipes:
        var[f"w:{p}"] = 0.0
        var[f"r:{p}"] = 0.0
    eqs = []  # (lhs_var, rhs_var, G)  meaning lhs - rhs = G
    # RAW is the only reliably-binding family (tight in throughput-bound code).
    # WAW is informative only when w[P]>w[C] (cross-pipe heavy->light); same-pipe
    # WAW and all WAR edges are floored ~1-2 (non-binding) so we exclude them.
    for (dep, pp, cp), G in obs.items():
        if dep == "RAW":
            eqs.append((f"w:{pp}", f"r:{cp}", G))
        elif dep == "WAW" and pp != cp and G >= 4:
            eqs.append((f"w:{pp}", f"w:{cp}", G))
    for a, b, _ in eqs:  # ensure every referenced var exists
        var.setdefault(a, 0.0); var.setdefault(b, 0.0)
    for _ in range(2000):
        acc = collections.defaultdict(list)
        for a, b, G in eqs:
            if a == f"r:{anchor}":
                acc[b].append(var[a] - G)
            elif b == f"r:{anchor}":
                acc[a].append(var[b] + G)
            else:
                acc[a].append(var[b] + G)
                acc[b].append(var[a] - G)
        var[f"r:{anchor}"] = 0.0
        for k, lst in acc.items():
            if k == f"r:{anchor}":
                continue
            var[k] = sum(lst) / len(lst)
    print(f"  {'pipe':18} {'w':>6} {'r':>6}")
    for p in pipes:
        seen = any(p in (k.split(':')[1],) for (a, b, _) in eqs for k in (a, b))
        if not seen:
            continue
        print(f"  {p:18} {var[f'w:{p}']:6.1f} {var[f'r:{p}']:6.1f}")
    # residuals: |predicted - observed| per equation
    res = []
    for dep, pp, cp in obs:
        if dep == "RAW":
            pv = var.get(f"w:{pp}", 0) - var.get(f"r:{cp}", 0)
        elif dep == "WAW" and pp != cp and obs[(dep, pp, cp)] >= 4:
            pv = var.get(f"w:{pp}", 0) - var.get(f"w:{cp}", 0)
        else:
            continue
        G = obs[(dep, pp, cp)]
        res.append((abs(pv - G), dep, pp, cp, G, pv))
    rms = (sum(r[0] ** 2 for r in res) / len(res)) ** 0.5
    print(f"\n  fit RMS residual = {rms:.2f} cycles over {len(res)} binding (RAW + heavy-WAW) constraints")
    print("  worst residuals:")
    for d, dep, pp, cp, G, pv in sorted(res, reverse=True)[:5]:
        print(f"    {dep:4} {pp}->{cp}: obs={G} model={pv:.1f} |Δ|={d:.1f}")


def main(argv):
    path = argv[0]
    limit = None
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])
    lat = PL.Latencies()
    if "--cumulative" in argv:
        analyze_cumulative(path, lat, limit)
        return
    if "--overlap" in argv:
        analyze_overlap(path, lat, limit)
        return
    if "--solve" in argv:
        solve_trifamily(path, lat, limit)
        return

    n_insn = 0
    # adjacent RAW producer->consumer, producer fixed-latency
    # key (prod_pipe, cons_pipe) -> Counter of (predicted_latency, eff_stall)
    raw_stats = collections.defaultdict(collections.Counter)
    # per (prod_mnem, cons_mnem): stall distribution vs predicted
    pair_stall = collections.defaultdict(collections.Counter)
    pair_pred = {}
    # group/yield: does immediate successor RAW-depend?  group_bit distribution
    group_by_dep = collections.Counter()   # (has_dep, group_bit)
    n_blocks = 0

    with open(path) as fp:
        for block in parse_stream(fp, lat):
            n_blocks += 1
            for i in range(len(block) - 1):
                p = block[i]; c = block[i + 1]
                n_insn += 1
                if limit and n_insn > limit:
                    break
                if p.is_branch:
                    continue
                raw = bool(p.gpr_def & c.gpr_use) and not (p.gpr_def & set())  # gpr RAW
                # exclude if consumer also is the producer overwrite only
                has_dep = raw or bool(p.pred_def & c.pred_use)
                group_by_dep[(has_dep, p.group)] += 1
                if not raw:
                    continue
                if p.pipe not in FIXED_PIPES:
                    continue
                # predicted GPR true latency producer->consumer
                pl = lat.lookup("TRUE", "GPR", p.mnem, c.mnem)
                vals = [int(v) for v, *_ in pl if v.isdigit()]
                if not vals:
                    continue
                pred = min(vals)
                raw_stats[(p.pipe, c.pipe)][(pred, p.stall)] += 1
                pair_stall[(p.mnem, c.mnem)][p.stall] += 1
                pair_pred[(p.mnem, c.mnem)] = pred
        # summary
    print(f"blocks={n_blocks}  adjacent-pairs~={n_insn}")
    print("\n== group_bit vs has-dependent-successor ==")
    tot = sum(group_by_dep.values())
    for (hd, gb), n in sorted(group_by_dep.items()):
        print(f"  has_dep={hd!s:5}  group_bit={gb}  n={n:9}  ({100*n/tot:.1f}%)")

    print("\n== adjacent RAW: predicted latency vs eff_stall (fixed-latency producer) ==")
    for key in sorted(raw_stats):
        c = raw_stats[key]
        total = sum(c.values())
        # match rate where stall == predicted
        match = sum(n for (pred, st), n in c.items() if st == pred)
        ge = sum(n for (pred, st), n in c.items() if st >= pred)
        print(f"\n  {key[0]} -> {key[1]}  (n={total})  stall==pred:{100*match/total:.0f}%  stall>=pred:{100*ge/total:.0f}%")
        for (pred, st), n in sorted(c.items(), key=lambda x: -x[1])[:6]:
            print(f"      pred={pred:2} stall={st:2}  n={n:8}  ({100*n/total:.1f}%)")

    print("\n== top producer->consumer pairs: predicted vs modal stall ==")
    rows = sorted(pair_stall.items(), key=lambda kv: -sum(kv[1].values()))[:20]
    for (pm, cm), c in rows:
        total = sum(c.values())
        modal = c.most_common(1)[0]
        print(f"  {pm:7}->{cm:7} pred={pair_pred[(pm,cm)]:2}  n={total:8}  modal_stall={modal[0]}({100*modal[1]/total:.0f}%)  dist={dict(c.most_common(5))}")


if __name__ == "__main__":
    main(sys.argv[1:])
