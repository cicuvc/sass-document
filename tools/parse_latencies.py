#!/usr/bin/env python3
"""Parse sm_90_latencies.txt into a queryable latency model.

Model pieces:
  * OPERATION SETS / CONNECTOR SETS  -> name -> set(base mnemonics)
  * base functional-unit pipes       -> pipe_of(mnem)
  * TABLE_<DEP>(<RES>) matrices       -> lookup(dep,res,producer,consumer,cons_role)

A TABLE block is laid out as:

    TABLE_TRUE(GPR) : <ColClass>`{roles..}
                      <ColClass>`{roles..}
                      ... =
    {
        <RowClass>`{roles..} : v1 v2 v3 ...
        ...
    };

Columns are CONSUMERS (readers, roles Ra/Rb/Rc/Re/Pr.. ), rows are PRODUCERS
(writers, roles Rd/Rd2/Pu/Pv..). cell[row][col] = dependency latency in cycles.
Values may be ints or tokens (ORDERED_ZERO / HARD(n)).

Stdlib only. Usage:
    parse_latencies.py sets [-v]
    parse_latencies.py set <NAME>
    parse_latencies.py pipe <MNEM>
    parse_latencies.py table <DEP> <RES>        (e.g. table TRUE GPR)
    parse_latencies.py lookup <DEP> <RES> <PROD> <CONS> [cons_role]
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAT = os.path.join(os.path.dirname(HERE), "sm_90_latencies.txt")

PIPE_SETS = ["int_pipe", "mio_pipe", "fe_pipe", "fmalighter_pipe", "fp16_pipe",
             "cbu_pipe", "fma64lite_pipe", "fma64heavy_pipe", "udp_pipe"]

# pipe suffixes that a mnemonic token may carry inside OPERATION SETS
_SUFFIX_RE = re.compile(r"(" + "|".join(PIPE_SETS) + r")$")


def strip_suffix(tok):
    """IADD3int_pipe -> IADD3 ; leave real mnemonics untouched."""
    m = _SUFFIX_RE.search(tok)
    if m and m.start() > 0:
        return tok[: m.start()]
    return tok


class Latencies:
    def __init__(self, path=LAT):
        self.text = open(path).read()
        self.sets = {}          # name -> frozenset(base mnemonics)
        self.tables = []        # list of dict(dep,res,cols,rows)
        self._parse_sets()
        self._parse_tables()

    # ---------- set algebra ----------
    def _literal_set(self, body):
        toks = [t.strip() for t in body.split(",") if t.strip()]
        return set(strip_suffix(t) for t in toks)

    def _eval_expr(self, expr):
        """Evaluate  A + B - {X,Y} - C  over already-known names/literals."""
        expr = expr.strip()
        # tokenize into +/- terms, respecting {...}
        terms = []
        i = 0
        sign = "+"
        buf = ""
        depth = 0
        while i < len(expr):
            c = expr[i]
            if c == "{":
                depth += 1; buf += c
            elif c == "}":
                depth -= 1; buf += c
            elif c in "+-" and depth == 0:
                if buf.strip():
                    terms.append((sign, buf.strip()))
                sign = c; buf = ""
            else:
                buf += c
            i += 1
        if buf.strip():
            terms.append((sign, buf.strip()))
        acc = set()
        for sg, term in terms:
            if term.startswith("{"):
                s = self._literal_set(term.strip("{} "))
            else:
                # strip [cond] and `{roles}
                base = term.split("`")[0]
                base = re.sub(r"\[.*?\]", "", base).strip()
                s = set(self.sets.get(base, set()))
            if sg == "+":
                acc |= s
            else:
                acc -= s
        return acc

    def _parse_sets(self):
        # Grab "NAME = {...};"  and "NAME = expr;" inside OPERATION/CONNECTOR SETS.
        # We scan the whole file line-oriented, joining multi-line {...}.
        raw = self.text
        # normalize: collapse assignments spanning lines
        # find all  IDENT = .... ;   where the RHS starts with { or a set expr
        for m in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?);",
                              raw, re.MULTILINE | re.DOTALL):
            name, rhs = m.group(1), m.group(2)
            rhs = rhs.strip()
            # skip CONNECTOR CONDITIONS (contain == or comparison / ranges)
            if "==" in rhs or ">>" in rhs or "MD_PRED" in rhs or "_OR_" in rhs:
                continue
            if rhs.startswith("{") and rhs.endswith("}"):
                self.sets[name] = frozenset(self._literal_set(rhs.strip("{} ")))
            elif re.match(r"^[A-Za-z0-9_ +\-`{},\[\]]+$", rhs):
                # set expression over names/literals
                try:
                    self.sets[name] = frozenset(self._eval_expr(rhs))
                except Exception:
                    pass

    # ---------- tables ----------
    @staticmethod
    def _parse_col_entry(tok):
        base = tok.split("`")[0]
        base = re.sub(r"\[.*?\]", "", base).strip()
        roles = re.findall(r"`\{([^}]*)\}", tok)
        rlist = []
        if roles:
            for part in roles[0].split(","):
                nm = part.strip().split()[0] if part.strip() else ""
                if nm:
                    rlist.append(nm)
        return base, rlist

    # an entry: NAME optional[cond] optional`{roles...}  (roles may contain spaces)
    _ENTRY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]*\])?(?:`\{[^}]*\})?")

    def _parse_tables(self):
        # body is delimited by  =\s*{ ... };  (role braces are '}' not '};')
        for m in re.finditer(
                r"TABLE_([A-Z_0-9]+)\(([A-Z_]+)\)\s*:(.*?)=\s*\{(.*?)\};",
                self.text, re.DOTALL):
            dep, res, header, body = m.groups()
            cols = []
            for tok in self._ENTRY_RE.findall(header):
                cols.append(self._parse_col_entry(tok))
            rows = []
            for line in body.splitlines():
                line = line.strip()
                if not line or ":" not in line:
                    continue
                lhs, rhs = line.split(":", 1)
                base, rlist = self._parse_col_entry(lhs.strip())
                vals = rhs.strip().split()
                rows.append((base, rlist, vals))
            self.tables.append(dict(dep=dep, res=res, cols=cols, rows=rows))

    # ---------- queries ----------
    def pipe_of(self, mnem):
        mnem = strip_suffix(mnem)
        return [p for p in PIPE_SETS if mnem in self.sets.get(p, ())]

    def member(self, mnem, setname):
        return strip_suffix(mnem) in self.sets.get(setname, ())

    def _row_classes_for(self, mnem):
        mnem = strip_suffix(mnem)
        return [n for n, s in self.sets.items() if mnem in s]

    def lookup(self, dep, res, producer, consumer, cons_role=None):
        """Return list of (value, colclass, colroles, rowclass) matches."""
        producer = strip_suffix(producer); consumer = strip_suffix(consumer)
        out = []
        for t in self.tables:
            if t["dep"] != dep or t["res"] != res:
                continue
            # find matching columns (consumer)
            col_idx = []
            for ci, (cbase, croles) in enumerate(t["cols"]):
                s = self.sets.get(cbase, ())
                if consumer in s:
                    if cons_role is None or (not croles) or cons_role in croles:
                        col_idx.append((ci, cbase, croles))
            if not col_idx:
                continue
            for rbase, rroles, vals in t["rows"]:
                s = self.sets.get(rbase, ())
                if producer not in s:
                    continue
                for ci, cbase, croles in col_idx:
                    if ci < len(vals):
                        out.append((vals[ci], cbase, croles, rbase))
        return out


def _fmt_table(t):
    lines = [f"TABLE_{t['dep']}({t['res']})  cols={len(t['cols'])} rows={len(t['rows'])}"]
    lines.append("  columns (consumers): " +
                 ", ".join(f"{c}`{{{','.join(r)}}}" for c, r in t["cols"]))
    for rbase, rroles, vals in t["rows"]:
        lines.append(f"  {rbase}`{{{','.join(rroles)}}} : {' '.join(vals)}")
    return "\n".join(lines)


def main(argv):
    L = Latencies()
    if not argv:
        print(__doc__); return
    cmd = argv[0]
    if cmd == "sets":
        for n in sorted(L.sets):
            if "-v" in argv:
                print(f"{n} ({len(L.sets[n])}): {' '.join(sorted(L.sets[n]))}")
            else:
                print(f"{n:40} {len(L.sets[n])}")
    elif cmd == "set":
        n = argv[1]
        print(f"{n} ({len(L.sets.get(n,[]))}):")
        print("  " + " ".join(sorted(L.sets.get(n, []))))
    elif cmd == "pipe":
        print(f"{argv[1]}: {L.pipe_of(argv[1])}")
    elif cmd == "table":
        dep, res = argv[1], argv[2]
        for t in L.tables:
            if t["dep"] == dep and t["res"] == res:
                print(_fmt_table(t)); print()
    elif cmd == "lookup":
        dep, res, prod, cons = argv[1:5]
        role = argv[5] if len(argv) > 5 else None
        res_list = L.lookup(dep, res, prod, cons, role)
        if not res_list:
            print("no match")
        for v, cb, cr, rb in res_list:
            print(f"  {v:>4}  [{rb} -> {cb}`{{{','.join(cr)}}}]")
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
