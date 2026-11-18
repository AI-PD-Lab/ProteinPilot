import csv, json, sys, statistics as st, random
M = json.load(open('/path/to/work_rfd3/v2/eval/specs_v2_manifest.json'))['targets']
NAT = {t: v['native'] for t, v in M.items()}
ARMS = ('full', 'sparse8', 'dense0_true')
def f(x):
    try: return float(x)
    except Exception: return float('nan')
def ok(r): return r['error'] in ('', '0', 'False', 'None')
def mean(x):
    x = [v for v in x if v == v]; return st.mean(x) if x else float('nan')
def load(tag): return [r for r in csv.DictReader(open(f'{tag}.designs.csv')) if ok(r)]
def boot(v, B=1000):
    rnd = random.Random(0); n = len(v); m = sorted(sum(rnd.choice(v) for _ in range(n)) / n for _ in range(B)); return m[int(.025*B)], m[int(.975*B)]
def row(rows, cfg, arms=ARMS):
    R = [r for r in rows if r['arm'] in arms and abs(f(r['cfg']) - cfg) < 1e-6]
    if not R: return None
    sf = [int(f(r['recovery']) >= .9 and f(r['half_sep']) <= 1.15 * NAT[r['target']]['half_sep'] and f(r['ca_breaks']) == 0 and f(r['heavy_clash']) <= 4 and r['len_ok'] in ('True', '1', 'true')) for r in R if r['target'] in NAT]
    lo, hi = boot(sf)
    return dict(n=len(R), SFAR=mean(sf), lo=lo, hi=hi,
        rec=mean([f(r['recovery']) for r in R]), rec90=mean([int(f(r['recovery']) >= .9) for r in R]),
        viol=mean([f(r['violation']) for r in R if r['violation'] not in ('', '-')]),
        ext=mean([f(r['extent_ratio']) for r in R]),
        hs=mean([f(r['half_sep']) / NAT[r['target']]['half_sep'] for r in R if r['target'] in NAT]),
        ss=mean([f(r['ss_acc']) for r in R]),
        rg=mean([f(r['rg_n04']) / NAT[r['target']]['rg_n04'] for r in R if r['target'] in NAT]),
        lr=mean([f(r['lr_cb']) / NAT[r['target']]['lr_cb'] for r in R if r['target'] in NAT]),
        brk=mean([f(r['ca_breaks']) for r in R]), clash=mean([f(r['heavy_clash']) for r in R]))
tags = sys.argv[1:]
for cfg in (1.0, 1.5, 2.0):
    print(f'\n### CFG {cfg}  (conditioned arms full+sparse8+dense0_true, 5 targets; ratios are design/native)')
    print(f"{'ckpt':16s}{'n':>5}{'SFAR':>7}{'[CI]':>13}{'rec':>6}{'rec>=.9':>8}{'viol':>6}{'extent':>7}{'hs/nat':>7}{'ss_acc':>7}{'Rg/nat':>7}{'lrCB/nat':>9}{'brk':>5}{'clash':>6}")
    for t in tags:
        x = row(load(t), cfg)
        if x: print(f"{t:16s}{x['n']:5d}{x['SFAR']:7.3f}  [{x['lo']:.2f},{x['hi']:.2f}]{x['rec']:6.2f}{x['rec90']:8.2f}{x['viol']:6.2f}{x['ext']:7.2f}{x['hs']:7.2f}{x['ss']:7.2f}{x['rg']:7.3f}{x['lr']:9.2f}{x['brk']:5.2f}{x['clash']:6.2f}")
print('\n### per-arm recovery / half_sep-over-native at CFG 1.5')
print(f"{'ckpt':16s}" + ''.join(f'{a:>22s}' for a in ARMS + ('ss_only',)))
for t in tags:
    R = load(t); out = f'{t:16s}'
    for a in ARMS + ('ss_only',):
        x = row(R, 1.5, (a,))
        out += f"{'  rec %.2f hs %.2f' % (x['rec'], x['hs']) if x and a != 'ss_only' else ('  hs %.2f' % x['hs'] if x else '-'):>22s}"
    print(out)
print('\n### unconditional panel (CFG-free; Rg/N^0.4, long-range CB contacts/res, helix, strand, breaks, clash), mean over L=64/120/200')
for t in tags:
    R = [r for r in load(t) if r['arm'] == 'unconditional']
    if R: print(f"{t:16s} n={len(R)} Rg/N^.4={mean([f(r['rg_n04']) for r in R]):.3f}  lr_cb={mean([f(r['lr_cb']) for r in R]):.2f}  helix={mean([f(r['helix']) for r in R]):.2f} strand={mean([f(r['strand']) for r in R]):.2f}  breaks={mean([f(r['ca_breaks']) for r in R]):.2f} clash={mean([f(r['heavy_clash']) for r in R]):.2f}")
