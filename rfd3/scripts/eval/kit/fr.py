import csv, json, sys, statistics as st, random, os
M = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'manifest.json')))
NAT = {t: v['native'] for t, v in M['targets'].items()}
ARM = {k: v['record'] if 'record' in v else v for k, v in M['arms'].items()}
def f(x):
    try: return float(x)
    except Exception: return float('nan')
def nn(x): return x == x
def mean(x):
    x = [v for v in x if nn(v)]; return st.mean(x) if x else float('nan')
def load(tag):
    rows = [r for r in csv.DictReader(open(f'{tag}.designs.csv')) if r['error'] in ('', '0', 'False', 'None')]
    tm = {}
    if os.path.exists(f'{tag}.tm.csv'):
        for r in csv.DictReader(open(f'{tag}.tm.csv')): tm[(r['target'], r['arm'], f"{float(r['cfg']):.1f}", r['batch'], r['model'])] = float(r['tm_ref'])
    for r in rows:
        a = ARM[f"{r['target']}__{r['arm']}"] if f"{r['target']}__{r['arm']}" in ARM else None
        P = (len(a['pos_pairs']) if isinstance(a['pos_pairs'], list) else a['pos_pairs']) if a else 0; Z = (len(a['neg_pairs']) if isinstance(a['neg_pairs'], list) else a['neg_pairs']) if a else 0
        rec, viol, ss = f(r['recovery']), f(r['violation']), f(r['ss_acc'])
        adj_n = (P if nn(rec) else 0) + (Z if nn(viol) else 0)
        adj = ((rec * P if nn(rec) else 0) + ((1 - viol) * Z if nn(viol) else 0)) / adj_n if adj_n else float('nan')
        r['_adj'] = adj; r['_ss'] = ss; r['_fr'] = (ss + adj) / 2 if nn(ss) and nn(adj) else float('nan')
        r['_perfect'] = int(ss >= .8 and rec >= .9 and (not nn(viol) or viol <= .1)) if nn(ss) and nn(rec) else float('nan')
        r['_tm'] = tm.get((r['target'], r['arm'], f"{float(r['cfg']):.1f}", r['batch'], r['model']), float('nan'))
        r['_split'] = int(f(r['half_sep']) > 1.15 * NAT[r['target']]['half_sep']) if r['target'] in NAT else float('nan')
    return rows
CONDARMS = ('full', 'sparse8', 'dense0_true')
def agg(rows, cfg, arms=CONDARMS, target=None):
    R = [r for r in rows if r['arm'] in arms and abs(f(r['cfg']) - cfg) < 1e-6 and (target is None or r['target'] == target)]
    if not R: return None
    return dict(n=len(R), fr=mean([r['_fr'] for r in R]), ss=mean([r['_ss'] for r in R]), adj=mean([r['_adj'] for r in R]), rec=mean([f(r['recovery']) for r in R]),
                nc=mean([1 - f(r['violation']) for r in R if nn(f(r['violation']))]), perfect=mean([r['_perfect'] for r in R]), tm=mean([r['_tm'] for r in R]),
                tm5=mean([int(r['_tm'] >= .5) for r in R if nn(r['_tm'])]), split=mean([r['_split'] for r in R]), hs=mean([f(r['half_sep']) / NAT[r['target']]['half_sep'] for r in R if r['target'] in NAT]),
                lr=mean([f(r['lr_cb']) / NAT[r['target']]['lr_cb'] for r in R if r['target'] in NAT]))
if __name__ == '__main__':
    tags = sys.argv[1:]; D = {t: load(t) for t in tags}
    for cfg in (1.0, 1.5, 2.0):
        print(f'\n### FOLD RECOVERY, CFG {cfg} (arms full+sparse8+dense0_true; FR = (SS accuracy + adjacency accuracy)/2; adjacency accuracy = specified contacts realised + specified non-contacts kept apart)')
        print(f"{'ckpt':14s}{'n':>5}{'FR':>6}{'SS':>6}{'adj':>6}{'contact':>8}{'noncon':>7}{'perfect':>8}{'TM':>6}{'TM>=.5':>7}{'split%':>7}{'hs/nat':>7}{'lrCB/nat':>9}")
        for t in tags:
            x = agg(D[t], cfg)
            if x: print(f"{t:14s}{x['n']:5d}{x['fr']:6.3f}{x['ss']:6.3f}{x['adj']:6.3f}{x['rec']:8.3f}{x['nc']:7.3f}{x['perfect']:8.3f}{x['tm']:6.3f}{x['tm5']:7.3f}{100*x['split']:7.1f}{x['hs']:7.2f}{x['lr']:9.2f}")
    print('\n### PER TARGET at CFG 1.5: FR [TM>=0.5 rate | split rate]')
    tg = ['rossmann_3chy', 'ferredoxin_1urn', 'sandwich_1tit', 'bundle_2hmq', 'bundle_2ccy']
    print(f"{'ckpt':14s}" + ''.join(f'{t[:11]:>22s}' for t in tg))
    for t in tags:
        out = f'{t:14s}'
        for g in tg:
            x = agg(D[t], 1.5, target=g)
            out += f"{('%.2f [%.2f|%2.0f%%]' % (x['fr'], x['tm5'], 100 * x['split'])) if x else '-':>22s}"
        print(out)

    print('\n### ROSSMANN (3CHY, alpha/beta) by arm and CFG: FR | half_sep/native | lrCB/native | TM>=.5 | split%')
    for t in tags:
        for cfg in (1.0, 1.5, 2.0):
            out = f'{t:14s} CFG {cfg}: '
            for a in CONDARMS + ('ss_only',):
                x = agg(D[t], cfg, arms=(a,), target='rossmann_3chy')
                out += f"{a[:6]:>7s} " + (('%.2f|%.2f|%.2f|%.2f|%2.0f%%  ' % (x['fr'], x['hs'], x['lr'], x['tm5'] if x['tm5']==x['tm5'] else -1, 100*x['split'])) if x and a != 'ss_only' else ('hs %.2f lr %.2f  ' % (x['hs'], x['lr']) if x else '-  '))
            print(out)
