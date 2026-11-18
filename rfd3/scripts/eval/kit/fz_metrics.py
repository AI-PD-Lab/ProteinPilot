import sys, glob, os, re, json, statistics as st
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import eval_common as ec, design_metrics as dm
from multiprocessing import Pool
NAT={t:v['native'] for t,v in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'manifest.json')))['targets'].items()}
def one(f):
    try:
        ch=ec.protein_chains(ec.read_structure(f, os.path.basename(f)), min_len=40)[0]; r=dm.design_row(ch); r['file']=f; r['target']=os.path.basename(f).split('__')[0]; return r
    except Exception: return None
def split(r):
    n=NAT[r['target']]
    return int(r['split']<min(0.67,0.6*n['split']) or r['lr_cb']<min(0.92,0.8*n['lr_cb']) or r['rg_n04']>max(2.15,1.12*n['rg_n04']))
def strict(r): return int(r['split']<0.67 or r['lr_cb']<0.92 or r['rg_n04']>2.15)
if __name__=='__main__':
    root=sys.argv[1]
    print(f"{'cell':16s}{'target':16s}{'n':>3}{'split(nat-relaxed)%':>20}{'Rg/nat':>8}{'lr/nat':>8}{'cut/nat':>8}{'helix':>6}{'strand':>7}{'clash':>6}{'brk':>5}")
    for c in sorted(d for d in os.listdir(root) if os.path.isdir(f'{root}/{d}')):
        files=sorted(glob.glob(f'{root}/{c}/*.cif.gz'))
        with Pool(16) as p: R=[r for r in p.map(one,files) if r]
        allsp=[split(r) for r in R]
        print(f"{c:16s}{'ALL':16s}{len(R):3d}{100*st.mean(allsp):20.1f}{st.mean(r['rg_n04']/NAT[r['target']]['rg_n04'] for r in R):8.3f}{st.mean(r['lr_cb']/NAT[r['target']]['lr_cb'] for r in R):8.2f}{st.mean(r['split']/NAT[r['target']]['split'] for r in R):8.2f}{st.mean(r['helix'] for r in R):6.2f}{st.mean(r['strand'] for r in R):7.2f}{st.mean(r['heavy_clash'] for r in R):6.1f}{st.mean(r['ca_breaks'] for r in R):5.2f}")
        for t in NAT:
            L=[r for r in R if r['target']==t]
            if L: print(f"{'':16s}{t:16s}{len(L):3d}{100*st.mean(split(r) for r in L):20.1f}{st.mean(r['rg_n04']/NAT[t]['rg_n04'] for r in L):8.3f}{st.mean(r['lr_cb']/NAT[t]['lr_cb'] for r in L):8.2f}{st.mean(r['split']/NAT[t]['split'] for r in L):8.2f}")
