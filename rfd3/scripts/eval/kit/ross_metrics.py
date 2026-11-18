import sys, glob, os, re, csv, statistics as st
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import eval_common as ec, design_metrics as dm
from multiprocessing import Pool
def one(f):
    try:
        ch=ec.protein_chains(ec.read_structure(f, os.path.basename(f)), min_len=40)[0]; r=dm.design_row(ch); r['file']=f; return r
    except Exception: return None
T_SPLIT,T_LR,T_RG=0.67,0.92,2.15
if __name__=='__main__':
    root=sys.argv[1]; cells=sorted(d for d in os.listdir(root) if os.path.isdir(f'{root}/{d}'))
    print(f"{'cell':26s}{'case':24s}{'n':>4}{'SPLIT%':>8}{'cut':>6}{'lr':>6}{'Rg':>6}  {'Rg/N^.4':>8}{'lr_cb':>7}{'cutdens':>8}{'helix':>6}{'strand':>7}{'clash':>6}{'brk':>5}")
    for c in cells:
        files=sorted(glob.glob(f'{root}/{c}/*.cif.gz'))
        with Pool(16) as p: R=[r for r in p.map(one,files) if r]
        by={}
        for r in R:
            case=re.search(r'inputs?_?(.*?)_\d+_model',os.path.basename(r['file'])); case=re.sub(r'^old_rossmann_inputs_','',os.path.basename(r['file']).split('_0_model')[0].split('_1_model')[0]) 
            case=re.sub(r'_\d+_model_\d+.*$','',os.path.basename(r['file'])); by.setdefault(case,[]).append(r)
        for case,L in sorted(by.items()):
            f=lambda k:[float(x[k]) for x in L]
            sp=[int(x['split']<T_SPLIT or x['lr_cb']<T_LR or x['rg_n04']>T_RG) for x in L]
            print(f"{c:26s}{case[-24:]:24s}{len(L):4d}{100*st.mean(sp):8.1f}{100*st.mean(int(x['split']<T_SPLIT) for x in L):6.0f}{100*st.mean(int(x['lr_cb']<T_LR) for x in L):6.0f}{100*st.mean(int(x['rg_n04']>T_RG) for x in L):6.0f}  {st.mean(f('rg_n04')):8.3f}{st.mean(f('lr_cb')):7.2f}{st.mean(f('split')):8.2f}{st.mean(f('helix')):6.2f}{st.mean(f('strand')):7.2f}{st.mean(f('heavy_clash')):6.1f}{st.mean(f('ca_breaks')):5.2f}")
