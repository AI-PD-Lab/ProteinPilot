#!/bin/bash
# Score HARMONIA-RFD4 eval tars with the RFD3 HARMONIA scorer, unchanged (CPU, login node, minutes): per-design metrics, TM to native,
# fold recovery (fr.py), strict + native-relaxed split (split2.py / split3.py). Baselines p0694 (RFD3) and stock_rfd4 are always included.
#   harmonia_ext/score_eval.sh <out_dir> <tag> [<tag> ...]     (tars in /path/to/work/harmonia_rfd4/eval/<tag>.tar.gz)
set -uo pipefail
O=$1; shift; mkdir -p $O
PY=/path/to/env/bin/python
S=${RFD3_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../rfd3" && pwd)}; K=$S/scripts/eval/kit
E4=/path/to/work/harmonia_rfd4/eval; E3=/path/to/work_rfd3/v2/eval
cp $K/*.py $O/
# red-team N001-2 #6: the kit's per-target split table indexes CFG 1.5, which RFD4 evals do not have
sed -i "s|(100\*a\[1\], 100\*b\[1\])) if a else|(100*a[1], 100*b[1] if b else float('nan'))) if a else|" $O/split3.py
# red-team N001-2 #4: the scorer and the kit helpers know only the natural-panel manifest; refuse tars generated from another one
for T in "$@"; do
  M=$(tar xzf $E4/$T.tar.gz $T/meta.json -O 2>/dev/null | $PY -c "import json,sys; print(json.load(sys.stdin).get('manifest',''))")
  case "$M" in */kit/manifest.json) ;; *) echo "REFUSED $T: generated from manifest '$M' (score with its own scorer)"; exit 2 ;; esac
done
for T in "$@" stock_rfd4; do
  [ -f $O/$T.designs.csv ] || $PY -I $S/scripts/eval/score_designs.py score --tar $E4/$T.tar.gz --out $O --specs-tar $E3/specs_v2.tar.gz --procs 16 2>&1 | tail -1 | cut -c1-80
  [ -f $O/$T.tm.csv ] || (cd $O && sed "s#^E=.*#E='$E4'#" tm_all.py > tm_all_h4.py && $PY -I tm_all_h4.py $T 2>&1 | tail -1)
done
[ -f $O/p0694.designs.csv ] || { $PY -I $S/scripts/eval/score_designs.py score --tar $E3/p0694.tar.gz --out $O --procs 16 >/dev/null 2>&1; (cd $O && $PY -I tm_all.py p0694 >/dev/null 2>&1); }
cd $O
$PY -I fr.py p0694 stock_rfd4 "$@" 2>&1 | sed -n '/CFG 1.0 (/,/^$/p;/CFG 2.0 (/,/^$/p'
$PY -I split3.py p0694 stock_rfd4 "$@" 2>&1 | head -40
