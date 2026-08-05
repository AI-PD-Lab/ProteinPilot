# atomworks patch

HARMONIA depends on the public [AtomWorks](https://github.com/RosettaCommons/atomworks)
package (`atomworks[ml]>=2.1.1`) rather than vendoring it. Two small changes
were made on top of the upstream package for this project — mainly to support
tar-backed dataset rows for large training mirrors. Apply them after
installing atomworks:

```bash
pip install "atomworks[ml]>=2.1.1"
ATOMWORKS_SRC=$(python -c "import atomworks, pathlib; print(pathlib.Path(atomworks.__file__).parent)")
cp -r src/atomworks/* "$ATOMWORKS_SRC/"
```

Files changed vs. upstream:

| File | Change |
|---|---|
| `src/atomworks/ml/datasets/parsers/tar_backed_row_parsers.py` | New — row parser that reads structures directly out of tar-backed PDB mirrors instead of loose files. |
| `src/atomworks/ml/datasets/parsers/__init__.py` | +1 line — exports the new parser. |
| `src/atomworks/io/utils/bonds.py` | Small bugfix in bond reconstruction. |
| `src/atomworks/ml/samplers.py` | Small addition used by the SS/ADJ dataset splits. |

If you don't need tar-backed data loading (i.e. you're training on a regular
directory of structure files), you can skip this patch entirely and use
stock `atomworks[ml]`.
