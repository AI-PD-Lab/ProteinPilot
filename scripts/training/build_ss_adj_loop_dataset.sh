#!/bin/bash
# Build the curated SS/ADJ loop dataset on an approved compute node.
set -euo pipefail

ROOT=${ROOT:-/work/gerstung/HARMONIA}
DATA_ROOT=${DATA_ROOT:-/work/gerstung/RFD3_Training/Data_Mirrors}
LOG_DIR=${LOG_DIR:-${ROOT}/logs}
PYTHON=${PYTHON:-/zfs/s01/z04/home/gerstung/miniforge3/envs/atomworks/bin/python3}
NODELIST=${NODELIST:-t38cn056}
PARTITION=${PARTITION:-barracuda}
RESERVATION=${RESERVATION:-AIPD}
TIME=${TIME:-24:00:00}
MEM=${MEM:-48G}
CPUS=${CPUS:-16}

OUT_DIR=${DATA_ROOT}/ss_adj_loop
SOURCE_METADATA_DIR=${OUT_DIR}/source_metadata
SOURCE_CSV=${OUT_DIR}/loop_source_records.csv
TRAIN_CSV=${OUT_DIR}/loop_dataset.csv
SABDAB_TSV=${SOURCE_METADATA_DIR}/sabdab_summary.tsv
STCRDAB_TSV=${SOURCE_METADATA_DIR}/stcrdab_summary.tsv
MCSA_JSON=${SOURCE_METADATA_DIR}/mcsa_entries.json
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

for source_file in "${SABDAB_TSV}" "${STCRDAB_TSV}" "${MCSA_JSON}"; do
    if [ ! -s "${source_file}" ]; then
        echo "ERROR: missing source metadata snapshot: ${source_file}"
        exit 1
    fi
done

JOB_SCRIPT=$(mktemp /tmp/build_ss_adj_loop.XXXXXX.sh)
cat > "${JOB_SCRIPT}" << EOFJOB
#!/bin/bash
#SBATCH --job-name=build_ss_adj_loop
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --nodelist=${NODELIST}
#SBATCH --reservation=${RESERVATION}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/build_ss_adj_loop-%j.out
#SBATCH --error=${LOG_DIR}/build_ss_adj_loop-%j.err

set -euo pipefail
export DEBUG=false
cd ${ROOT}

${PYTHON} scripts/build_loop_dataset.py \
    --out_csv ${SOURCE_CSV} \
    --cif_dir ${OUT_DIR}/structures \
    --sources sabdab stcrdab mcsa \
    --max_resolution 3.0 \
    --max_identity 100 \
    --max_loop_len 25 \
    --sabdab_tsv ${SABDAB_TSV} \
    --stcrdab_tsv ${STCRDAB_TSV} \
    --mcsa_json ${MCSA_JSON} \
    --dry_run

${PYTHON} scripts/prepare_ss_adj_loop_training_metadata.py \
    --source_csv ${SOURCE_CSV} \
    --pn_units_parquet ${DATA_ROOT}/metadata/splits/pn_units_df_train.parquet \
    --pdb_dir ${DATA_ROOT}/pdb \
    --out_csv ${TRAIN_CSV} \
    --max_resolution 3.0

${PYTHON} scripts/split_ss_adj_training_metadata.py \
    --loop_csv ${TRAIN_CSV} \
    --loop_train_csv ${OUT_DIR}/loop_train_dataset.csv \
    --loop_val_csv ${OUT_DIR}/loop_val_dataset.csv \
    --val_fraction 0.10 \
    --seed 42 \
    --filter_parquet ${DATA_ROOT}/metadata/splits/pn_units_df_train.parquet ${OUT_DIR}/standard_splits/pn_units_df_train_no_loop_val.parquet \
    --filter_parquet ${DATA_ROOT}/metadata/splits/interfaces_df_train.parquet ${OUT_DIR}/standard_splits/interfaces_df_train_no_loop_val.parquet

${PYTHON} -c "import pandas as pd; d=pd.read_csv('${OUT_DIR}/loop_train_dataset.csv'); v=pd.read_csv('${OUT_DIR}/loop_val_dataset.csv'); print('train'); print(d.groupby('source').size()); print('val'); print(v.groupby('source').size())"
EOFJOB

JOB=$(sbatch --parsable "${JOB_SCRIPT}")
echo "Submitted curated loop dataset build job ${JOB} to ${NODELIST}"
echo "Output: ${LOG_DIR}/build_ss_adj_loop-${JOB}.out"
