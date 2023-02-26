#!/usr/bin/env bash

# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="moebert (sixer)"
export WANDB_RUN_GROUP="moebert-mnli"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPU=4

OUTROOT=/data5/vchua/run/d-playround/moebert-mnli
WORKDIR=/data5/vchua/dev/d-playround/MoEBERT/examples/text-classification

CONDAROOT=/data5/vchua/miniconda3
CONDAENV=d-playround
# ---------------------------------------------------------------------------------------------

OUTDIR=$OUTROOT/$RUNID

# override label if in dryrun mode
if [[ $1 == "dryrun" ]]; then
    OUTDIR=$OUTROOT/dryrun-${RUNID}
    RUNID=dryrun-${RUNID}
fi

mkdir -p $OUTDIR
cd $WORKDIR

export MASTER_PORT=13888

cmd="
python -m torch.distributed.launch \
    --nproc_per_node $NGPU \
    --master_port $MASTER_PORT \
    run_glue.py \
    --model_name_or_path vuiseng9/bert-base-uncased-mnli \
    --task_name mnli \
    --do_eval \
    --preprocess_importance \
    --max_seq_length 128 \
    --logging_steps 1 \
    --logging_dir $OUTDIR/logdir \
    --output_dir $OUTDIR \
    --overwrite_output_dir \
"

if [[ $1 == "local" ]]; then
    echo "${cmd}" > $OUTDIR/run.log
    echo "### End of CMD ---" >> $OUTDIR/run.log
    cmd="nohup ${cmd}"
    eval $cmd >> $OUTDIR/run.log 2>&1 &
    echo "logpath: $OUTDIR/run.log"
elif [[ $1 == "dryrun" ]]; then
    echo "[INFO: dryrun, add --max_steps 25 to cli"
    cmd="${cmd} --max_steps 25"
    echo "${cmd}" > $OUTDIR/dryrun.log
    echo "### End of CMD ---" >> $OUTDIR/dryrun.log
    eval $cmd >> $OUTDIR/dryrun.log 2>&1 &
    echo "logpath: $OUTDIR/dryrun.log"
else
    source $CONDAROOT/etc/profile.d/conda.sh
    conda activate ${CONDAENV}
    eval $cmd
fi