#!/usr/bin/env bash

# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="moebert (sixer)"
export WANDB_RUN_GROUP="moebert-mnli"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPU=4
export PYTHONHASHSEED=0

OUTROOT=/data5/vchua/run/d-playround/train-moebert-mnli
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
    --max_seq_length 128 \
    --moebert moe \
    --moebert_distill 5.0 \
    --moebert_expert_num 4 \
    --moebert_expert_dim 768 \
    --moebert_expert_dropout 0.1 \
    --moebert_load_balance 0.0 \
    --moebert_load_importance ./importance_mnli.pkl \
    --moebert_route_method hash-random \
    --moebert_share_importance 512 \
    --fp16 \
    --do_train \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.0 \
    --weight_decay 0.0 \
    --num_train_epochs 5 \
    --seed 0 \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy no \
    --logging_steps 1 \
    --logging_dir $OUTDIR/logdir \
    --report_to tensorboard \
    --output_dir $OUTDIR/model \
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