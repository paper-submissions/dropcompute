#!/bin/bash
#set -x #echo on

##########################################################################################
# Example: Pretraining phase 1 of BERT with 1.5B parameters on multinode with 8 card each
##########################################################################################

DATA_DIR=$HL_DATA_DIR_ROOT/data/pytorch/bert/pretraining/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus
MODEL_CONFIG=${HL_MODEL_CONFIG:-./scripts/bert_1.5b_config.json}
DS_CONFIG=${HL_DS_CONFIG:-./scripts/deepspeed_config_bert_1.5b.json}
NEW_DS_CONFIG=${HL_DS_CONFIG:-./scripts/temp_config.json}
HOSTSFILE=${HL_HOSTSFILE:-./scripts/hostsfile}

MAX_SEQ_LENGTH=128
NUM_STEPS_PER_CP=200
MAX_STEPS=155000
RUN_STEPS=${HL_RUN_STEPS:--1}
LR=0.0015
WARMUP=0.05
CONST=0.25
LOG_FREQ=1
MAX_PRED=80

# Params: DeepSpeed
NUM_NODES=${HL_NUM_NODES:-4}
NGPU_PER_NODE=8


COMPUTE_THRESHOLD=$3
LOCAL_BATCH_SIZE=$1
MICRO_BATCH_SIZE=$2
DEBUG=$4
WORLD_SIZE=$(($NUM_NODES*$NGPU_PER_NODE))
TOTAL_BATCH_SIZE=$(($WORLD_SIZE*$LOCAL_BATCH_SIZE))

# Parse deepspeed config file using python
cp $DS_CONFIG $NEW_DS_CONFIG
jq ".train_batch_size = ${TOTAL_BATCH_SIZE}" $NEW_DS_CONFIG > ${NEW_DS_CONFIG}_tmp && mv ${NEW_DS_CONFIG}_tmp $NEW_DS_CONFIG
jq ".train_micro_batch_size_per_gpu = ${MICRO_BATCH_SIZE}" $NEW_DS_CONFIG > ${NEW_DS_CONFIG}_tmp && mv ${NEW_DS_CONFIG}_tmp $NEW_DS_CONFIG

#jq '.tensorboard.output_path = $RESULTS_DIR/tensorboard' $NEW_DS_CONFIG > ${NEW_DS_CONFIG}_tmp && mv ${NEW_DS_CONFIG}_tmp $NEW_DS_CONFIG
GRAD_ACCUMULATION=$(($LOCAL_BATCH_SIZE / $MICRO_BATCH_SIZE))

host_name=$(hostname)
host_name_split=(${host_name//-/ })
cluster_name=${host_name_split[3]}

if [ $cluster_name == "c06" ] || [ $cluster_name == "c07" ]
then
    device="hpu-2"
else
    device="hpu-1"
fi

EXP_DIR="n_${WORLD_SIZE}_b_${LOCAL_BATCH_SIZE}_acc_${GRAD_ACCUMULATION}_drop_${COMPUTE_THRESHOLD//./-}_$device"
TIMESTAMP=$(TZ='Asia/Jerusalem' date +%d-%m-%y_\%H-%M)
RESULTS_DIR="runs_deepspeed/${EXP_DIR}_${TIMESTAMP}"

if [ -z "$DEBUG" ] # set DEBUG to False by default
then
        DEBUG=""
else
        DEBUG="--debug"
        RESULTS_DIR="${RESULTS_DIR}_debug"
        MAX_STEPS=40
fi

CHECKPOINTS_DIR="$RESULTS_DIR/checkpoints"

DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

CMD="python -u ./run_pretraining.py \
     --use_hpu \
     --disable_progress_bar \
     --optimizer=lans \
     --use_lr_scheduler \
     --resume_from_checkpoint \
     --do_train \
     --bert_model=bert-base-uncased \
     --config_file=$MODEL_CONFIG \
     --json-summary=$RESULTS_DIR/dllogger.json \
     --output_dir=$CHECKPOINTS_DIR \
     --seed=12439 \
     --input_dir=$DATA_DIR \
     --max_seq_length $MAX_SEQ_LENGTH \
     --max_predictions_per_seq=$MAX_PRED \
     --max_steps=$MAX_STEPS \
     --steps_this_run=$RUN_STEPS \
     --num_steps_per_checkpoint=$NUM_STEPS_PER_CP \
     --learning_rate=$LR \
     --warmup_proportion=$WARMUP \
     --constant_proportion=$CONST \
     --scheduler_degree=1.0 \
     --log_freq=$LOG_FREQ \
     --tensor_logger_path=$RESULTS_DIR \
     --deepspeed \
     --compute_threshold=$COMPUTE_THRESHOLD \
     ${DEBUG} \
     --skip_checkpoint \
     --deepspeed_config=$NEW_DS_CONFIG \
     --delay_factor=1.0"

#Configure multinode

MASTER_ADDR=$(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p)
if [ -z "$MASTER_ADDR" ] # check that MASTER_ADDR holds IP
then
      echo "\$MASTER_ADDR is empty"
      exit 0
else
      echo MASTER_ADDR - $MASTER_ADDR
fi

if [ "$NUM_NODES" -ne "1" -a -f "$HOSTSFILE" ]
then
    MULTINODE_CMD="--hostfile=$HOSTSFILE --master_addr $MASTER_ADDR "
fi

mkdir -p $RESULTS_DIR
git config --global --add safe.directory /software/users/sgottlieb/dist_work_new
git -C /software/users/sgottlieb/dist_work_new/ log --format="Version - %H (%ad)" -n 1 > $RESULTS_DIR/run.config
cluster=$(hostname | grep -oh -E 'c0[1-8]{1}')
echo Cluster: ${cluster} Nodes: ${NUM_NODES} HPUs per node: ${NGPU_PER_NODE}  >> $RESULTS_DIR/run.config
cat $HOSTSFILE >> $RESULTS_DIR/run.config
echo $CMD >> $RESULTS_DIR/run.config


echo "#################################################"
echo MODEL: Bert-1.5B NODES_NUMBER: "$NUM_NODES" BATCH: "$TOTAL_BATCH_SIZE" GRAD_ACCUMULATION: "$GRAD_ACCUMULATION" COMPUTE_THRESHOLD: "$COMPUTE_THRESHOLD"
echo "Experiment directory - ${RESULTS_DIR}"
echo $CMD
sleep 10

deepspeed --num_nodes ${NUM_NODES} \
          --num_gpus ${NGPU_PER_NODE} \
          --no_local_rank \
          --no_python \
          --no_ssh_check \
          $MULTINODE_CMD \
          $CMD 2>&1 | tee ${RESULTS_DIR}/output.log
