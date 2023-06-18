#bin/bash
#set -x #echo on

# Example: HOSTSFILE=hostsfile_name ./run_pretraining_phase_2.sh 64 16 -1

export PDSH_SSH_ARGS_APPEND="-p 3022"

DATA_DIR=/data/pytorch/bert/pretraining/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus
CODE_DIR=/drop_compute/bert_large_pretraining
BASE_DIR=/drop_compute/bert_large_pretraining
HOSTSFILE=${HOSTSFILE:=hostsfile}
HOSTSFILE="$BASE_DIR/$HOSTSFILE"
MAX_SEQ_LENGTH=512
NUM_STEPS_PER_CP=2000
MAX_STEPS=1563
LR=0.004
WARMUP=0.128
MAX_PRED=80

nodes=($(wc -l $HOSTSFILE))
NUM_NODES=${nodes[0]}
# assuming all nodes have the same number.
NGPU_PER_NODE=8

LOCAL_BATCH_SIZE=$1
MICRO_BATCH_SIZE=$2
COMPUTE_THRESHOLD=$3
WORLD_SIZE=$(($NUM_NODES*$NGPU_PER_NODE))
TOTAL_BATCH_SIZE=$(($WORLD_SIZE*$LOCAL_BATCH_SIZE))
GRAD_ACCUMULATION=$(($LOCAL_BATCH_SIZE / $MICRO_BATCH_SIZE))

EXP_DIR="bert-large_n_${WORLD_SIZE}_b_${LOCAL_BATCH_SIZE}_acc_${GRAD_ACCUMULATION}_drop_${COMPUTE_THRESHOLD//./-}"
TIMESTAMP=$(date +%d-%m-%y_\%H-%M)
RESULTS_DIR="${BASE_DIR}/runs/${EXP_DIR}_${TIMESTAMP}"

CHECKPOINTS_DIR="$RESULTS_DIR/checkpoints"
INIT_CHECKPOINT_PATH=${INIT_CHECKPOINT_PATH:=""}

DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

arr=$(cat $HOSTSFILE)
export IP_LIST=(`echo $arr | sed 's/slots=8/\n/g'`)
ALL_NODES_STR=""
for ((j=0; j<${NUM_NODES}; j++));
    do
        echo Adding IP: ${IP_LIST[$j]}
        ALL_NODES_STR+="${IP_LIST[$j]}:8,"
done

ALL_NODES_STR=${ALL_NODES_STR::-1}  #remove last ","
MASTER_ADDR=$(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p)
if [ -z "$MASTER_ADDR" ] # check that MASTER_ADDR holds IP
then
      echo "\$MASTER_ADDR is empty"
      exit 0
else
      echo MASTER_ADDR - $MASTER_ADDR
fi

if [ -z "$INIT_CHECKPOINT_PATH" ]
then
      INIT_CHECKPOINT=""
else
      INIT_CHECKPOINT="--init_checkpoint=$INIT_CHECKPOINT_PATH"
fi

CMD="
  MASTER_PORT=12355 \
  MASTER_ADDR=${MASTER_ADDR} \
  PT_HPU_USE_PT_STORE_SYNC=0 \
  mpirun --allow-run-as-root -np ${WORLD_SIZE} \
  --mca btl_tcp_if_include ${MASTER_ADDR}/${WORLD_SIZE} --merge-stderr-to-stdout
  --prefix ${MPI_ROOT} -H ${ALL_NODES_STR} -x LD_LIBRARY_PATH \
  -x PYTHONPATH -x GC_KERNEL_PATH -x MASTER_ADDR -x MASTER_PORT \
  -x PT_HPU_USE_PT_STORE_SYNC \
  python ${CODE_DIR}/run_pretraining.py \
  --do_train \
  --bert_model=bert-large-uncased \
  --hmp \
  --hmp_bf16=${CODE_DIR}/ops_bf16_bert.txt \
  --hmp_fp32=${CODE_DIR}/ops_fp32_bert.txt \
  --config_file=${CODE_DIR}/bert_config.json \
  --use_habana \
  --allreduce_post_accumulation \
  --allreduce_post_accumulation_fp16 \
  --log-dir=${RESULTS_DIR} \
  --output_dir=${CHECKPOINTS_DIR} \
  --use_fused_lamb \
  --input_dir=${DATA_DIR} \
  --train_batch_size=${LOCAL_BATCH_SIZE} \
  --max_seq_length=${MAX_SEQ_LENGTH} \
  --max_predictions_per_seq=${MAX_PRED} \
  --warmup_proportion=${WARMUP} \
  --max_steps=${MAX_STEPS} \
  --num_steps_per_checkpoint=${NUM_STEPS_PER_CP} \
  --learning_rate=${LR} \
  --gradient_accumulation_steps=${GRAD_ACCUMULATION} \
  --enable_packed_data_mode False \
  --phase1_end_step=7038 \
  --phase2 \
  --resume_step=7038 \
  --resume_from_checkpoint \
  --disable_progress_bar \
  --compute_threshold=${COMPUTE_THRESHOLD} \
  ${INIT_CHECKPOINT} \
  "

mkdir -p $RESULTS_DIR
mkdir -p $CHECKPOINTS_DIR
git config --global --add safe.directory $CODE_DIR
echo $CODE_DIR > $RESULTS_DIR/run.config
git -C $CODE_DIR log --format="commit id - %H (%ad)" -n 1 > $RESULTS_DIR/run.config
cat $HOSTSFILE >> $RESULTS_DIR/run.config
echo $CMD >> $RESULTS_DIR/run.config


echo "Experiment directory - ${RESULTS_DIR}"
echo $CMD
sleep 7

eval $CMD 2>&1 | tee ${RESULTS_DIR}/output.log
echo Train script done