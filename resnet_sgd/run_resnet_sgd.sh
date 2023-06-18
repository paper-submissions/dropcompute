#bin/bash
#set -x #echo on

# Example: HOSTSFILE=hostsfile_name ./run_resnet_sgd.sh 256 0.8 5

DATA_DIR=/data/pytorch/imagenet/ILSVRC2012
CODE_DIR=/drop_compute/resnet_sgd
BASE_DIR=/drop_compute/resnet_sgd
HOSTSFILE=${HOSTSFILE:=hostsfile}
HOSTSFILE="$BASE_DIR/$HOSTSFILE"
EPOCHS=90
SEED=${SEED:-123}
LOG_FREQ=10

nodes=($(wc -l $HOSTSFILE))
NUM_NODES=${nodes[0]}
# assuming all nodes have the same number.
NGPU_PER_NODE=8

LOCAL_BATCH_SIZE=$1
LR=$2
DROP_RATE=$3
WORLD_SIZE=$(($NUM_NODES*$NGPU_PER_NODE))
TOTAL_BATCH_SIZE=$(($WORLD_SIZE*$LOCAL_BATCH_SIZE))

EXP_DIR="resnet_${WORLD_SIZE}_b_${LOCAL_BATCH_SIZE}_drop_${DROP_RATE}"
TIMESTAMP=$(date +%d-%m-%y_\%H-%M)
RESULTS_DIR="${BASE_DIR}/runs/${EXP_DIR}_${TIMESTAMP}"

CHECKPOINTS_DIR="$RESULTS_DIR/checkpoints"

DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

arr=$(cat $HOSTSFILE)
export IP_LIST=(`echo $arr | sed 's/slots=8/\n/g'`)
ALL_NODES_STR=""
for ((j=0; j<${NUM_NODES}; j++));
    do
        echo Adding IP: ${IP_LIST[$j]}
        ALL_NODES_STR+="${IP_LIST[$j]}:${NGPU_PER_NODE},"
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

export PT_RECIPE_CACHE_PATH="/tmp/recipe_cache"

CMD="
  NUMEXPR_MAX_THREADS=192 \
  MASTER_PORT=12355 \
  MASTER_ADDR=${MASTER_ADDR} \
  PT_HPU_USE_PT_STORE_SYNC=0 \
  mpirun --allow-run-as-root -np ${WORLD_SIZE} \
  --mca btl_tcp_if_include ${MASTER_ADDR}/${WORLD_SIZE} --merge-stderr-to-stdout \
  --prefix ${MPI_ROOT} -H ${ALL_NODES_STR} -x LD_LIBRARY_PATH -x HABANA_LOGS \
  -x PYTHONPATH -x GC_KERNEL_PATH -x MASTER_ADDR -x MASTER_PORT \
  -x PT_HPU_USE_PT_STORE_SYNC -x LOG_LEVEL_ALL_HCL -x PT_RECIPE_CACHE_PATH \
  python ${CODE_DIR}/train.py \
  --data-path=${DATA_DIR} \
  --deterministic \
  --dl-worker-type=HABANA
  --output-dir=${RESULTS_DIR} \
  --arch=resnet50 \
  --device=hpu \
  --autocast \
  --batch-size=${LOCAL_BATCH_SIZE} \
  --epochs=${EPOCHS} \
  --print-freq=${LOG_FREQ} \
  --output-dir=${RESULTS_DIR} \
  --seed=${SEED} \
  --workers=8 \
  --lr=${LR} \
  --drop-rate=${DROP_RATE} "

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
