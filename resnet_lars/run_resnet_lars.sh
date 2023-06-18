#!/bin/bash

function print_synopsis()
{
    cat << EOF
NAME
        `basename $0`

SYNOPSIS
        `basename $0` [-c <config>] [-ld <log-dir>] [-wd <work-dir>] [-dd <data-dir>] [-h]

DESCRIPTION
        Runs 8-gaudi local MLPerf Resnet training on PyTorch.

        -c <config-file>, --config <config-file>
            configuration file containing series of "export VAR_NAME=value" commands
            overrides default settings for Resnet training

        -ld <log-dir>, --log-dir <log-dir>
            specify the loggin directory, used to store mllogs and outputs from all mpi processes

        -wd <work-dir>, --work-dir <work-dir>
            specify the work directory, used to store temporary files during the training

        -dd <data-dir>
            specify the data directory, containing the ImageNet dataset

        -h, --help
            print this help message

EXAMPLES
       `basename $0` -wd /data/imagenet
            MLPerf Resnet training on dataset stored in /data/imagenet

EOF
}

function parse_config()
{
    while [ -n "$1" ]; do
        case "$1" in
            -c | --config )
                CONFIG_FILE=$2
                if [[ -f ${CONFIG_FILE} ]]; then
	                source $CONFIG_FILE
                    return
                else
                    echo "Could not find ${CONFIG_FILE}"
                    exit 1
                fi
                ;;
            * )
                shift
                ;;
        esac
    done
}

function parse_args()
{
    while [ -n "$1" ]; do
        case "$1" in
            -c | --config )
                shift 2
                ;;
            -ld | --log-dir )
                LOG_DIR=$2
                shift 2
                ;;
            -wd | --work-dir )
                WORK_DIR=$2
                shift 2
                ;;
            -dd | --data-dir )
                DATA_DIR=$2
                shift 2
                ;;
            -h | --help )
                print_synopsis
                exit 0
                ;;
            * )
                echo "error: invalid parameter: $1"
                print_synopsis
                exit 1
                ;;
        esac
    done
}

# Default setting for Pytorch Resnet trainig

export HABANA_PROFILE=0

DATA_DIR=/data/pytorch/imagenet/ILSVRC2012/
BASE_DIR=/drop_compute/resnet_lars
NUM_WORKERS_PER_HLS=8
HOSTSFILE=${HOSTSFILE:=hostsfile}
HOSTSFILE="$BASE_DIR/$HOSTSFILE"
nodes=($(wc -l $HOSTSFILE))
NUM_NODES=${nodes[0]}
WORLD_SIZE=$(($NUM_NODES*$NUM_WORKERS_PER_HLS))
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-256}
DROP_RATE=${DROP_RATE:-0}

EXP_DIR="resnet_${WORLD_SIZE}_b_${LOCAL_BATCH_SIZE}_drop_${DROP_RATE}"
TIMESTAMP=$(date +%d-%m-%y_\%H-%M)
RESULTS_DIR="${BASE_DIR}/runs/${EXP_DIR}_${TIMESTAMP}"

arr=$(cat $HOSTSFILE)
export IP_LIST=(`echo $arr | sed 's/slots=8/\n/g'`)
ALL_NODES_STR=""
for ((j=0; j<${NUM_NODES}; j++));
    do
        echo Adding IP: ${IP_LIST[$j]}
        ALL_NODES_STR+="${IP_LIST[$j]}:${NUM_WORKERS_PER_HLS},"
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

EVAL_OFFSET_EPOCHS=3
EPOCHS_BETWEEN_EVALS=4
DISPLAY_STEPS=1000

NUM_WORKERS=8
TRAIN_EPOCHS=35
LARS_DECAY_EPOCHS=36
WARMUP_EPOCHS=3
BASE_LEARNING_RATE=9
END_LEARNING_RATE=0.0001
WEIGHT_DECAY=0.00005
LR_MOMENTUM=0.9
LABEL_SMOOTH=0.1
STOP_THRESHOLD=0.759

WORK_DIR=/tmp/resnet50
LOG_DIR=/tmp/resnet_log
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

MPI_PATH=/opt/amazon/openmpi

# apply optional config, overwriting default settings
parse_config "$@"

# optional command line arguments overwrite both default and config settings
parse_args "$@"

# prepare directories
rm -rf $LOG_DIR
mkdir -p $WORK_DIR
mkdir -p $LOG_DIR

# run Pytorch Resnet training
CMD="
MASTER_PORT=12355 \
MASTER_ADDR=${MASTER_ADDR} \
PT_HPU_USE_PT_STORE_SYNC=0 \
mpirun \
--allow-run-as-root \
--np ${WORLD_SIZE} \
--mca btl_tcp_if_include ${MASTER_ADDR}/${WORLD_SIZE} \
-H ${ALL_NODES_STR} -x LD_LIBRARY_PATH \
-x PYTHONPATH -x GC_KERNEL_PATH -x MASTER_ADDR -x MASTER_PORT \
-x PT_HPU_USE_PT_STORE_SYNC \
--merge-stderr-to-stdout \
--prefix $MPI_PATH \
python $SCRIPT_DIR/PyTorch/train.py \
--model resnet50 \
--device hpu \
--print-freq $DISPLAY_STEPS \
--channels-last False \
--dl-time-exclude False \
--output-dir $WORK_DIR \
--log-dir $RESULTS_DIR \
--data-path $DATA_DIR \
--eval_offset_epochs $EVAL_OFFSET_EPOCHS \
--epochs_between_evals $EPOCHS_BETWEEN_EVALS \
--workers $NUM_WORKERS_PER_HLS \
--batch-size $LOCAL_BATCH_SIZE \
--epochs $TRAIN_EPOCHS \
--lars_decay_epochs $LARS_DECAY_EPOCHS \
--warmup_epochs $WARMUP_EPOCHS \
--base_learning_rate $BASE_LEARNING_RATE \
--end_learning_rate $END_LEARNING_RATE \
--weight-decay $WEIGHT_DECAY \
--momentum $LR_MOMENTUM \
--label-smoothing $LABEL_SMOOTH \
--target_accuracy $STOP_THRESHOLD \
--hmp \
--hmp-bf16 $SCRIPT_DIR/PyTorch/ops_bf16_Resnet.txt \
--hmp-fp32 $SCRIPT_DIR/PyTorch/ops_fp32_Resnet.txt \
--dl-worker-type HABANA"
echo $CMD
sleep 5
eval $CMD

# finalize LOG_DIR folder
chmod -R 777 ${LOG_DIR}
exit 0
