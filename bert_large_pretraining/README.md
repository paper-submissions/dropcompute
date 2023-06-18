# BERT-Large pretraining

This implements BERT-Large pretraining with DropCompute

## Requirements
1) Install [PyTorch](https://pytorch.org/)
2) For HPU support, follow [Habana model references](https://github.com/HabanaAI/Model-References/tree/master)
3) Follow [BERT pretraining instructions](https://github.com/HabanaAI/Model-References/tree/master/PyTorch/nlp/bert)

## Training
To train a model -
1) Specify all the nodes in an hostsfile e.g.
```
user-node-identifier-worker-0 slots=8
user-node-identifier-worker-1 slots=8
user-node-identifier-worker-2 slots=8
user-node-identifier-worker-3 slots=8
```
2) Specify DATA_DIR, CODE_DIR, and BASE_DIR (where the hostsfile is) in run_pretraining_phase_1.sh and run_pretraining_phase_2.sh
5) For phase-1 training, run run_pretraining_phase_1.sh. For phase-2 training, run run_pretraining_phase_2.sh.
Specify 3 arguments - local batch size, micro batch size (for gradient accumulations), and compute threshold in seconds.
```
# Phase-1, local batch size 64, micro batch size 64, no drop compute
HOSTSFILE=hostsfile_name ./run_pretraining_phase_1.sh 64 16 -1
# Phase-2, local batch size 64, micro batch size 16, no drop compute
HOSTSFILE=hostsfile_name ./run_pretraining_phase_2.sh 64 16 -1
# Phase-1, local batch size 64, micro batch size 64, compute threshold of 2.5 seconds
HOSTSFILE=hostsfile_name ./run_pretraining_phase_1.sh 64 16 2.5
``` 
