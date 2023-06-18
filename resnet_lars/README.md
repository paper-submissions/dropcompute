# ResNet stochastic batch size training
This implements training using stochastic batch size with multinode to simulate DropCompute.

## Requirements
1) Install [PyTorch](https://pytorch.org/)
2) For HPU support, follow [Habana model references](https://github.com/HabanaAI/Model-References/tree/master)
3) Follow ResNet50 [MLPERF instructions](https://github.com/mlcommons/training_results_v2.1/tree/main/Intel-HabanaLabs/benchmarks)

## Training
To train a model -
1) Specify all the nodes in an hostsfile e.g.
```
user-node-identifier-worker-0 slots=8
user-node-identifier-worker-1 slots=8
user-node-identifier-worker-2 slots=8
user-node-identifier-worker-3 slots=8
```
2) Specify DATA_DIR and BASE_DIR (where the hostsfile is) in run_resnet_sgd.sh
5) Run run_resnet_sgd.sh
