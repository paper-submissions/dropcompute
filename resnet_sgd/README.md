# ResNet stochastic batch size training
This implements training using stochastic batch size with multinode to simulate DropCompute.

## Requirements
1) Install [PyTorch](https://pytorch.org/)
2) For HPU support, follow [Habana model references](https://github.com/HabanaAI/Model-References/tree/master)
3) Install [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html)

## Training
To train a model -
1) Specify all the nodes in an hostsfile name e.g.
```
user-node-identifier-worker-0 slots=8
user-node-identifier-worker-1 slots=8
user-node-identifier-worker-2 slots=8
user-node-identifier-worker-3 slots=8
```
2) Specify DATA_DIR, CODE_DIR, and BASE_DIR (where the hostsfile is) in run_resnet_sgd.sh
5) Run run_resnet_sgd.sh with 3 arguments - local batch size, learning rate, and drop rate.
```
# Running the script with local batch size of 256, learning rate 0.8 and drop rate 5%
HOSTSFILE=hostsfile_name /path/to/run_resnet_sgd.sh 256 0.8 5
```
