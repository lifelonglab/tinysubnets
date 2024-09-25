# TSN - TinySubNetworks

TSN is a forget-free continual learning algorithm with low memory consumption. 
It has following features:
- it adapts mask optimization per task to minimize the bias between tasks
- it incorporates cluster-based nonlinear quantization to reduce the memory consumption per each task
- the further compression (mask compression) can be performed by huffman encoding
- it can increase original sparsity levels by non-gradient post-training pruning
- the tasks can work on model duplicates if the KL divergence between tasks is very high

## Installation and configuration

* Dependencies

```
pip install numpy
pip install scipy
pip install torch
pip install torchvision
pip install matplotlib
pip install seaborn
pip install pandas
```

* Most important input parameters
- training epochs 
- train and test batch size
- learning rate (starting lr and minimum lr)
- starting sparsity level
- replay memory on/off
- KL divergence threshold (when tasks can have seperate copy of the model)
- replay memory size
- quantization mode on/off
- quantization bit for activations and weights
- number of clusters for non-linear weights quantization (weight capacity reduction)
- post-pruning iterations
- sparsity scaler - parameter for increasing sparsity in post-training pruning
- threshold for accuracy drop in post-pruning process

## Usage for three examples for CIFAR100, PMNIST and Tiny Imagenet:
python main_tsn_cifar100.py --dataset cifar100_100
python main_tsn_pmnist.py
python main_tsn_tiny_imagenet.py



