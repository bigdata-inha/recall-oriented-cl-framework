# Recall-Oriented-CL-Framework
Recall-Oriented Continual Learning with Generative Adversarial Meta-Model
## Abstract
The stability-plasticity dilemma is a major challenge in continual learning, as it involves balancing the conflicting objectives of maintaining performance on previous tasks while learning new tasks. In this paper, we propose the recall-oriented continual learning framework to address this challenge. Inspired by the human brain's ability to separate the mechanisms responsible for stability and plasticity, our framework consists of a two-level architecture where an inference network effectively acquires new knowledge and a generative network recalls past knowledge when necessary. In particular, to maximize the stability of past knowledge, we investigate the complexity of knowledge depending on different representations, and thereby introducing generative adversarial meta-model (GAMM) that incrementally learns task-specific parameters instead of input data samples of the task. Through our experiments, we show that our framework not only effectively learns new knowledge without any disruption but also achieves high stability of previous knowledge in both task-aware and task-agnostic  learning scenarios.

## Experiment Command
This repository currently contains experiments reported in the paper for Split CIFAR-10, Split CIFAR-100, Permuted MNIST, 5-Datasets.
All these experiments can be run using the following command:
### Split CIFAR-10
```
python main.py --dataset='cifar10' --nb_cl_f=2 --nb_cl=2 --ts_epochs=120 --ts_lr=0.1  --lr_factor=0.1 --chunk_size=2000 --ra_lambda=5.0
```

### Split CIFAR-100
```
python main.py --dataset='cifar100' --nb_cl_f=10 --nb_cl=10 --ts_epochs=250 --ts_lr=0.1 --lr_factor=0.1 --chunk_size=25000  --ra_lambda=15.0
```

### Split CIFAR-100 using 5-layer AlexNet
```
python main.py --dataset='cifar100_alexnet' --nb_cl_f=10 --nb_cl=10 --ts_epochs=160 --ts_lr=0.1 --lr_factor=0.1 --chunk_size=25000  --ra_lambda=10.0
```

### Permuted MNIST
```
python main_pmnist.py 
```

### 5-Datasets
```
python main_fivedatasets.py
```

### Requirements
python 3.8.5  
pytorch 1.12.0
