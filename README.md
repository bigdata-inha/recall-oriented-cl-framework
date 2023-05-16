# Recall-Oriented-CL-Framework
Recall-Oriented Continual Learning with Generative Adversarial Meta-Model
<p align="center">
  <img src="https://github.com/haneol0415/recall-oriented-cl-framework/assets/61872888/f547bb26-916b-4cf9-98ca-0ff1ba83d229">
</p>

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
