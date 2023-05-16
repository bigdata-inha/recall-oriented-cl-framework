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
```python
python main.py --dataset='cifar100' --nb_cl_f=10 --nb_cl=10 --ts_epochs=250 --ts_lr=0.1 --lr_factor=0.1 --chunk_size=25000  --ra_lambda=15.0
```

### Split CIFAR-100 using 5-layer AlexNet
```python
python main.py --dataset='cifar100_alexnet' --nb_cl_f=10 --nb_cl=10 --ts_epochs=160 --ts_lr=0.1 --lr_factor=0.1 --chunk_size=25000  --ra_lambda=10.0
```

### Permuted MNIST
```python
python main_pmnist.py 
```

### 5-Datasets
```python
python main_fivedatasets.py
```

```
python3 train_resnet_bbb.py --kl_scale=0 --kl_schedule=0 --regularizer=mse --train_sample_size=1 --val_sample_size=1 --momentum=-1 --chmlp_chunk_size=7000 --beta=50 --cl_scenario=3 --split_head_cl3 --num_tasks=5 --num_classes_per_task=2 --batch_size=32 --epochs=40 --lr=0.0005 --use_adam --clip_grad_norm=-1 --net_type=resnet --resnet_block_depth=5 --resnet_channel_sizes=16,16,32,64 --hnet_type=chunked_hmlp --hmlp_arch= --cond_emb_size=16 --chunk_emb_size=16 --hnet_net_act=sigmoid --std_normal_temb=0.01 --std_normal_emb=1.0 --mean_only
```
