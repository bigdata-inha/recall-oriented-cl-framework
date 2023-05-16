import os
import argparse
from trainer.trainer import Trainer
from utils.gpu_tools import occupy_memory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ### Basic parameters
    parser.add_argument('--gpu', default='0', help='the index of GPU')
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar10', 'cifar100', 'cifar100_alexnet'])
    parser.add_argument('--num_workers', default=1, type=int, help='the number of workers for loading data')
    parser.add_argument('--random_seed', default=1995, type=int, help='random seed')
    parser.add_argument('--train_batch_size', default=128, type=int, help='the batch size for train loader')
    parser.add_argument('--eval_batch_size', default=128, type=int, help='the batch size for validation loader')
    parser.add_argument('--test_batch_size', default=1, type=int, help='the batch size for test loader')
    parser.add_argument('--disable_gpu_occupancy', default=False, action='store_false', help='disable GPU occupancy')
    parser.add_argument('--nb_cl_fg', default=10, type=int, help='the number of classes in the 0-th phase')
    parser.add_argument('--nb_cl', default=10, type=int, help='the number of classes for each phase')
    parser.add_argument('--lr_factor', default=0.1, type=float, help='learning rate decay factor')
    parser.add_argument('--custom_weight_decay', default=5e-4, type=float, help='weight decay parameter for the optimizer')
    parser.add_argument('--custom_momentum', default=0.9, type=float, help='momentum parameter for the optimizer')
    parser.add_argument('--base_lr1', default=0.1, type=float, help='learning rate for the 0-th phase')

    ### Lightweight model parameters
    parser.add_argument('--ts_epochs', default=120, type=int)  ################################
    parser.add_argument('--ts_lr', default=0.1, type=float, help='learning rate for the student model')

    ### SWAG parameters
    parser.add_argument('--cov_mat', action='store_true', help='save sample covariance')
    parser.add_argument('--no_cov_mat', action='store_false', help='using covariance matrix for swag')
    parser.add_argument('--max_num_models', type=int, default=20, help='maximum number of SWAG models to save')

    ### GAN parameters
    parser.add_argument('--chunk_size', default=2000, type=int, help='the size of chunk output of GAN')
    parser.add_argument('--latent_dim', default=100, type=int, help='the size of latent vector of GAN')
    parser.add_argument('--gan_epochs', default=300, type=int, help='the number of training epoch of GAN')
    parser.add_argument('--gan_batch_size', default=512, type=int, help='the batch size of GAN')
    parser.add_argument('--mse_threshold', default=0.1, type=int, help='')
    parser.add_argument('--ra_lambda', default=5.0, help='')
    parser.add_argument('--gan_lr', default=0.0002, type=float, help='the learning rate of GAN')
    parser.add_argument('--gan_b1', default=0.5, type=float, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--gan_b2', default=0.999, type=float, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--num_critic', default=5, type=int, help='the number of training steps for discriminator per iter')
    parser.add_argument('--lambda_gp', default=10, type=float, help='')
    parser.add_argument('--num_models', default=30, type=float, help='')
    parser.add_argument('--task_ag', default='ent', type=str, help='')


    the_args = parser.parse_args()

    # Checke the number of classes, ensure they are reasonable
    assert(the_args.nb_cl_fg % the_args.nb_cl == 0)
    assert(the_args.nb_cl_fg >= the_args.nb_cl)

    # Print the parameters
    print(the_args)

    # Set GPU index
    os.environ['CUDA_VISIBLE_DEVICES'] = the_args.gpu
    print('Using gpu:', the_args.gpu)

    # Occupy GPU memory in advance
    if the_args.disable_gpu_occupancy:
        occupy_memory(the_args.gpu)
        print('Occupy GPU memory in advance.')

    # Set the trainer and start training
    trainer = Trainer(the_args)
    trainer.train()
