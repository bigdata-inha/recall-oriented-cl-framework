import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import random
import pdb
import argparse, time
import math
from copy import deepcopy
from copy import deepcopy
from models.gan import *
from swag.posteriors import *
from swag.swag_utils import *
from dataloader.five_datasets_loader import get
from models.fivedataset_resnet18 import ResNet18
from models.lightweight_fivedataset_resnet18 import lightweight_ResNet18


def train(args, model, device, x, y, optimizer, criterion):
    model.train()
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0, len(r), args.batch_size_train):
        if i + args.batch_size_train <= len(r):
            b = r[i:i+args.batch_size_train]
        else:
            b = r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def swag_train(args, ts_model, swag_model, device, epoch, x, y, student_optimizer, criterion):

    ts_model.train
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)

    # Loop batches
    for i in range(0, len(r), args.batch_size_train):
        if i + args.batch_size_train <= len(r):
            b = r[i:i + args.batch_size_train]
        else:
            b = r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        student_optimizer.zero_grad()
        ts_outputs = ts_model(data)

        loss = criterion(ts_outputs, target)
        loss.backward()
        student_optimizer.step()

        # SWAG Collection
        if epoch >= int(args.n_epochs // 2):
            swag_model.collect_model(ts_model)


def test(args, model, device, x, y, criterion):
    model.eval()
    total_loss = 0
    total_num = 0
    correct = 0
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0, len(r), args.batch_size_test):
            if i + args.batch_size_test <= len(r):
                b = r[i:i + args.batch_size_test]
            else:
                b = r[i:]
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item() * len(b)
            total_num += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc


def swag_test(args, swag_model, device, x, y, criterion):
    if args.no_cov_mat==True:
        cov_mat = False
    else:
        cov_mat = True
    swag_model.sample(scale=1.0, cov=cov_mat, device=device)
    swag_model.eval()
    total_loss = 0
    total_num = 0
    correct = 0
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)

    with torch.no_grad():
        # Loop batches
        for i in range(0, len(r), args.batch_size_test):
            if i + args.batch_size_test <= len(r):
                b = r[i:i + args.batch_size_test]
            else:
                b = r[i:]
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = swag_model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item() * len(b)
            total_num += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num

    return final_loss, acc


def get_model_set(args, generator, swag_model, device, model_size, chunk_size, cov_mat, inputsize, curr_task, ncla):
    model_set = {}
    original_num_chunks = math.ceil(model_size / chunk_size)
    _, remainder, mu, sigma = swag_model.chunk_sample(1.0, cov_mat, chunk_size, original_num_chunks)
    for i in range(curr_task+1):
        model_set[i] = []
    for model_id in range(curr_task+1):
        for _ in range(args.num_models):
            chunkid = torch.arange(original_num_chunks * model_id, original_num_chunks * (model_id + 1)).to(device)
            z = torch.rand(original_num_chunks, args.latent_dim).to(device)
            fake_parameter = generator(z, chunkid).to('cpu')
            fake_parameter = sigma * fake_parameter + mu  # de-normalization
            model_structure = Net(inputsize, args.n_hidden, ncla).to(device)
            fake_parameter = fake_parameter.view(-1)[:-remainder]
            new_model = param_insert(model_structure, fake_parameter).to(device)
            model_set[model_id].append(new_model)

    return model_set


def get_entropy(args, model_set, data, curr_task):
    ent_set = torch.zeros(curr_task+1)
    for i in range(len(model_set)):
        prob_sum = 0
        for model_sample in model_set[i]:
            model_sample.eval()
            outputs = model_sample(data)
            prob = F.softmax(outputs, dim=1)
            prob_sum += prob
        prob_mean = prob_sum / args.num_models
        ent = -torch.sum(prob_mean * torch.log(prob_mean))
        ent_set[i] = ent
    _, t_id = ent_set.view(-1, curr_task+1).min(1)

    return t_id


def test_task_agnostic(args, model_set, device, x, y, criterion, curr_task):
    total_loss = 0
    total_num = 0
    correct = 0
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0, len(r), args.batch_size_test_t_ag):
            if i + args.batch_size_test_t_ag <= len(r):
                b = r[i:i + args.batch_size_test_t_ag]
            else:
                b = r[i:]
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            # inference task ID
            t_id = get_entropy(args, model_set, data, curr_task)
            new_model = model_set[int(t_id)][int(torch.randint(0, args.num_models, (1,)))] # int(torch.randint(0, args.num_models, (1,)))
            new_model.eval()
            output = new_model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item() * len(b)
            total_num += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num

    return final_loss, acc


def gan_training(args, iter, ncla, inputsize, chunk_size, swag_model, model_size, data, prev_g=None, capa_exceed=False,
                 device=None):

    MSELoss = nn.MSELoss()
    Crossentropy = torch.nn.CrossEntropyLoss()
    gan_batch_size = args.gan_batch_size
    gan_epochs = args.gan_epochs
    latent_dim = args.latent_dim
    ra_lambda = args.ra_lambda
    if args.no_cov_mat==True:
        cov_mat = False
    else:
        cov_mat = True
    # num_model_sample = self.args.num_model_sample

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if prev_g is not None:
        prev_g.eval()

    # Number of chunks
    if capa_exceed == True:     # chunk_size == 2000 / capa_exceed == True
        prev_chunk_size = chunk_size / 2
        prev_num_chunks = math.ceil(model_size / prev_chunk_size)
    original_num_chunks = math.ceil(model_size / chunk_size)
    num_chunks = original_num_chunks * (iter + 1)   # base_num_chunks = original_num_chunks - num_saved_chunks

    _, _, mu, sigma = swag_model.chunk_sample(1.0, cov_mat, chunk_size, original_num_chunks)

    # Initialize G and D
    generator = Generator(latent_dim, iter, chunk_size, num_chunks).to(device)
    discriminator = Discriminator(iter, chunk_size, num_chunks).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.gan_lr,
                                   betas=(args.gan_b1, args.gan_b2))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.gan_lr,
                                   betas=(args.gan_b1, args.gan_b2))

    # GAN model size
    generator_model_size = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    discriminator_model_size = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print('G model size: {:d}, D model size: {:d}'.format(generator_model_size, discriminator_model_size))

    # Training phase
    ChunksMSE = torch.zeros(iter + 1, original_num_chunks)  # MSE between generated parameters and real parameters
    for epoch in range(gan_epochs):
        print('Task: {:d}, Epoch: {:d}, learning rate: {:.4f}, Chunk size: {:d}'.format(
            iter, epoch, args.gan_lr, chunk_size))
        # Training iteration in one epoch
        for i in range(50000 * (iter + 1) // gan_batch_size):
            sampled_model, remainder, means, stds = swag_model.chunk_sample(1.0, cov_mat, chunk_size,
                                                                            original_num_chunks)

            # Training Discriminator
            discriminator.train()
            generator.eval()
            # Create (real sample, condition) pair for training cGAN
            if iter == 0:
                chunkid = torch.randint(num_chunks, size=(gan_batch_size,)).to(device)
                real_parameter = sampled_model[chunkid].to(device)

            # Pseudo-rehearsal: considering generated parameter as real one sampled from BNN
            else:
                chunkid = torch.randint(num_chunks, size=(gan_batch_size,))

                curr_idx = np.where(chunkid >= original_num_chunks * iter)
                prev_idx = np.where(chunkid < original_num_chunks * iter)

                prev_chunkid = chunkid[prev_idx].to(device)
                curr_chunkid = chunkid[curr_idx].to(device)
                chunkid = torch.cat([prev_chunkid, curr_chunkid]).to(device)
                if capa_exceed == False:    # chunk_size == 1000 / capa_exceed == False
                    num_generated = prev_chunkid.size(0)
                    z_generated = torch.randn(num_generated, latent_dim).to(device)
                    param_generated = prev_g(z_generated, prev_chunkid)
                # When GAN expanded
                else:
                    prev_model_list = []
                    for prev_t in range(iter):
                        prev_g_chunkid = torch.arange(prev_num_chunks * prev_t,
                                                      prev_num_chunks * (prev_t + 1)).to(device)
                        z_ = torch.randn(prev_num_chunks, latent_dim).to(device)
                        prev_model = prev_g(z_, prev_g_chunkid).detach().view(1, -1)

                        difference = int(original_num_chunks * chunk_size - prev_num_chunks * prev_chunk_size)
                        if difference >= 0:
                            zeros = torch.zeros(1, difference).to(device)
                            prev_model = torch.cat((prev_model, zeros), dim=1).view(-1, chunk_size)
                        else:
                            prev_model = prev_model[0][:difference].view(-1, chunk_size)
                        prev_model_list.append(prev_model)
                    prev_model = torch.cat(prev_model_list)
                    param_generated = prev_model[prev_chunkid]

                param_sampled = sampled_model[curr_chunkid-(original_num_chunks*iter)].to(device)
                real_parameter = torch.cat([param_generated, param_sampled], dim=0)

                shuffle_idx = torch.randperm(gan_batch_size)
                chunkid = chunkid[shuffle_idx]
                real_parameter = real_parameter[shuffle_idx]

            z = torch.randn(gan_batch_size, latent_dim).to(device)

            d_optimizer.zero_grad()

            fake_parameter = generator(z, chunkid)
            real_validity = discriminator(real_parameter, chunkid)
            fake_validity = discriminator(fake_parameter, chunkid)
            gradient_penalty = compute_gradient_penalty(discriminator, real_parameter.data, fake_parameter.data,
                                                        chunkid, device)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + args.lambda_gp * gradient_penalty
            d_loss.backward()
            d_optimizer.step()

            # Training Generator
            g_optimizer.zero_grad()
            if i % args.num_critic == 0:
                discriminator.eval()
                generator.train()
                fake_parameter = generator(z, chunkid)
                fake_validity = discriminator(fake_parameter, chunkid)

                if iter == 0:
                    replay_alignment = 0
                # Replay alignment
                else:
                    prev_idx = torch.where(chunkid < original_num_chunks * iter)
                    prev_chunkid = chunkid[prev_idx].to(device)
                    num_generated = prev_chunkid.size(0)
                    z_generated = torch.randn(num_generated, latent_dim).to(device)
                    fake_curr = generator(z_generated, prev_chunkid)

                    if capa_exceed == False:  # chunk_size == 1000 / capa_exceed == False
                        fake_prev = prev_g(z_generated, prev_chunkid).detach()

                    # When GAN expanded
                    else:
                        fake_prev_list = []
                        for idx in range(num_generated):
                            z_ = z_generated[idx].view(1,-1)
                            z_cmt = deepcopy(z_)
                            for _ in range(prev_num_chunks-1):
                                z_ = torch.cat((z_, z_cmt))
                            prev_model_list = []
                            for prev_t in range(iter):
                                prev_g_chunkid = torch.arange(prev_num_chunks * prev_t,
                                                              prev_num_chunks * (prev_t + 1)).to(device)
                                prev_model = prev_g(z_, prev_g_chunkid).detach().view(1, -1)
                                difference = int(original_num_chunks*chunk_size - prev_num_chunks*prev_chunk_size)
                                if difference >= 0:
                                    zeros = torch.zeros(1, difference).to(device)
                                    prev_model = torch.cat((prev_model, zeros), dim=1).view(-1, chunk_size)
                                else:
                                    prev_model = prev_model[0][:difference].view(-1, chunk_size)
                                prev_model_list.append(prev_model)
                            prev_model = torch.cat(prev_model_list)
                            fake_prev = prev_model[prev_chunkid[idx]].view(1, -1)
                            fake_prev_list.append(fake_prev)
                        fake_prev = torch.cat(fake_prev_list, dim=0)

                    replay_alignment = MSELoss(fake_curr, fake_prev)
                    prev_model = []
                    fake_prev = []
                    prev_model_list = []

                g_loss = -torch.mean(fake_validity) + ra_lambda * replay_alignment
                g_loss.backward()
                g_optimizer.step()

        # Evaluation phase
        if (epoch+1) % 100 == 0:
            # BNN sampling only for checking chunk MSE
            sampled_model_list = []
            for iteration_ in range(iter + 1):
                swag_model = torch.load('./models_saved/five_dataset_swag_model_'+str(iteration_)+'.pt')
                sampled_model, remainder, means, stds = swag_model.chunk_sample(1.0, cov_mat, chunk_size,
                                                                                original_num_chunks)
                sampled_model = stds * sampled_model + means    # de-normalization
                sampled_model_list.append(sampled_model)
            sampled_model_list = torch.cat(sampled_model_list).to('cpu')

            generator.eval()
            for t in range(iter + 1):
                chunkid = torch.arange(original_num_chunks*t, original_num_chunks*(t+1)).to(device)
                z = torch.rand(original_num_chunks, args.latent_dim).to(device)
                fake_parameter = generator(z, chunkid).to('cpu')
                fake_parameter = sigma * fake_parameter + mu    # de-normalization

                # Calculating MSE for each chunks
                chunk_mse = torch.mean(torch.square(
                    fake_parameter - sampled_model_list[original_num_chunks * t:original_num_chunks * (t + 1)]),
                    dim=1)
                ChunksMSE[t] = chunk_mse

                if (epoch+1) == gan_epochs and t == iter:
                    capa_exceed = torch.sum(chunk_mse > args.mse_threshold) > 0

                model_structure = lightweight_ResNet18(ncla, nf=5).to(device)
                fake_parameter = fake_parameter.view(-1)[:-remainder]
                new_model = param_insert(model_structure, fake_parameter).to(device)
                # bn_update(new_model, trainloader, device)

                xtest = data[t]['test']['x']
                ytest = data[t]['test']['y']

                test_loss, test_acc = test(args, new_model, device, xtest, ytest, Crossentropy)

                # task_acc[iter][t] = 100. * correct / total
                print('Task {:d} Test Loss: {:.4f} Accuracy: {:.4f}'.format(t, test_loss, test_acc))
                print('Task: {:d}, Chunk MSE: \n'.format(t), chunk_mse)
                print()

    return generator, discriminator, capa_exceed


def main(args):
    tstart = time.time()
    ## Device Setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.no_cov_mat==True:
        cov_mat = False
    else:
        cov_mat = True

    prev_generator = None
    chunk_size = args.chunk_size
    capa_exceed = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ## Load dataset
    data, taskcla, inputsize = get(seed=args.seed, pc_valid=args.pc_valid)

    acc_matrix = np.zeros((5, 5))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    for k, ncla in taskcla:
        print('*' * 100)
        print('Task {:2d} ({:s})'.format(k, data[k]['name']))
        print('*' * 100)
        xtrain = data[k]['train']['x']
        ytrain = data[k]['train']['y']
        xvalid = data[k]['valid']['x']
        yvalid = data[k]['valid']['y']
        xtest = data[k]['test']['x']
        ytest = data[k]['test']['y']
        task_list.append(k)

        lr = args.lr
        print('-' * 40)
        print('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print('-' * 40)

        ######################################### Training compressed BNN #############################################
        ts_model = lightweight_ResNet18(ncla, nf=5)
        ts_model = ts_model.to(device)
        ts_model_size = sum(p.numel() for p in ts_model.parameters() if p.requires_grad)
        print("\nNumber of parameters of the student model: ", ts_model_size)

        swag_model = SWAG(base=lightweight_ResNet18(ncla, nf=5),
                          no_cov_mat=args.no_cov_mat,
                          max_num_models=args.max_num_models)
        swag_model = swag_model.to(device)

        ts_optimizer = optim.SGD(ts_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        ts_lr_scheduler = lr_scheduler.MultiStepLR(ts_optimizer, milestones=args.lr_step, gamma=args.lr_factor)

        for epoch in range(1, args.n_epochs + 1):
            # Train
            clock0 = time.time()
            swag_train(args, ts_model, swag_model, device, epoch, xtrain, ytrain, ts_optimizer, criterion)
            clock1 = time.time()
            tr_loss, tr_acc = swag_test(args, swag_model, device, xtrain, ytrain, criterion)
            print('Task {:d} | Student Training | Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'
                  .format(k, epoch, tr_loss, tr_acc, 1000 * (clock1 - clock0)), end='')
            # Validate
            student_valid_loss, student_valid_acc = test(args, ts_model, device, xvalid, yvalid, criterion)
            swag_valid_loss, swag_valid_acc = swag_test(args, swag_model, device, xvalid, yvalid, criterion)
            print(' Student Valid: loss={:.3f}, acc={:5.1f}% |'.format(student_valid_loss, student_valid_acc), end='')
            print(' SWAG Valid: loss={:.3f}, acc={:5.1f}% |'.format(swag_valid_loss, swag_valid_acc), end='')
            print()
            ts_lr_scheduler.step()
        # Test
        print('-' * 40)
        ts_test_loss, student_test_acc = test(args, ts_model, device, xtest, ytest, criterion)
        swag_test_loss, swag_test_acc = swag_test(args, swag_model, device, xtest, ytest, criterion)
        print('Task {:d}, Student Test: loss={:.3f} , acc={:5.1f}%'.format(k, ts_test_loss, student_test_acc))
        print('Task {:d}, SWAG Test: loss={:.3f} , acc={:5.1f}%'.format(k, swag_test_loss, swag_test_acc))

        torch.save(swag_model, './models_saved/five_dataset_swag_model_' + str(k) + '.pt')
        swag_model = torch.load('./models_saved/five_dataset_swag_model_' + str(k) + '.pt')

        # GAN training
        generator, discriminator, capa_exceed = gan_training(args=args,
                                                             iter=k,
                                                             ncla=ncla,
                                                             inputsize=inputsize,
                                                             chunk_size=chunk_size,
                                                             swag_model=swag_model,
                                                             model_size=ts_model_size,
                                                             data=data,
                                                             prev_g=prev_generator,
                                                             capa_exceed=capa_exceed,
                                                             device=device,
                                                             )

        prev_generator = deepcopy(generator)




if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential CIFAR-100 with GPM')

    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')

    parser.add_argument('--n_epochs', type=int, default=40, metavar='N',
                        help='number of training epochs/task (default: 120)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--pc_valid', default=0.05, type=float,
                        help='fraction of training data used for validation')


    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--lr_step', default=[20])

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')

    parser.add_argument('--lr_factor', type=int, default=0.1, metavar='LRF',
                        help='lr decay factor (default: 2)')

    # SWAG(BNN)
    parser.add_argument('--no_cov_mat', default=True)

    parser.add_argument('--max_num_models', type=int, default=20)

    parser.add_argument('--kd_temp', type=int, default=20)

    parser.add_argument('--kd_alpha', type=float, default=1.0)

    # GAN
    parser.add_argument('--chunk_size', default=5000, type=int, help='the size of chunk output of GAN')

    parser.add_argument('--gan_batch_size', default=512, type=int, help='the batch size of GAN')

    parser.add_argument('--mse_threshold', default=1.00, type=float, help='')

    parser.add_argument('--gan_epochs', default=300, type=int, help='the number of training epoch of GAN')

    parser.add_argument('--latent_dim', default=100, type=int, help='the size of latent vector of GAN')

    parser.add_argument('--ra_lambda', default=30.0)

    parser.add_argument('--gan_lr', default=0.0002, type=float, help='the learning rate of GAN')

    parser.add_argument('--gan_b1', default=0.5, type=float, help='adam: decay of first order momentum of gradient')

    parser.add_argument('--gan_b2', default=0.999, type=float, help='adam: decay of first order momentum of gradient')

    parser.add_argument('--num_critic', default=5, type=int, help='the number of training steps for discriminator per iter')

    parser.add_argument('--lambda_gp', default=10, type=float, help='')

    parser.add_argument('--task_agnostic', default=False)

    parser.add_argument('--num_models', default=30)

    parser.add_argument('--task_ag', default='ent', type=str, help='')

    parser.add_argument('--batch_size_test_t_ag', default=1, type=int)


    args = parser.parse_args()
    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':', getattr(args, arg))
    print('='*100)

    main(args)
