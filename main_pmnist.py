import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import argparse, time
from copy import deepcopy
import os, sys
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from models.gan import *
from swag.posteriors import *
from swag.swag_utils import *
import math
mnist_dir = './data/'
pmnist_dir = './data/binary_pmnist'

class Net(torch.nn.Module):

    def __init__(self,inputsize, n_hidden=100, num_classes=10):
        super(Net, self).__init__()

        ncha, size, _ = inputsize

        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(ncha * size * size, n_hidden, bias=False)
        self.linear2 = torch.nn.Linear(n_hidden, n_hidden, bias=False)
        self.fc = torch.nn.Linear(n_hidden, num_classes, bias=False)

        return

    def forward(self, x):
        h = x.view(x.size(0), -1)
        h = self.relu(self.linear1(h))
        h = self.relu(self.linear2(h))
        y = self.fc(h)

        return y


def get(seed=0, fixed_order=False, pc_valid=0.1):
    data = {}
    taskcla = []
    size = [1, 28, 28]

    nperm = 10  # 10 tasks
    seeds = np.array(list(range(nperm)), dtype=int)
    if not fixed_order:
        seeds = shuffle(seeds, random_state=seed)

    if not os.path.isdir(pmnist_dir):
        os.makedirs(pmnist_dir)
        # Pre-load
        # MNIST
        mean = (0.1307,)
        std = (0.3081,)
        dat = {}
        dat['train'] = datasets.MNIST(mnist_dir, train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.MNIST(mnist_dir, train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for i, r in enumerate(seeds):
            print(i, end=',')
            sys.stdout.flush()
            data[i] = {}
            data[i]['name'] = 'pmnist-{:d}'.format(i)
            data[i]['ncla'] = 10
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[i][s] = {'x': [], 'y': []}
                for image, target in loader:
                    aux = image.view(-1).numpy()
                    aux = shuffle(aux, random_state=r * 100 + i)
                    image = torch.FloatTensor(aux).view(size)
                    data[i][s]['x'].append(image)
                    data[i][s]['y'].append(target.numpy()[0])

            # "Unify" and save
            for s in ['train', 'test']:
                data[i][s]['x'] = torch.stack(data[i][s]['x']).view(-1, size[0], size[1], size[2])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                torch.save(data[i][s]['x'], os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'x.bin'))
                torch.save(data[i][s]['y'], os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'y.bin'))
        print()

    else:

        # Load binary files
        for i, r in enumerate(seeds):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[i]['ncla'] = 10
            data[i]['name'] = 'pmnist-{:d}'.format(i)

            # Load
            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'y.bin'))

    # Validation
    for t in data.keys():
        r=np.arange(data[t]['train']['x'].size(0))
        # r=np.array(shuffle(r,random_state=seed),dtype=int)
        r=np.array(r,dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y'] = data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x'] = data[t]['train']['x'][itrain].clone()
        data[t]['train']['y'] = data[t]['train']['y'][itrain].clone()

    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size


def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


def train (args, model, swag_model, device, epoch, x, y, optimizer,criterion, task_id):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b].view(-1,28*28)
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # SWAG training
    if epoch >= 2:
        swag_model.collect_model(model)


def test(args, model, device, x, y, criterion, task_id):
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
            data = x[b].view(-1, 28 * 28)
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
            data = x[b].view(-1, 28 * 28)
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
            data = x[b].view(-1, 28 * 28)
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
    ## Load PMNIST DATASET
    data, taskcla, inputsize = get(seed=args.seed, pc_valid=args.pc_valid)

    acc_matrix = np.zeros((args.n_tasks, args.n_tasks))
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

        model = Net(inputsize, args.n_hidden, ncla).to(device)
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Model size: {:d}".format(model_size))
        print('Model parameters ---')

        swag_model = SWAG(base=Net(inputsize, args.n_hidden, ncla),
                          no_cov_mat=args.no_cov_mat,
                          max_num_models=args.max_num_models)
        swag_model = swag_model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=lr)

        for epoch in range(1, args.n_epochs + 1):
            # Train
            clock0 = time.time()
            train(args, model, swag_model, device, epoch, xtrain, ytrain, optimizer, criterion, k)
            clock1 = time.time()
            tr_loss, tr_acc = test(args, model, device, xtrain, ytrain, criterion, k)
            print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'
                  .format(epoch, tr_loss, tr_acc, 1000 * (clock1 - clock0)), end='')
            # Validate
            valid_loss, valid_acc = test(args, model, device, xvalid, yvalid, criterion, k)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc), end='')
            print()
        # Test
        print('-' * 40)
        test_loss, test_acc = test(args, model, device, xtest, ytest, criterion, k)
        swag_test_loss, swag_test_acc = swag_test(args, swag_model, device, xtest, ytest, criterion)
        print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss, test_acc))
        print('SWAG Test: loss={:.3f} , acc={:5.1f}%'.format(swag_test_loss, swag_test_acc))

        torch.save(swag_model,'./models_saved/pmnist_swag_model_' + str(k) + '.pt')
        swag_model = torch.load('./models_saved/pmnist_swag_model_' + str(k) + '.pt')

        # GAN training
        generator, discriminator, capa_exceed = gan_training(args=args,
                                                             iter=k,
                                                             ncla=ncla,
                                                             inputsize=inputsize,
                                                             chunk_size=chunk_size,
                                                             swag_model=swag_model,
                                                             model_size=model_size,
                                                             data=data,
                                                             prev_g=prev_generator,
                                                             capa_exceed=capa_exceed,
                                                             device=device,
                                                             )

        prev_generator = deepcopy(generator)


        # save accuracy
        jj = 0
        generator.eval()
        original_num_chunks = math.ceil(model_size / chunk_size)
        num_chunks = original_num_chunks * (k + 1)  # base_num_chunks = original_num_chunks - num_saved_chunks
        _, remainder, mu, sigma = swag_model.chunk_sample(1.0, cov_mat, chunk_size, original_num_chunks)

        for ii in np.array(task_list)[0:task_id + 1]:
            curr_task = task_id
            xtest = data[ii]['test']['x']
            ytest = data[ii]['test']['y']

            if args.task_agnostic == True:
                model_set = get_model_set(args, generator, swag_model, device, model_size, chunk_size, cov_mat,
                                          inputsize, curr_task, ncla)
                _, acc_matrix[task_id, jj] = test_task_agnostic(args, model_set, device, xtest, ytest, criterion,
                                                                curr_task)

            else:
                chunkid = torch.arange(original_num_chunks * ii, original_num_chunks * (ii + 1)).to(device)
                z = torch.rand(original_num_chunks, args.latent_dim).to(device)
                fake_parameter = generator(z, chunkid).to('cpu')
                fake_parameter = sigma * fake_parameter + mu  # de-normalization

                model_structure = Net(inputsize, args.n_hidden, ncla).to(device)
                fake_parameter = fake_parameter.view(-1)[:-remainder]
                new_model = param_insert(model_structure, fake_parameter).to(device)
                _, acc_matrix[task_id, jj] = test(args, new_model, device, xtest, ytest, criterion, ii)
            jj += 1

        print('Accuracies =')
        for i_a in range(task_id + 1):
            print('\t', end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a, j_a]), end='')
            print()
        # update task id
        task_id += 1
    print('-' * 50)
    # Simulation Results
    print('Task Order : {}'.format(np.array(task_list)))
    print('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean()))
    bwt = np.mean((acc_matrix[-1] - np.diag(acc_matrix))[:-1])
    print('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time() - tstart) * 1000))
    print('-' * 50)
    # Plots
    array = acc_matrix
    df_cm = pd.DataFrame(array, index=[i for i in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]],
                         columns=[i for i in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]])
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    plt.show()


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
                swag_model = torch.load('./models_saved/pmnist_swag_model_'+str(iteration_)+'.pt')
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

                model_structure = Net(inputsize, args.n_hidden, ncla).to(device)
                fake_parameter = fake_parameter.view(-1)[:-remainder]
                new_model = param_insert(model_structure, fake_parameter).to(device)
                # bn_update(new_model, trainloader, device)

                xtest = data[t]['test']['x']
                ytest = data[t]['test']['y']

                test_loss, test_acc = test(args, new_model, device, xtest, ytest, Crossentropy, t)

                # task_acc[iter][t] = 100. * correct / total
                print('Task {:d} Test Loss: {:.4f} Accuracy: {:.4f}'.format(t, test_loss, test_acc))
                print('Task: {:d}, Chunk MSE: \n'.format(t), chunk_mse)
                print()
            if (epoch+1) == gan_epochs and capa_exceed == True:
                print("\nTask: {:d}, GAN Capacity: Exceeded\n".format(iter))

    return generator, discriminator, capa_exceed


if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=10, metavar='N', # 10
                        help='input batch size for training (default: 10)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=5, metavar='N',
                        help='number of training epochs/task (default: 5)')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--pc_valid',default=0.1,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    # Architecture
    parser.add_argument('--n_hidden', type=int, default=100, metavar='NH',
                        help='number of hidden units in MLP (default: 100)')
    parser.add_argument('--n_outputs', type=int, default=10, metavar='NO',
                        help='number of output units in MLP (default: 10)')
    parser.add_argument('--n_tasks', type=int, default=10, metavar='NT',
                        help='number of tasks (default: 10)')
    # SWAG(BNN)
    parser.add_argument('--no_cov_mat', default=True)
    parser.add_argument('--max_num_models', type=int, default=20)

    # GAN
    parser.add_argument('--chunk_size', default=10000, type=int,
                        help='the size of chunk output of GAN')
    parser.add_argument('--gan_batch_size', default=256, type=int,
                        help='the batch size of GAN')
    parser.add_argument('--mse_threshold', default=0.001, type=float,
                        help='')
    parser.add_argument('--gan_epochs', default=200, type=int,
                        help='the number of training epoch of GAN')
    parser.add_argument('--latent_dim', default=100, type=int,
                        help='the size of latent vector of GAN')
    parser.add_argument('--ra_lambda', default=50.0)
    parser.add_argument('--gan_lr', default=0.001, type=float,
                        help='the learning rate of GAN')
    parser.add_argument('--gan_b1', default=0.5, type=float,
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--gan_b2', default=0.999, type=float,
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--num_critic', default=5, type=int,
                        help='the number of training steps for discriminator per iter')
    parser.add_argument('--lambda_gp', default=10, type=float, help='')
    parser.add_argument('--task_agnostic', default=True)
    parser.add_argument('--num_models', default=30)
    parser.add_argument('--task_ag', default='ent', type=str, help='')
    parser.add_argument('--batch_size_test_t_ag', default=1, type=int)




    args = parser.parse_args()
    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)

    main(args)


