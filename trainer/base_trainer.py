import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import copy
from torch.utils.data import DataLoader

try:
    import cPickle as pickle
except:
    import pickle
import math
import models.lightweight_resnet32 as lightweight_resnet32
import models.lightweight_resnet18 as lightweight_resnet18
import models.lightweight_alexnet as lightweight_alexnet
import warnings
from swag.posteriors import *
from swag.swag_utils import *
from models.gan import *

warnings.filterwarnings('ignore')


class BaseTrainer(object):

    def __init__(self, the_args):

        self.args = the_args
        self.set_cuda_device()
        self.set_dataset_variables()

    def set_cuda_device(self):
        """The function to set CUDA device."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_dataset_variables(self):
        """The function to set the dataset parameters."""
        if self.args.dataset == 'cifar100':
            self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                       transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                       transforms.Normalize((0.5071, 0.4866, 0.4409),
                                                                            (0.2009, 0.1984, 0.2023)), ])
            self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.5071, 0.4866, 0.4409),
                                                                           (0.2009, 0.1984, 0.2023)), ])
            self.trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                          transform=self.transform_train)
            self.testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                                         transform=self.transform_test)
            self.evalset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False,
                                                         transform=self.transform_test)
            self.balancedset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False,
                                                             transform=self.transform_train)
            self.lr_strat = [100, 150, 200]
            # self.lr_strat = [int(self.args.ts_epochs * 0.5), int(self.args.ts_epochs * 0.75)]

        elif self.args.dataset == 'cifar10':
            self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5071, 0.4866, 0.4409),
                                                                            (0.229, 0.224, 0.225)), ])
            self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.5071, 0.4866, 0.4409),
                                                                           (0.229, 0.224, 0.225)), ])
            self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                          transform=self.transform_train)
            self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                         transform=self.transform_test)
            self.evalset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                                         transform=self.transform_test)
            self.balancedset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                                             transform=self.transform_train)
            self.lr_strat = [int(self.args.ts_epochs * 0.5), int(self.args.ts_epochs * 0.75)]

        elif self.args.dataset == 'cifar100_alexnet':
            self.transform_train = transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize((0.5071, 0.4866, 0.4409),
                                                                            (0.2009, 0.1984, 0.2023)), ])
            self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.5071, 0.4866, 0.4409),
                                                                           (0.2009, 0.1984, 0.2023)), ])
            self.trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                          transform=self.transform_train)
            self.testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                                         transform=self.transform_test)
            self.evalset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False,
                                                         transform=self.transform_test)
            self.balancedset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False,
                                                             transform=self.transform_train)
            self.lr_strat = [int(self.args.epochs * 0.5), int(self.args.epochs * 0.75)]

    def map_labels(self, order_list, Y_set):
        map_Y = []
        for idx in Y_set:
            map_Y.append(order_list.index(idx))
        map_Y = np.array(map_Y)
        return map_Y

    def set_dataset(self):
        """The function to set the datasets.
        Returns:
          X_train_total: an array that contains all training samples
          Y_train_total: an array that contains all training labels 
          X_valid_total: an array that contains all validation samples
          Y_valid_total: an array that contains all validation labels 
        """
        if self.args.dataset == 'cifar100' or self.args.dataset == 'cifar10':
            X_train_total = np.array(self.trainset.data)
            Y_train_total = np.array(self.trainset.targets)
            X_valid_total = np.array(self.testset.data)
            Y_valid_total = np.array(self.testset.targets)
        else:
            raise ValueError('Please set the correct dataset.')

        return X_train_total, Y_train_total, X_valid_total, Y_valid_total

    def init_class_order(self):
        # Set the random seed according to the config
        np.random.seed(self.args.random_seed)
        print("Generating a new class order")
        # For mathcing the class order in the experiments
        if self.args.dataset == 'cifar10':
            num_classes = 10
            task_order = [0, 1, 2, 3, 4]
        elif self.args.dataset == 'cifar100' or self.args.dataset == 'cifar100_alexnet':
            num_classes = 100
            task_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        num_task = int(num_classes / self.args.nb_cl)
        order = []
        for i in range(num_task):
            single_task = np.arange(self.args.nb_cl * i, self.args.nb_cl * (i + 1))
            order.append(single_task)
        order = np.array(order)
        order = np.concatenate(order[task_order])
        order_list = list(order)
        # Print the class order
        print(order_list)
        return order, order_list

    def init_current_phase_model(self, iteration, start_iter):
        last_iter = 0

        if iteration > start_iter:
            last_iter = iteration

        return last_iter

    def init_current_phase_dataset(self, iteration, start_iter, last_iter, order, order_list, \
                                   X_train_total, Y_train_total, X_valid_total, Y_valid_total, \
                                   X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls, map_Y_valid_cumuls):

        # Get the indexes of new-class samples (including training and test)
        indices_train_10 = np.array(
            [i in order[range(last_iter * self.args.nb_cl, (iteration + 1) * self.args.nb_cl)] for i in Y_train_total])
        indices_test_10 = np.array(
            [i in order[range(last_iter * self.args.nb_cl, (iteration + 1) * self.args.nb_cl)] for i in Y_valid_total])

        # Get the samples according to the indexes
        cur_X_train = X_train_total[indices_train_10]
        cur_X_valid = X_valid_total[indices_test_10]

        # Add the new-class samples to the cumulative X array
        X_valid_cumuls.append(cur_X_valid)
        X_train_cumuls.append(cur_X_train)

        # Get the labels according to the indexes, and add them to the cumulative Y array
        cur_Y_train = Y_train_total[indices_train_10]
        cur_Y_valid = Y_valid_total[indices_test_10]
        Y_valid_cumuls.append(cur_Y_valid)
        Y_train_cumuls.append(cur_Y_train)

        # Generate the mapped labels, according the order list
        map_Y_train = np.array([order_list.index(i) for i in cur_Y_train]) - iteration * self.args.nb_cl
        map_Y_valid = np.array([order_list.index(i) for i in cur_Y_valid]) - iteration * self.args.nb_cl

        # map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
        map_Y_valid_cumuls.append(map_Y_valid)

        # Return different variables for different phases
        return indices_train_10, X_train_cumuls, X_valid_cumuls, Y_train_cumuls, Y_valid_cumuls, cur_X_train, map_Y_train, map_Y_valid_cumuls


    def update_train_and_valid_loader(self, X_train, map_Y_train, X_valid, map_Y_valid):
        print('Setting the dataloaders ...')
        if self.args.dataset == 'cifar100' or self.args.dataset == 'cifar10' or self.args.dataset == 'cifar100_alexnet':
            # Set the training dataloader
            self.trainset.data = X_train.astype('uint8')
            self.trainset.targets = map_Y_train
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.train_batch_size,
                                                      shuffle=True, num_workers=self.args.num_workers)
            # Set the test dataloader
            self.testset.data = X_valid.astype('uint8')
            self.testset.targets = map_Y_valid
            testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.args.test_batch_size,
                                                     shuffle=False, num_workers=self.args.num_workers)
        else:
            raise ValueError('Please set the correct dataset.')
        return trainloader, testloader

    def knowledge_construction(self, epochs, trainloader, testloader, device=None):
        Crossentropy = nn.CrossEntropyLoss()
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.args.dataset == 'cifar10':
            ts_model = lightweight_resnet32.resnet20(num_classes=self.args.nb_cl)   # Task-Specific Lightweight Model
            swag_model = SWAG(base=lightweight_resnet32.resnet20(num_classes=self.args.nb_cl),
                              no_cov_mat=self.args.no_cov_mat,
                              max_num_models=self.args.max_num_models)
        elif self.args.dataset == 'cifar100':
            ts_model = lightweight_resnet18.resnet32(num_classes=self.args.nb_cl)
            swag_model = SWAG(base=lightweight_resnet18.resnet32(num_classes=self.args.nb_cl),
                              no_cov_mat=self.args.no_cov_mat,
                              max_num_models=self.args.max_num_models)
        elif self.args.dataset == 'cifar100_alexnet':
            ts_model = lightweight_alexnet.Lightweight_AlexNet(num_classes=self.args.nb_cl)
            swag_model = SWAG(base=lightweight_alexnet.Lightweight_AlexNet(num_classes=self.args.nb_cl),
                              no_cov_mat=self.args.no_cov_mat,
                              max_num_models=self.args.max_num_models)

        ts_model = ts_model.to(device)
        swag_model = swag_model.to(device)

        model_size = sum(p.numel() for p in ts_model.parameters() if p.requires_grad)
        print("\nNumber of parameters of the student model: ", model_size)

        if self.args.dataset == 'cifar10':
            ts_model_optimizer = optim.SGD(ts_model.parameters(), lr=self.args.ts_lr,
                                           momentum=self.args.custom_momentum,
                                           weight_decay=self.args.custom_weight_decay)

            model_lr_scheduler = lr_scheduler.MultiStepLR(ts_model_optimizer, milestones=[80], gamma=self.args.lr_factor)

        elif self.args.dataset == 'cifar100':
            ts_model_optimizer = optim.SGD(ts_model.parameters(), lr=self.args.ts_lr,
                                           momentum=self.args.custom_momentum,
                                           weight_decay=self.args.custom_weight_decay)

            model_lr_scheduler = lr_scheduler.MultiStepLR(ts_model_optimizer,
                                                          milestones=[int(self.args.student_epochs * 0.4),
                                                                      int(self.args.student_epochs * 0.6),
                                                                      int(self.args.student_epochs * 0.8)],
                                                          gamma=self.args.lr_factor)
        elif self.args.dataset == 'cifar100_alexnet':
            ts_model_optimizer = optim.SGD(ts_model.parameters(), lr=self.args.ts_lr,
                                           momentum=self.args.custom_momentum,
                                           weight_decay=self.args.custom_weight_decay)

            model_lr_scheduler = lr_scheduler.MultiStepLR(ts_model_optimizer,
                                                          milestones=[int(self.args.epochs * 0.5),
                                                                      int(self.args.epochs * 0.75)],
                                                          gamma=self.args.lr_factor)

        for epoch in range(epochs):
            ts_model.train()
            train_loss = 0
            correct = 0
            total = 0

            print('\nEpoch: {:d}, learning rate: {:.4f}'.format(epoch, model_lr_scheduler.get_lr()[0]))

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, dtype=torch.long)

                ts_model_optimizer.zero_grad()
                outputs = ts_model(inputs)

                loss = Crossentropy(outputs, targets)

                loss.backward()
                ts_model_optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            # SWAG training
            if epoch >= int(epochs // 2):
                swag_model.collect_model(ts_model)

            print('Train set: {}, train loss: {:.4f} accuracy: {:.4f}'
                  .format(len(trainloader), train_loss / (batch_idx + 1), 100. * correct / total))

            ts_model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device, dtype=torch.long)
                    outputs = ts_model(inputs)
                    Loss = Crossentropy(outputs, targets)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    test_loss += Loss.item()
            print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(
                len(testloader), test_loss / (batch_idx + 1), 100. * correct / total))

            # SWAG test
            if epoch >= int(epochs // 2) and (epoch+1) % 20 == 0:
                swag_model.sample(scale=1.0, cov=self.args.cov_mat, device=device)
                bn_update(swag_model, trainloader, device)
                swag_model.eval()
                correct = 0
                total = 0
                swag_test_loss = 0
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(testloader):
                        inputs, targets = inputs.to(device), targets.to(device, dtype=torch.long)
                        outputs = swag_model(inputs)
                        swag_Loss = Crossentropy(outputs, targets)
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                        swag_test_loss += swag_Loss.item()
                print('SWAG Test set: {} SWAG test loss: {:.4f} SWAG accuracy: {:.4f}'.format(
                    len(testloader), swag_test_loss / (batch_idx + 1), 100. * correct / total))

            model_lr_scheduler.step()

        return ts_model, model_size, swag_model

    def gan_training(self, iter, chunk_size, swag_model, model_size, dataloaders, prev_g=None,
                     capa_exceed=False, device=None):

        MSELoss = nn.MSELoss()
        Crossentropy = nn.CrossEntropyLoss()
        gan_batch_size = self.args.gan_batch_size
        gan_epochs = self.args.gan_epochs
        latent_dim = self.args.latent_dim
        ra_lambda = self.args.ra_lambda

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if prev_g is not None:
            prev_g.eval()

        # Number of chunks
        if capa_exceed == True:
            prev_chunk_size = chunk_size / 2
            prev_num_chunks = math.ceil(model_size / prev_chunk_size)
        original_num_chunks = math.ceil(model_size / chunk_size)
        num_chunks = original_num_chunks * (iter + 1)

        _, _, mu, sigma = swag_model.chunk_sample(1.0, self.args.cov_mat, chunk_size, original_num_chunks)

        # Always Initialize G and D for every task
        generator = Generator(self.args.latent_dim, iter, chunk_size, num_chunks).to(device)
        discriminator = Discriminator(iter, chunk_size, num_chunks).to(device)

        g_optimizer = torch.optim.Adam(generator.parameters(), lr=self.args.gan_lr,
                                       betas=(self.args.gan_b1, self.args.gan_b2))
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=self.args.gan_lr,
                                       betas=(self.args.gan_b1, self.args.gan_b2))

        # GAN model size
        generator_model_size = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        discriminator_model_size = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        print('G model size: {:d}, D model size: {:d}'.format(generator_model_size, discriminator_model_size))

        # Training phase
        ChunksMSE = torch.zeros(iter + 1, original_num_chunks)  # MSE between generated parameters and real parameters
        for epoch in range(gan_epochs):
            print('Task: {:d}, Epoch: {:d}, learning rate: {:.4f}, Chunk size: {:d}'.format(
                iter, epoch, self.args.gan_lr, chunk_size))
            # Training iteration in one epoch
            for i in range(50000 * (iter + 1) // gan_batch_size):
                sampled_model, remainder, means, stds = swag_model.chunk_sample(1.0, self.args.cov_mat, chunk_size,
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
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.args.lambda_gp * gradient_penalty
                d_loss.backward()
                d_optimizer.step()

                # Training Generator
                g_optimizer.zero_grad()
                if i % self.args.num_critic == 0:
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
                                z_cmt = copy.deepcopy(z_)
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

                    g_loss = -torch.mean(fake_validity) + ra_lambda * replay_alignment
                    g_loss.backward()
                    g_optimizer.step()

            # Evaluation phase
            if (epoch+1) % 100 == 0:
                # BNN sampling only for checking chunk MSE
                sampled_model_list = []
                for iteration_ in range(iter + 1):
                    swag_model = torch.load('./models_saved/cifar10_swag_model_smaller_'+str(self.args.random_seed)+'_'+str(iteration_)+'.pt')
                    sampled_model, remainder, means, stds = swag_model.chunk_sample(1.0, self.args.cov_mat, chunk_size,
                                                                                    original_num_chunks)
                    sampled_model = stds * sampled_model + means    # de-normalization
                    sampled_model_list.append(sampled_model)
                sampled_model_list = torch.cat(sampled_model_list).to('cpu')

                generator.eval()
                for t in range(iter + 1):
                    trainloader = dataloaders[t]['trainloader']
                    testloader = dataloaders[t]['testloader']

                    chunkid = torch.arange(original_num_chunks*t, original_num_chunks*(t+1)).to(device)
                    z = torch.rand(original_num_chunks, self.args.latent_dim).to(device)
                    fake_parameter = generator(z, chunkid).to('cpu')
                    fake_parameter = sigma * fake_parameter + mu    # de-normalization

                    # Calculating MSE for each chunks
                    chunk_mse = torch.mean(torch.square(
                        fake_parameter - sampled_model_list[original_num_chunks * t:original_num_chunks * (t + 1)]),
                        dim=1)
                    ChunksMSE[t] = chunk_mse

                    if (epoch+1) == gan_epochs and t == iter:
                        capa_exceed = torch.sum(chunk_mse > self.args.mse_threshold) > 0

                    if self.args.dataset == 'cifar10':
                        model_structure = lightweight_resnet32.resnet20(num_classes=self.args.nb_cl)
                    elif self.args.dataset == 'cifar100':
                        model_structure = lightweight_resnet18.resnet32(num_classes=self.args.nb_cl)
                    elif self.args.dataset == 'cifar100_alexnet':
                        model_structure = lightweight_alexnet.Lightweight_AlexNet(num_classes=self.args.nb_cl)
                    fake_parameter = fake_parameter.view(-1)[:-remainder]
                    new_model = param_insert(model_structure, fake_parameter).to(device)
                    bn_update(new_model, trainloader, device)

                    new_model.eval()
                    correct = 0
                    total = 0
                    test_loss = 0
                    with torch.no_grad():
                        for batch_idx, (inputs, targets) in enumerate(testloader):
                            inputs, targets = inputs.to(device), targets.to(device, dtype=torch.long)
                            outputs = new_model(inputs)
                            loss = Crossentropy(outputs, targets)
                            _, predicted = outputs.max(1)
                            total += targets.size(0)
                            correct += predicted.eq(targets).sum().item()
                            test_loss += loss.item()

                    print('Task {:d} Generated Model Test set: {} Test Loss: {:.4f} Accuracy: {:.4f}'.format(
                        t, len(testloader), test_loss / (batch_idx + 1), 100. * correct / total))
                    print('Task: {:d}, Chunk MSE: \n'.format(t), chunk_mse)
                    print()
                if (epoch+1) == gan_epochs and capa_exceed == True:
                    print("\nTask: {:d}, GAN Capacity: Exceeded\n".format(iter))

        return generator, discriminator, capa_exceed


