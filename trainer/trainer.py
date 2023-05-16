import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from trainer.base_trainer import BaseTrainer
from swag.swag_utils import *
from swag.posteriors import *
import models.lightweight_resnet32 as lightweight_resnet32
import models.lightweight_resnet18 as lightweight_resnet18
import models.lightweight_alexnet as lightweight_alexnet
import warnings
warnings.filterwarnings('ignore')

import models.lightweight_resnet32 as student_resnet_cifar

class Trainer(BaseTrainer):
    def train(self):
        # Load the training and test samples from the dataset
        X_train_total, Y_train_total, X_valid_total, Y_valid_total = self.set_dataset()

        # Initialize the class order
        order, order_list = self.init_class_order()
        np.random.seed(None)

        # Set empty lists for the data    
        X_valid_cumuls    = []
        X_train_cumuls    = []
        Y_valid_cumuls    = []
        Y_train_cumuls    = []
        map_Y_valid_cumuls= []
        dataloaders = {}

        # Set the starting iteration
        # We start training the class-incremental learning system from e.g., 50 classes to provide a good initial encoder
        if self.args.dataset == 'cifar10':
            num_classes = 10
        elif self.args.dataset == 'cifar100' or self.args.dataset == 'cifar100_alexnet':
            num_classes = 100
        start_iter = int(self.args.nb_cl_fg/self.args.nb_cl)-1
        num_iter = int(num_classes / self.args.nb_cl)

        task_acc = torch.zeros(num_iter, num_iter)

        b1_model = None
        prev_generator = None
        chunk_size = self.args.chunk_size

        for iteration in range(start_iter, num_iter):
            ### Initialize models for the current phase
            last_iter = self.init_current_phase_model(iteration, start_iter)

            ### Initialize datasets for the current phase
            indices_train_10, X_train_cumuls, X_valid_cumuls, \
            Y_train_cumuls, Y_valid_cumuls, cur_X_train, map_Y_train, map_Y_valid_cumuls = \
                self.init_current_phase_dataset(iteration=iteration, start_iter=start_iter, last_iter=last_iter,
                                                order=order, order_list=order_list,
                                                X_train_total=X_train_total, Y_train_total=Y_train_total,
                                                X_valid_total=X_valid_total, Y_valid_total=Y_valid_total,
                                                X_train_cumuls=X_train_cumuls, Y_train_cumuls=Y_train_cumuls,
                                                X_valid_cumuls=X_valid_cumuls, Y_valid_cumuls=Y_valid_cumuls,
                                                map_Y_valid_cumuls=map_Y_valid_cumuls)

            # Update training and test dataloader
            trainloader, testloader = self.update_train_and_valid_loader(X_train=cur_X_train,
                                                                         map_Y_train=map_Y_train,
                                                                         X_valid=X_valid_cumuls[iteration],
                                                                         map_Y_valid=map_Y_valid_cumuls[iteration])

            dataloaders[iteration] = dict.fromkeys(['trainloader', 'testloader'])
            dataloaders[iteration]['trainloader'] = copy.deepcopy(trainloader)
            dataloaders[iteration]['testloader'] = copy.deepcopy(testloader)

            ts_model, ts_model_size, swag_model = self.knowledge_construction(self.args.ts_epochs,
                                                                              trainloader, testloader, self.device)
            torch.save(swag_model, './models_saved/cifar10_swag_model_smaller_'+str(self.args.random_seed)+'_'+str(iteration)+'.pt')

            generator, discriminator, capa_exceed = self.gan_training(
                iter=iteration,
                chunk_size=chunk_size,
                swag_model=swag_model,
                model_size=ts_model_size,
                dataloaders=dataloaders,
                prev_g=prev_generator,
                capa_exceed=capa_exceed,
                device=self.device,
            )

            prev_generator = copy.deepcopy(generator)

        #------------------------------------- evaluation ----------------------------------------------------------
        swag_model = torch.load(
            './models_saved/cifar10_swag_model_smaller_' + str(self.args.random_seed) + '_' + str(iteration) + '.pt')
        student_model = student_resnet_cifar.resnet20(num_classes=self.args.nb_cl)
        student_model_size = sum(p.numel() for p in student_model.parameters() if p.requires_grad)

        chunk_size = self.args.chunk_size
        Crossentropy = nn.CrossEntropyLoss()
        original_num_chunks = math.ceil(student_model_size / chunk_size)
        device = self.device
        _, remainder, mu, sigma = swag_model.chunk_sample(1.0, self.args.cov_mat, chunk_size, original_num_chunks)
        generator.eval()
        model_set = {}
        for i in range(iteration+1):
            model_set[i] = []
        for model_id in range(iteration +1):
            print(model_id)
            for _ in range(self.args.num_models):
                chunkid = torch.arange(original_num_chunks * model_id, original_num_chunks * (model_id + 1)).to(device)
                z = torch.rand(original_num_chunks, self.args.latent_dim).to(device)
                fake_parameter = generator(z, chunkid).to('cpu')
                fake_parameter = sigma * fake_parameter + mu  # de-normalization

                if self.args.dataset == 'cifar10':
                    model_structure = lightweight_resnet32.resnet20(num_classes=self.args.nb_cl)
                elif self.args.dataset == 'cifar100':
                    model_structure = lightweight_resnet18.resnet32(num_classes=self.args.nb_cl)
                elif self.args.dataset == 'cifar100_alexnet':
                    model_structure = lightweight_alexnet.Lightweight_AlexNet(num_classes=self.args.nb_cl)

                fake_parameter = fake_parameter.view(-1)[:-remainder]
                new_model = param_insert(model_structure, fake_parameter).to(device)
                # BN update
                bn_update(new_model, dataloaders[model_id]['trainloader'], device)
                model_set[model_id].append(new_model)
        torch.save(model_set, './models_saved/model_set_cifar10_generator_wo_expansion_' + str(iteration) + '_' + str(
            self.args.random_seed) + '_' + str(self.args.num_models) + '.pt')
        model_set = torch.load('./models_saved/model_set_cifar10_generator_wo_expansion_' + str(iteration) + '_' + str(
            self.args.random_seed) + '_' + str(self.args.num_models) + '.pt')

        for t in range(iteration + 1):
            testloader = dataloaders[t]['testloader']

            if self.args.task_ag == 'ent':
                correct = 0
                total = 0
                test_loss = 0
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(testloader):
                        inputs, targets = inputs.to(device), targets.to(device, dtype=torch.long)
                        ent_set = torch.zeros(iteration+1)
                        for i in range(len(model_set)):
                            prob_sum = 0
                            for model_sample in model_set[i]:
                                model_sample.eval()
                                outputs = model_sample(inputs)
                                p = F.softmax(outputs, dim=1)
                                prob_sum += p
                            prob_mean = prob_sum / self.args.num_models
                            ent = -torch.sum(prob_mean * torch.log(prob_mean))
                            ent_set[i] = ent

                        _, t_id = ent_set.view(-1, iteration+1).min(1)

                        new_model = model_set[int(t_id)][int(torch.randint(1, self.args.num_models, (1,)))]
                        new_model.eval()
                        outputs = new_model(inputs)
                        loss = Crossentropy(outputs, targets)
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                        test_loss += loss.item()
                print('Task {:d} Generated Model Test set: {} Test Loss: {:.4f} Accuracy: {:.4f}'.format(
                    t, len(testloader), test_loss / (batch_idx + 1), 100. * correct / total))


            task_acc[iteration][t] = 100. * correct / total

        print(task_acc)
        print(torch.mean(task_acc[iteration]))





