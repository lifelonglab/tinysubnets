import numpy as np
import torch
import importlib
import torch.nn as nn
import networks.tiny.learner as Learner
import networks.tiny.modelfactory as mf

class BaseNet(torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(BaseNet, self).__init__()

        self.args = args
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks

        # setup network
        if args.dataset == 'tinyimagenet':
            config = mf.ModelFactory.get_model(sizes=[self.n_outputs], dataset=self.args.dataset, args=args)
            self.net = Learner.Learner(config, args)
            self.net.n_rep = 6
            self.net.multi_head = True
        else:
            if args.dataset in ['mnist_permutations', 'pmnist']:
                network = importlib.import_module('networks.mlp')
            elif args.dataset == 'cifar100':
                network = importlib.import_module('networks.alexnet')
            elif args.dataset == 'cifar100_superclass':
                network = importlib.import_module('networks.lenet')

            self.net = network.Learner(self.n_inputs, self.n_outputs, self.n_tasks, self.args)

        #if self.cuda:
        self.net = self.net.to()

        if self.net.multi_head:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs

        # setup losses
        self.loss_ce = torch.nn.CrossEntropyLoss()

        # setup optimizer
        self.lr = args.lr
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=args.momentum)

        # define training params
        self.current_task = -1
        self.epoch = 0
        self.real_epoch = 0
        self.iter = 0
        self.glances = args.glances
        self.n_epochs = args.n_epochs
        self.batchSize = args.batch_size_train
        self.test_batch_size = args.batch_size_test

        # setup replay buffer (episodic memory)
        self.memories = args.memories  # mem_size of M
        self.age = 0  # total number of training samples
        self.M = []

        # setup GPM
        self.mem_batch_size = args.mem_batch_size
        self.M_vec = []  # bases of GPM
        self.M_val = []  # eigenvalues of each basis of GPM
        self.M_task = []  # the task id of each basis, only used to analysis GPM

    def compute_offsets(self, t):
        if self.net.multi_head:
            # mapping from classes to their idx within a task
            offset1 = t * self.nc_per_task
            offset2 = min((t + 1) * self.nc_per_task, self.n_outputs)
        else:
            offset1 = 0
            offset2 = self.nc_per_task

        return offset1, offset2

    def take_loss(self, x, y, t, fast_weights=None):
        """
            Get loss of a task t
            """
        outputs = self.net.forward(x, fast_weights)
        if self.net.multi_head:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            loss = self.loss_ce(outputs[:, offset1:offset2], y - offset1)
        else:
            loss = self.loss_ce(outputs, y)

        return loss

    def meta_loss(self, x, y, tasks, fast_weights=None):
        """
            Get loss of multiple tasks tasks
            """
        outputs = self.net.forward(x, fast_weights)
        loss = 0.0
        for task in np.unique(tasks.data.cpu().numpy()):
            task = int(task)
            idx = torch.nonzero(tasks == task).view(-1)

            if self.net.multi_head:
                offset1, offset2 = self.compute_offsets(task)
                loss += self.loss_ce(outputs[idx, offset1:offset2], y[idx] - offset1) * len(idx)
            else:
                loss += self.loss_ce(outputs[idx], y[idx]) * len(idx)

        return loss/len(y)

    def update_optimizer(self, lr=None):
        if lr is None:
            lr = self.lr
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=self.args.momentum)

    def push_to_mem(self, batch_x, batch_y, t):
        """
            Reservoir sampling to push subsampled stream of data points to replay buffer
            """
        if self.real_epoch > 0:
            return

        batch_x = batch_x.cpu()
        batch_y = batch_y.cpu()
        t = t.cpu()

        for i in range(batch_x.shape[0]):
            self.age += 1
            if len(self.M) < self.memories:
                self.M.append([batch_x[i], batch_y[i], t])
            else:
                p = np.random.randint(0, self.age)
                if p < self.memories:
                    self.M[p] = [batch_x[i], batch_y[i], t]

    def get_batch(self, x, y, t):
        """
            Given the new data points, create a batch of old + new data,
            where old data is sampled from the replay buffer
            """
        t = (torch.ones(x.shape[0]).int() * t)

        if len(self.M) > 0:
            MEM = self.M
            order = np.arange(len(MEM))
            np.random.shuffle(order)
            index = order[:min(x.shape[0], len(MEM))]

            x = x.cpu()
            y = y.cpu()

            for k, idx in enumerate(index):
                ox, oy, ot = MEM[idx]
                x = torch.cat((x, ox.unsqueeze(0)), 0)
                y = torch.cat((y, oy.unsqueeze(0)), 0)
                t = torch.cat((t, ot.unsqueeze(0)), 0)

        # handle gpus if specified
        if self.cuda:
            x = x.cuda()
            y = y.cuda()
            t = t.cuda()

        return x, y, t

    