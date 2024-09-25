
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random

import argparse,time
import math
from copy import deepcopy

from utils import safe_save, save_pickle
from copy import deepcopy

from networks.subnet import SubnetLinear, SubnetConv2d
from networks.mlp import SubnetMLPNet as MLPNet
from networks.utils import *
import pdb

from sklearn import cluster
import scipy.cluster.vq as vq
from kmeans_scaler_hist import kmeans_scaler_hist
import numpy as np
from numpy import linalg as LA


def vq_and_back(filt, n_clusters, sparsity_threshold=0):
    X = filt.reshape((-1, 1))  # We need an (n_sample, n_feature) array
    X = X.cpu().numpy()
    sparsity_enabled=(sparsity_threshold!=0)
    clusters_used = n_clusters
    k_means = cluster.KMeans(n_clusters=clusters_used, n_init=1, verbose=0, n_jobs=-1)
    k_means.fit(X)
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    if sparsity_enabled:
        min_idx = np.argmin(values)
        values[min_idx] = 0
    
    print("Cluster Values = {}".format(values))
    out = np.take(values, labels)
    out.shape = filt.shape
    return out, values, labels


def vq_and_back_fast(filt, n_clusters, sparsity_threshold=0):
    X = filt.reshape((-1, 1))  # We need an (n_sample, n_feature) array
    sparsity_enabled=(sparsity_threshold!=0)
    #print("X.Shape")
    #print(X.shape)
    clusters_used = n_clusters
    k_means = cluster.KMeans(n_clusters=clusters_used, n_init=1, verbose=0, n_jobs=-1)
    sz = X.shape
    print(sz)
    if False:#sz[0] > 1000000:
        idx = np.random.choice(sz[0],100000)
        x_short = X[idx,:]
    else:
        x_short = X
    k_means.fit(x_short)
    values = k_means.cluster_centers_#.squeeze()
    labels = k_means.labels_
    if sparsity_enabled:
        min_idx = np.argmin(values)
        values[min_idx] = 0
    #    for ix in range(len(values)):
    #        if values[ix] < sparsity_threshold:
    #            values[ix] = 0

    # create an array from labels and values
    #out = np.choose(labels, values)
    print("Cluster Values = {}".format(values))
    print("shape x")
    print(X.shape)
    print("shape values")
    print(values.shape)
    labels, dist = vq.vq(X, values)
    print("shape labels")
    print(labels)
    out = np.take(values, labels)
    out.shape = filt.shape
    return out


def vq_and_back_fastest(filt, n_clusters, sparsity_threshold=0):
    X = filt.reshape((-1, 1))  # We need an (n_sample, n_feature) array
    sparsity_enabled=(sparsity_threshold!=0)
    clusters_used = n_clusters
    sz = X.shape
    print(sz)
    idx = np.random.choice(sz[0],100000)
    x_short = X[idx,:]
    values, edges = kmeans_scaler_hist(x_short, clusters_used)
    if sparsity_enabled:
        min_idx = np.argmin(values)
        values[min_idx] = 0

    print("Cluster Values = {}".format(values))
    print("shape x")
    print(X.shape)
    print("shape values")
    print(values.shape)
    labels, dist = vq.vq(X.flatten(), values)
    print("shape labels")
    print(labels)
    ids, counts = np.unique(labels, return_counts=True)
    print("Counts")
    print(counts)
    out = np.take(values, labels)
    out.shape = filt.shape
    return out


def vquant(in_tensor, n_clusters=16, sparsity_threshold=0, fast=False):
    in_np = in_tensor
    np.random.seed(0)
    shape = in_np.shape
    out_combined = np.zeros(in_np.shape)
    if False: #in_np.ndim == 4:
        for itr in range(shape[0]):
            print(str(itr) + ': shape' + str(in_np.shape))
            filt = in_np[itr,:,:,:]
            out = vq_and_back(filt, n_clusters)
            out.shape = filt.shape
            out_combined[itr,:,:,:] = out
    else: #in_np.ndim == 2:
        print('shape' + str(in_np.shape))
        filt = in_np
        if fast == True:
            out = vq_and_back_fastest(filt, n_clusters, sparsity_threshold=sparsity_threshold)
        else:
            out, values, labels = vq_and_back(filt, n_clusters, sparsity_threshold=sparsity_threshold)
        out_combined = out
    #else:
    #   raise Exception('We Should not be here')

    out_tensor = out_combined

    return out_tensor, values, np.reshape(labels, out_tensor.shape)


def train(args, model, device, x, y, optimizer,criterion, task_id_nominal, consolidated_masks, replay_memory=None, replay_size=4):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)

    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if ((i + args.batch_size_train) <= len(r)):
            b=r[i:i+args.batch_size_train]
        else:
            b=r[i:]

        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output, memory = model(data, task_id_nominal, mask=None, mode="train")

        if i < replay_size*args.batch_size_train and replay_memory is not None:
            replay_memory.append(memory.detach().cpu().numpy())

        loss = criterion(output, target)
        loss.backward()

        # Continual Subnet no backprop
        curr_head_keys = ["last.{}.weight".format(task_id_nominal), "last.{}.bias".format(task_id_nominal)]
        if consolidated_masks is not None and consolidated_masks != {}: # Only do this for tasks 1 and beyond
            # if args.use_continual_masks:
            for key in consolidated_masks.keys():

                # Skip if not task head is not for curent task
                if 'last' in key:
                    if key not in curr_head_keys:
                        continue

                # Determine whether it's an output head or not
                if (len(key.split('.')) == 3):  # e.g. last.1.weight
                    module_name, task_num, module_attr = key.split('.')
                    # curr_module = getattr(model.module, module_name)[int(task_num)]
                else: # e.g. fc1.weight
                    module_name, module_attr = key.split('.')
                    # curr_module = getattr(model.module, module_name)

                # Zero-out gradients
                if (hasattr(getattr(model, module_name), module_attr)):
                    if (getattr(getattr(model, module_name), module_attr) is not None):
                        getattr(getattr(model, module_name), module_attr).grad[consolidated_masks[key] == 1.0] = 0
                #if (hasattr(getattr(model.module, module_name), module_attr)):
                #    if (getattr(getattr(model.module, module_name), module_attr) is not None):
                #        getattr(getattr(model.module, module_name), module_attr).grad[consolidated_masks[key] == 1.0] = 0
        
        per_task = {}

        optimizer.step()

        if i % 30 == 0:
        
            per_task[task_id_nominal] = model.get_masks(task_id_nominal)
            sd = model.state_dict()
           
            for k_, v in sd.items():
                if 'weight' in k_ and k_[7:] in per_task[task_id_nominal].keys():
                    #if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):

                    new_weight = v*(per_task[task_id_nominal][k_[7:]] == 1).float()
                    other_weights = v*(per_task[task_id_nominal][k_[7:]] != 1).float()

                    q_weight, values, labels = vquant(new_weight, n_clusters=16)                 
                    q_weight = torch.from_numpy(q_weight).cuda()             
                
                    q_weight[per_task[task_id_nominal][k_[7:]] != 1] = 0

                    new_weight = q_weight*(per_task[task_id_nominal][k_[7:]] == 1).float()
                    sd[k_] = new_weight + other_weights
            
            model.load_state_dict(sd)  
        

def test(args, model, device, x, y, criterion, task_id_nominal, curr_task_masks=None, mode="test"):
    model.eval()
    total_loss = 0
    total_num = 0
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if ((i + args.batch_size_test) <= len(r)):
                b=r[i:i+args.batch_size_test]
            else: b=r[i:]

            data = x[b]
            data, target = data.to(device), y[b].to(device)
            if curr_task_masks:
                output, _ = model(data, task_id_nominal, mask=curr_task_masks, mode=mode)
            else:
                output, _ = model(data, task_id_nominal, mask=None, mode=mode)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc


def generate_samples(device, x, y):
    memory = []
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    for i in range(0,len(r),args.batch_size_train):
        if ((i + args.batch_size_train) <= len(r)):
            b=r[i:i+args.batch_size_train]
        else:
            b=r[i:]

        data = x[b]
        data, target = data.to(device), y[b].to(device)

        for image_idx in range(data.size()[0]):
            if len(memory) < args.replay_memory_size:
                memory.append(data[image_idx])
            else:
                break
    return memory


def main(args):
    ## Device Setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## Prime task mask settings
    save_flag = True

    ## Load PermutedMNIST
    from dataloader import pmnist
    data, taskcla, inputsize = pmnist.get(seed=args.seed,
                                          pc_valid=args.pc_valid,
                                          nperm=args.nperm)

    tstart=time.time()
    acc_matrix=np.zeros((args.nperm,args.nperm))
    sparsity_matrix = []
    sparsity_per_task, saver_dict = {}, {}
    criterion = torch.nn.CrossEntropyLoss()

    # Replay memory
    kld = nn.KLDivLoss()
    replay_memory = {}

    # Model Instantiation
    model = MLPNet(taskcla, args.sparsity, memory=args.memory).to(device)
    #model = nn.DataParallel(model)

    print ('Model parameters ---')
    for k_t, (m, param) in enumerate(model.named_parameters()):
        print (k_t,m,param.shape)
    print ('-'*40)

    task_id = 0
    task_list = []
    per_task_masks, consolidated_masks = {}, {}

    task_models = {}
    task_consolidated_masks = {}
    per_task_masks = {}

    models = []
    ptm = []
    cm = []
    models.append(model)
    ptm.append(per_task_masks)
    cm.append(consolidated_masks)

    own_mask = {}
    common_mask = {}
    sparsities = {}

    for k, ncla in taskcla:

        if task_id == 5:
            model_ = MLPNet(taskcla, args.sparsity, memory=args.memory).to(device)
            #model_ = nn.DataParallel(model_)
            models.append(model_)    
            per_task_masks_, consolidated_masks_ = {}, {} 
            cm.append(consolidated_masks_)
            ptm.append(per_task_masks_)         

        if save_flag:
            saver_dict[task_id] = {}

        print('*'*40)
        print('Task {:2d} ({:s})'.format(k,data[k]['name']))
        print('*'*40)
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =data[k]['test']['x']
        ytest =data[k]['test']['y']
        task_list.append(k)

        memory_samples = generate_samples(device, xtrain, ytrain)
        replay_memory[task_id] = memory_samples 

        if args.KL_on:
            min_kld = 1.0
            id = -1
            for t in range(task_id-1, -1): 
                kld_ = kld(torch.from_numpy(np.asarray(replay_memory[task_id])), torch.from_numpy(np.asarray(replay_memory[t])))
                if kld_ < min_kld:
                    min_kld_ = kld_
                    id = t

            if min_kld_ > args.KL_threshold: 
                id = -1 

            if id == -1:
                #create new model
                model = MLPNet(taskcla, args.sparsity, memory=args.memory).to(device) 
            else: 
                #model by id
                model = models[id] 

        lr = args.lr
        best_loss=np.inf
        print ('-'*40)
        print ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print ('-'*40)

        if task_id > 0 and task_id < 10:
            model = torch.load('task_' + str(task_id - 1) + '.pt')
            consolidated_masks = torch.load('consolidated_masks.pt')
            per_task_masks = torch.load('task_masks.pt')

        #if task_id > 5:
        #    model = torch.load('task_' + str(task_id - 1) + '_.pt')
        #    consolidated_masks = torch.load('consolidated_masks_.pt')
        #    per_task_masks = torch.load('task_masks_.pt')

        #best_model=get_model(model)
        if task_id < 10:
            best_model=get_model(models[0])
        else:
            best_model=get_model(models[1])

        if args.optim == "sgd":
            if task_id < 10:   
                optimizer = optim.SGD(models[0].parameters(), lr=lr)
            else:
                optimizer = optim.SGD(models[1].parameters(), lr=lr)
        elif args.optim == "adam":
            if task_id < 10:
                optimizer = optim.Adam(models[0].parameters(), lr=lr)
            else:
                optimizer = optim.Adam(models[1].parameters(), lr=lr)
        else:
            raise Exception("[ERROR] The optimizer " + str(args.optim) + " is not supported!")  

        for epoch in range(1, args.n_epochs+1):
            # Train
            clock0 = time.time()
            if epoch == args.n_epochs:
                if task_id < 10:
                    train(args, models[0], device, xtrain, ytrain, optimizer, criterion, task_id, cm[0], replay_memory=replay_memory[task_id])
                    #train(args, model, device, xtrain, ytrain, optimizer, criterion, task_id, consolidated_masks, replay_memory=replay_memory[task_id])
                else:
                    train(args, models[1], device, xtrain, ytrain, optimizer, criterion, task_id, cm[1], replay_memory=replay_memory[task_id])
            else:
                if task_id < 10:
                    train(args, models[0], device, xtrain, ytrain, optimizer, criterion, task_id, cm[0])
                    #train(args, model, device, xtrain, ytrain, optimizer, criterion, task_id, consolidated_masks)
                else:
                    train(args, models[1], device, xtrain, ytrain, optimizer, criterion, task_id, cm[1])

            clock1 = time.time()

            if task_id < 10:

                tr_loss,tr_acc = test(args, models[0], device, xtrain, ytrain,  criterion, task_id, curr_task_masks=None, mode="valid")
                #tr_loss,tr_acc = test(args, model, device, xtrain, ytrain,  criterion, task_id, curr_task_masks=consolidated_masks, mode="valid")
            else:
                tr_loss,tr_acc = test(args, models[1], device, xtrain, ytrain,  criterion, task_id, curr_task_masks=None, mode="valid")
            clock2=time.time()
            print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms | test time={:5.1f}ms'.format(epoch,\
                                                        tr_loss,tr_acc, 1000*(clock1-clock0), (clock2 - clock1)*1000 ), end='')
            
            # Validate
            if task_id < 10:
                #valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, task_id, curr_task_masks=None, mode="valid")
                valid_loss,valid_acc = test(args, models[0], device, xvalid, yvalid,  criterion, task_id, curr_task_masks=None, mode="valid")
            else:
                valid_loss,valid_acc = test(args, models[1], device, xvalid, yvalid,  criterion, task_id, curr_task_masks=None, mode="valid")
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
                          
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                if task_id < 10:
                    best_model=get_model(models[0])
                    #best_model=get_model(model)
                else:
                    best_model=get_model(models[1])
                patience=args.lr_patience
                print(' *',end='')
            else:
                patience-=1

                if patience<=0:
                    lr/=args.lr_factor
                    print(' lr={:.1e}'.format(lr),end='')
                    if lr<args.lr_min:
                        print()
                        break
                    patience=args.lr_patience
                    adjust_learning_rate(optimizer, epoch, args)
            print()

        # Restore best model
        if task_id < 10:
            set_model_(models[0], best_model)
            #set_model_(model, best_model)
        else:
            set_model_(models[1], best_model)

        #model = nn.DataParallel(model)

        if task_id < 10:
            ptm[0][task_id] = models[0].get_masks(task_id)
            #ptm[0][task_id] = models[0].module.get_masks(task_id)
            per_task_masks[task_id] = model.get_masks(task_id)
        else:
            ptm[1][task_id] = models[1].get_masks(task_id)
            #ptm[1][task_id] = models[1].module.get_masks(task_id)

        # Consolidate task masks to keep track of parameters to-update or not
        curr_head_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]

        if task_id == 0:
            #consolidated_masks = deepcopy(per_task_masks[task_id])
            cm[0] = deepcopy(ptm[0][task_id])
        #elif task_id == 5:
        #    cm[1] = deepcopy(ptm[1][task_id])
        else:
            if task_id < 10:
                ptm_ = ptm[0]
            else:
                ptm_ = ptm[1]

            for key in ptm_[task_id].keys():
                # Skip output head from other tasks
                # Also don't consolidate output head mask after training on new tasks; continue
                if "last" in key:
                    if key in curr_head_keys:
                        if task_id < 10:
                            cm[0][key] = deepcopy(ptm[0][task_id][key])
                            #consolidated_masks[key] = deepcopy(per_task_masks[task_id][key])
                        else:
                            cm[1][key] = deepcopy(ptm[1][task_id][key])
                    continue

                # Or operation on sparsity
                if task_id < 10:
                    cm_ = cm[0]
                else:
                    cm_ = cm[1]
                if cm_[key] is not None and ptm_[task_id][key] is not None:
                    if task_id < 10:
                        cm[0][key] = 1-((1-cm[0][key])*(1-ptm[0][task_id][key]))
                        #consolidated_masks[key] = 1-((1-consolidated_masks[key])*(1-per_task_masks[task_id][key]))
                    else:
                        cm[1][key] = 1-((1-cm[1][key])*(1-ptm[1][task_id][key]))

        if k >= 0:

            #post pruning
            for iteration in range(args.post_pruning_iterations):
          
                keys = own_mask[task_id].keys()
                layer_id = np.random.choice(range(len(keys)))

                keys = list(keys)
                key = keys[layer_id]

                if own_mask[task_id][key] is not None:
                    own_mask_bp = deepcopy(own_mask[task_id][key])

                    if iteration == 0:
                        n_consolidated_masks_before = deepcopy(consolidated_masks)
                    else:
                        n_consolidated_masks_before = deepcopy(n_consolidated_masks)
                    
                sd = model.state_dict()

                if own_mask[task_id][key] is not None and ('module.' + key) in sd.keys() :
                    idx = (own_mask[task_id][key] > 0).nonzero()

                    temp = own_mask[task_id][key] * sd['module.' + key]
                    # mult weight and own_mask

                    idx = torch.where(temp != 0)
                    values = sd['module.' + key][idx] 
                    # get > 0

                    k = 1 + round(args.sparsity_scaler * (values.numel() - 1))
                    kvalue = torch.abs(values).cpu().kthvalue(k)[0].item()

                    temp[torch.abs(temp) < kvalue] = 0
                    own_mask[task_id][key][temp == 0] = 0


                if task_id == 0:
                    n_consolidated_masks = deepcopy(own_mask[task_id])
                else:
                    for key in own_mask[task_id].keys():
                        # Skip output head from other tasks
                        # Also don't consolidate output head mask after training on new tasks; continue
                        if "last" in key:
                            if key in curr_head_keys:
                                n_consolidated_masks[key] = deepcopy(own_mask[task_id][key])
                            continue

                        # Or operation on sparsity
                        if n_consolidated_masks[key] is not None and own_mask[task_id][key] is not None:
                            #n_consolidated_masks[key] = 1-((1-n_consolidated_masks[key])*(1-own_mask[task_id][key]))
                            n_consolidated_masks[key] = 1-((1-consolidated_masks_before[key])*(1-own_mask[task_id][key]))

                tr_loss_, tr_acc_ = test(args, model, device, xtrain, ytrain,  criterion, task_id, curr_task_masks=n_consolidated_masks, mode="valid")
            
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,  criterion, task_id, curr_task_masks=consolidated_masks, mode="valid")
                clock2=time.time()
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms | test time={:5.1f}ms'.format(epoch,\
                                                        tr_loss,tr_acc, 1000*(clock1-clock0), (clock2 - clock1)*1000 ), end='')

                #update sensitivity, go forward or cancel last step
                if tr_acc_ > tr_acc - args.drop_acc_threshold:
                    if own_mask[task_id][key] is not None:
                        n_consolidated_masks[key] = deepcopy(n_consolidated_masks[key])
                        #own_mask[task_id][key] = deepcopy(own_mask[task_id][key])
                else:  
                    if own_mask[task_id][key] is not None:
                        n_consolidated_masks[key] = deepcopy(n_consolidated_masks_before[key])
                        own_mask[task_id][key] = own_mask_bp

            if task_id < 10:
                #sd = model.state_dict()
                sd = models[0].state_dict()
            else:
                sd = models[1].state_dict()

            for k_, v in sd.items():
                if 'weight' in k_ and k_[7:] in consolidated_masks.keys():
                    #if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):

                    new_weight = v*(per_task_masks[task_id][k_[7:]] == 1).float()
                    other_weights = v*(per_task_masks[task_id][k_[7:]] != 1).float()

                    q_weight, values, labels = vquant(new_weight, n_clusters=args.clusters)                 
                    q_weight = torch.from_numpy(q_weight).cuda()             
                
                    q_weight[per_task_masks[task_id][k_[7:]] != 1] = 0

                    new_weight = q_weight*(per_task_masks[task_id][k_[7:]] == 1).float()
                    sd[k_] = new_weight + other_weights
            
            if task_id < 10:
                sd = models[0].load_state_dict(sd)
            else:
                sd = models[1].load_state_dict(sd)
            
            if task_id < 10:
                tr_loss,tr_acc = test(args, models[0], device, xtrain, ytrain,  criterion, task_id, curr_task_masks=cm[0], mode="valid")
                #tr_loss,tr_acc = test(args, model, device, xtrain, ytrain,  criterion, task_id, curr_task_masks=consolidated_masks, mode="valid")
            else:
                tr_loss,tr_acc = test(args, models[1], device, xtrain, ytrain,  criterion, task_id, curr_task_masks=cm[1], mode="valid")
            clock2=time.time()
            print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms | test time={:5.1f}ms'.format(epoch,\
                                                        tr_loss,tr_acc, 1000*(clock1-clock0), (clock2 - clock1)*1000 ), end='')

            if args.quantization_on:
  
                from dhc.quantization.basic_routines.quantization_routines import linear_
                from dhc.quantization.basic_routines.quantization_routines import linear_hist

                from dhc.quantization.parameters_quantization import ParametersQuantization
                parameter_quantization = ParametersQuantization(quantization_procedure=linear_, word_len=args.weight_bits)
                model, _, _ = parameter_quantization.get_quantized_model(model=model.cpu(), task_id=task_id)
                model.cuda()

                from dhc.quantization.activation_quantization import ActivationQuantization
                activations_quant = ActivationQuantization(quantization_procedure=linear_hist, word_len=args.activation_bits)
                model = activations_quant.get_stats_(model.cpu()) 
                model.cuda()

                tr_loss,tr_acc = test(args, models[0], device, xtrain, ytrain,  criterion, task_id, curr_task_masks=cm[0], mode="valid")
                model = activations_quant.get_quantized_model_(model.cpu())
                model.cuda()

        if task_id < 10:
            torch.save(models[0], 'task_' + str(task_id) + '.pt')
            torch.save(consolidated_masks, 'consolidated_masks.pt')
            torch.save(per_task_masks, 'task_masks.pt')
        else:        
            torch.save(models[1], 'task_' + str(task_id) + '_.pt')
            torch.save(consolidated_masks, 'consolidated_masks_.pt')
            torch.save(per_task_masks, 'task_masks_.pt')

        # === saver ===
        if save_flag:

            if task_id < 10:
                saver_dict[task_id]['per_task_masks'] = models[0].get_masks(task_id)
                saver_dict[task_id]['consolidated_masks'] = cm[0]
                saver_dict = save_model_params(saver_dict, models[0], task_id)
            else:
                saver_dict[task_id]['per_task_masks'] = models[1].get_masks(task_id)
                saver_dict[task_id]['consolidated_masks'] = cm[1]
                saver_dict = save_model_params(saver_dict, models[1], task_id)
        
        # Print Sparsity

        if task_id < 10:
            sparsity_per_layer = print_sparsity(cm[0])
            all_sparsity = global_sparsity(cm[0])
            print("Global Sparsity: {}".format(all_sparsity))
            sparsity_matrix.append(all_sparsity)
            sparsity_per_task[task_id] = sparsity_per_layer
        else:
            sparsity_per_layer = print_sparsity(cm[1])
            all_sparsity = global_sparsity(cm[1])
            print("Global Sparsity: {}".format(all_sparsity))
            sparsity_matrix.append(all_sparsity)
            sparsity_per_task[task_id] = sparsity_per_layer

        # Test
        print ('-'*40)
 
        if task_id < 10:
            test_loss, test_acc = test(args, models[0], device, xtest, ytest,  criterion, task_id, curr_task_masks=ptm[0][task_id], mode="test")
            #test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, task_id, curr_task_masks=per_task_masks[task_id], mode="test")
        else:
            test_loss, test_acc = test(args, models[1], device, xtest, ytest,  criterion, task_id, curr_task_masks=ptm[1][task_id], mode="test")
        print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))

        # save accuracy
        jj = 0
        for ii in np.array(task_list)[0:task_id+1]:
            if jj < task_id:
                acc_matrix[task_id, jj] = acc_matrix[task_id-1, jj]
            else:
                xtest = data[ii]['test']['x']
                ytest = data[ii]['test']['y']
                if task_id < 10:
                    _, acc_matrix[task_id,jj] = test(args, models[0], device, xtest, ytest,criterion, jj, curr_task_masks=ptm[0][jj], mode="test")
                    #_, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion, jj, curr_task_masks=per_task_masks[jj], mode="test")
                else:
                   _, acc_matrix[task_id,jj] = test(args, models[1], device, xtest, ytest,criterion, jj, curr_task_masks=ptm[1][jj], mode="test")
            jj +=1


        # save accuracy
        jj = task_id + 1
        for ii in range(task_id+1,10):

            xtest = data[ii]['test']['x']
            ytest = data[ii]['test']['y']
            if task_id < 10:
                _, acc_matrix[task_id,jj] = test(args, models[0], device, xtest, ytest,criterion, jj, curr_task_masks=ptm[0][task_id], mode="test")
                #_, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion, jj, curr_task_masks=per_task_masks[jj], mode="test")
            else:
                _, acc_matrix[task_id,jj] = test(args, models[1], device, xtest, ytest,criterion, jj, curr_task_masks=ptm[1][task_id], mode="test")
            jj +=1

        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(i_a + 1):
                print('{:5.1f} '.format(acc_matrix[i_a,j_a]),end='')
            print()

        # update task id
        task_id +=1     
    

    print('-'*40)
    print ('Task Order : {}'.format(np.array(task_list)))
    print ('Diagonal Final Avg Accuracy: {:5.2f}%'.format( np.mean([acc_matrix[i,i] for i in range(len(taskcla))] )))
    print ('Final Avg accuracy: {:5.2f}%'.format( np.mean([acc_matrix[i,i] for i in range(len(taskcla))] )))

    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1])
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    print('-'*40)
    print('Model compressed with {} clusters: the model weigths are represented in codebook by {} bits. Including sparsity the percentage of original weights capacity is {} percentage.'.format(args.clusters, int(np.log2(args.clusters)), (100*all_sparsity)/(32/np.log2(args.clusters))))
    print()
    

if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--batch_size_test', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=100, metavar='N',
                        help='number of training epochs/task (default: 5)')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--pc_valid',default=0.1,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--optim', type=str, default="sgd", metavar='OPTIM',
                        help='optimizer choice')
    parser.add_argument('--lr', type=float, default=3e-1, metavar='LR',
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
    parser.add_argument('--n_hidden', type=int, default=256, metavar='NH',
                        help='number of hidden units in MLP (default: 100)')
    parser.add_argument('--n_outputs', type=int, default=10, metavar='NO',
                        help='number of output units in MLP (default: 10)')
    parser.add_argument('--n_tasks', type=int, default=10, metavar='NT',
                        help='number of tasks (default: 10)')
    parser.add_argument('--memory', type=int, default=1, metavar='NT',
                        help='number of tasks (default: 10)')

    # CUDA parameters
    parser.add_argument('--gpu', type=str, default="0", metavar='GPU',
                        help="GPU ID for single GPU training")
    # CSNB parameters
    parser.add_argument('--sparsity', type=float, default=0.5, metavar='SPARSITY',
                        help="Target current sparsity for each layer")
    parser.add_argument('--pquant', type=float, default=0.01, metavar='SPARSITY',
                        help="increase sparsity for layer")
    # PMNIST parameters
    parser.add_argument('--nperm', type=int, default=10, metavar='NPERM',
                        help='number of permutations/tasks')

    parser.add_argument('--KL_threshold', type=float, default=0.9)
    parser.add_argument('--KL_on', type=int, default=0)
    parser.add_argument('--replay_memory_size', type=int, default=100)
    parser.add_argument('--post_pruning_iterations', type=int, default=0)
    parser.add_argument('--quantization_on', type=int, default=0)
    parser.add_argument('--activation_bits', type=int, default=16)
    parser.add_argument('--weight_bits', type=int, default=16)
    parser.add_argument('--clusters', type=int, default=16)
    parser.add_argument('--sparsity_scaler', type=float, default=0.2)
    parser.add_argument('--drop_acc_threshold', type=float, default=0.2)

    args = parser.parse_args()
    args.sparsity = 1 - args.sparsity

    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)

    main(args)
