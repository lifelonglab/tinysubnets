
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
from itertools import combinations, permutations

from utils import *

from networks.subnet import SubnetLinear, SubnetConv2d
from networks.alexnet import SubnetAlexNet_norm as AlexNet
from networks.lenet import SubnetLeNet as LeNet
from networks.utils import *

from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix

import importlib
#from utils_rle import compress_ndarray, decompress_ndarray, comp_decomp_mask
from utils_huffman import comp_decomp_mask_huffman
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


def train(args, model, device, x, y, optimizer, criterion, task_id_nominal, consolidated_masks, epoch=1, alpha=1.0):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)

    gradient = {}

    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if ((i + args.batch_size_train) <= len(r)):
            b=r[i:i+args.batch_size_train]
        else:
            b=r[i:]

        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data, task_id_nominal, mask=None, mode="train", epoch=epoch)

        l1_parameters = []
        for parameter in model.parameters():
            l1_parameters.append(parameter.view(-1))

        l1 = alpha * torch.cat(l1_parameters).pow(2).sum().sqrt()

        loss = criterion(output, target) #+ l1
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
                    # curr_module = getattr(model, module_name)[int(task_num)]
                else: # e.g. fc1.weight
                    module_name, module_attr = key.split('.')
                    # curr_module = getattr(model, module_name)

                # Zero-out gradients
                if (hasattr(getattr(model, module_name), module_attr)):
                    if (getattr(getattr(model, module_name), module_attr) is not None):
                        getattr(getattr(model, module_name), module_attr).grad[consolidated_masks[key] == 1] = 0

        optimizer.step()
        return gradient

def test(args, model, device, x, y, criterion, task_id_nominal, curr_task_masks=None, mode="test", comp_flag=False):
    model.eval()
    comp_ratio = 0
    total_loss = 0
    total_num = 0
    correct = 0
    r=np.arange(x.size(0))
    r=torch.LongTensor(r).to(device)

    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if ((i + args.batch_size_test) <= len(r)):
                b=r[i:i+args.batch_size_test]
            else: b=r[i:]

            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data, task_id_nominal, mask=curr_task_masks, mode=mode)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc, comp_ratio


def main(args):
    tstart=time.time()
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

    ## Load CIFAR100_100 DATASET
    dataloader = importlib.import_module('dataloader.' + args.dataset)

    data, output_info, input_size, n_tasks, n_outputs = dataloader.get(data_path=args.data_path, args=args, seed=args.seed, pc_valid=args.pc_valid, samples_per_task=args.samples_per_task)
    args.samples_per_task = int(data[0]['train']['y'].shape[0] / (1.0 - args.pc_valid))

    # Shuffle tasks
    if args.shuffle_task:
        ids = list(shuffle(np.arange(args.n_tasks), random_state=args.seed))
    else:
        ids = list(np.arange(args.n_tasks))

    print('Task info =', output_info)
    print('Input size =', input_size, '\nOutput number=', n_outputs, '\nTotal task=', n_tasks)
    print('Task order =', ids)
    print('-' * 100)

    tasks_ = [t for t  in ids]
    n_outputs_ = [n_outputs] * n_tasks
    taskcla = [(t,n) for t,n in zip(tasks_, n_outputs_)]

    acc_matrix=np.zeros((10,10))
    sparsity_matrix = []
    sparsity_per_task = {}
    criterion = torch.nn.CrossEntropyLoss()

    # Model Instantiation
    if args.model == "alexnet":
        model = AlexNet(taskcla, args.sparsity).to(device)
        model_ = AlexNet(taskcla, 0.0).to(device)
    elif args.model == "lenet":
        model = LeNet(taskcla, args.sparsity).to(device)
        model_ = LeNet(taskcla, 0.0).to(device)
    else:
        raise Exception("[ERROR] The model " + str(args.model) + " is not supported!")
    print ('Model parameters ---')
    for k_t, (m, param) in enumerate(model.named_parameters()):
        print (k_t,m,param.shape)
    print ('-'*40)

    task_id = 0
    task_list = []
    sparsities = {}
    task_accs = []

    per_task_masks, consolidated_masks, prime_masks = {},{},{}
    per_task_masks_, consolidated_masks_, prime_masks_ = {},{},{}

    # Replay memory
    kld = nn.KLDivLoss()
    replay_memory = {}
    models = []
    models.append(model)

    for k, ncla in taskcla:

        actual_task_sparsity = {}

        sd = model.state_dict()
        for kn, v in sd.items():
            if 'conv' in kn or 'fc' in kn or 'linear' in kn:
                if len(v.size()) >= 2:
                    actual_task_sparsity[kn] = args.sparsity

        model.actual_sparsity = actual_task_sparsity
        #import pdb
        #pdb.set_trace()

        print('*'*100)
        print('Task {:2d} ({:s})'.format(k,data[k]['name']))
        print('*'*100)
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
                model = AlexNet(taskcla, args.sparsity).to(device) 
            else: 
                #model by id
                model = models[id]
       
        lr = args.lr
        best_loss=np.inf
        #best_loss_=np.inf

        print ('-'*40)
        print ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print ('-'*40)

        if task_id == 3:
            best_model_=get_model(model_)
        else:
            best_model=get_model(model)

        if args.optim == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr)
            optimizer_ = optim.SGD(model_.parameters(), lr=lr)
        elif args.optim == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
            optimizer_ = optim.Adam(model_.parameters(), lr=lr)
        else:
            raise Exception("[ERROR] The optimizer " + str(args.optim) + " is not supported!")
        initial_optimizer_state_dict = optimizer.state_dict()

        # reinitialized weight score
        # model.init_masks(task_id=k)

        for epoch in range(1, args.n_epochs+1):
            # Train
            clock0 = time.time()

            #if task_id == 3:
            #    gradient = train(args, model_, device, xtrain, ytrain, optimizer_, criterion, task_id, consolidated_masks_, epoch=epoch)
            #else:
            gradient = train(args, model, device, xtrain, ytrain, optimizer, criterion, task_id, consolidated_masks, epoch=epoch)
            
            clock1 = time.time()

            #if task_id == 3:
            #    tr_loss,tr_acc, comp_ratio = test(args, model_, device, xtrain, ytrain,  criterion, task_id, curr_task_masks=None, mode="valid")
            #else:
            tr_loss,tr_acc, comp_ratio = test(args, model, device, xtrain, ytrain,  criterion, task_id, curr_task_masks=None, mode="valid")
            print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss,tr_acc, 1000*(clock1-clock0)),end='')
            # Validate
            
            #if task_id == 3:
            #    valid_loss,valid_acc, comp_ratio = test(args, model_, device, xvalid, yvalid,  criterion, task_id, curr_task_masks=None, mode="valid")
            #else:
            valid_loss,valid_acc, comp_ratio = test(args, model, device, xvalid, yvalid,  criterion, task_id, curr_task_masks=None, mode="valid")
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')

            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                #if task_id == 3:
                #    best_model_=get_model(model_)
                #else:
                best_model=get_model(model)
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
                    #if task_id == 3:
                    #    adjust_learning_rate(optimizer_, epoch, args)
                    #else:
                    adjust_learning_rate(optimizer, epoch, args)
            print()
            #set sparsities for layer 

        # Restore best model
        #if task_id == 3:
        #    set_model_(model_,best_model)
        #else:
        set_model_(model,best_model)

        # Save the per-task-dependent masks
        
        #if task_id == 3:
        #    per_task_masks_[task_id] = model_.get_masks(task_id)
        #else:
        per_task_masks[task_id] = model.get_masks(task_id)

        #per_task_masks[task_id] = model.get_masks(task_id)

        # Consolidate task masks to keep track of parameters to-update or not
        curr_head_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        if task_id == 0:
            consolidated_masks = deepcopy(per_task_masks[task_id])
        #if task_id == 3:
        #    consolidated_masks_ = deepcopy(per_task_masks_[task_id])

        else:
            if False: #task_id == 3:
                for key in per_task_masks_[task_id].keys():
                    # Skip output head from other tasks
                    # Also don't consolidate output head mask after training on new tasks; continue
                    if "last" in key:
                        if key in curr_head_keys:
                            #if task_id == 3: 
                            #    consolidated_masks_[key] = deepcopy(per_task_masks_[task_id][key])
                            #else:
                            consolidated_masks[key] = deepcopy(per_task_masks[task_id][key])

                    continue
            
                    #if task_id == 1 or task_id == 3:
                    # Or operation on sparsity
                    if consolidated_masks_[key] is not None and per_task_masks_[task_id][key] is not None:
                        consolidated_masks_[key] = 1 - ((1 - consolidated_masks_[key]) * (1 - per_task_masks_[task_id][key]))
                    #else:
                    # Or operation on sparsity
                    #if consolidated_masks[key] is not None and per_task_masks[task_id][key] is not None:
                    #    consolidated_masks[key] = 1 - ((1 - consolidated_masks[key]) * (1 - per_task_masks[task_id][key]))
            else:
                for key in per_task_masks[task_id].keys():
                    # Skip output head from other tasks
                    # Also don't consolidate output head mask after training on new tasks; continue
                    if "last" in key:
                        if key in curr_head_keys:
                            #if task_id == 1 or task_id == 3: 
                            consolidated_masks[key] = deepcopy(per_task_masks[task_id][key])
                            #else:
                            #    consolidated_masks[key] = deepcopy(per_task_masks[task_id][key])

                    continue
            
                    #if task_id == 1 or task_id == 3:
                    # Or operation on sparsity
                    if consolidated_masks[key] is not None and per_task_masks[task_id][key] is not None:
                        consolidated_masks[key] = 1 - ((1 - consolidated_masks[key]) * (1 - per_task_masks[task_id][key]))
                    #else:
                    # Or operation on sparsity
                    #if consolidated_masks[key] is not None and per_task_masks[task_id][key] is not None:
                    #    consolidated_masks[key] = 1 - ((1 - consolidated_masks[key]) * (1 - per_task_masks[task_id][key]))
                    

        #if task_id == 3:
        #    per_task_masks_[task_id] = model_.get_masks(task_id)
        #else:
        per_task_masks[task_id] = model.get_masks(task_id)

        if False: #k >= 0 and task_id == 3:
            sd = model_.state_dict()

            for k_, v in sd.items():
                if 'weight' in k_ and k_ in per_task_masks_[task_id].keys():
                    #if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):

                    new_weight = v*(per_task_masks_[task_id][k_] == 1).float()
                    other_weights = v*(per_task_masks_[task_id][k_] != 1).float()

                    q_weight, values, labels = vquant(new_weight, n_clusters=args.clusters)                 
                    q_weight = torch.from_numpy(q_weight).cuda()             
                
                    q_weight[per_task_masks_[task_id][k_] != 1] = 0

                    new_weight = q_weight*(per_task_masks_[task_id][k_] == 1).float()
                    sd[k_] = new_weight + other_weights
            
            model_.load_state_dict(sd)

        else:
            sd = model.state_dict()

            for k_, v in sd.items():
                if 'weight' in k_ and k_ in per_task_masks[task_id].keys():
                    #if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):

                    new_weight = v*(per_task_masks[task_id][k_] == 1).float()
                    other_weights = v*(per_task_masks[task_id][k_] != 1).float()

                    q_weight, values, labels = vquant(new_weight, n_clusters=args.clusters)                 
                    q_weight = torch.from_numpy(q_weight).cuda()             
                
                    q_weight[per_task_masks[task_id][k_] != 1] = 0

                    new_weight = q_weight*(per_task_masks[task_id][k_] == 1).float()
                    sd[k_] = new_weight + other_weights
            
            model.load_state_dict(sd)

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

                valid_loss,valid_acc, comp_ratio = test(args, model, device, xvalid, yvalid,  criterion, task_id, curr_task_masks=None, mode="valid")
                model = activations_quant.get_quantized_model_(model.cpu())
                model.cuda()

        import timeit
        from timeit import default_timer as timer

        # Print Sparsity

        #if task_id == 3:
        #    sparsity_per_layer_ = print_sparsity(consolidated_masks_)
        #    all_sparsity_ = global_sparsity(consolidated_masks_)

        sparsity_per_layer = print_sparsity(consolidated_masks)
        all_sparsity = global_sparsity(consolidated_masks)
        print("Global Sparsity: {}".format(all_sparsity))
        sparsity_matrix.append(all_sparsity)
        sparsity_per_task[task_id] = sparsity_per_layer    

        # Test
        print ('-'*40)
        #if task_id == 3:
        #    test_loss, test_acc, comp_ratio = test(args, model_, device, xtest, ytest,  criterion, task_id, curr_task_masks=per_task_masks_[task_id], mode="test")
        #    task_accs.append(test_acc)
        #    import pdb
        #    pdb.set_trace()
        #else:
        test_loss, test_acc, comp_ratio = test(args, model, device, xtest, ytest,  criterion, task_id, curr_task_masks=per_task_masks[task_id], mode="test")
        print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))

        # save accuracy
        jj = 0
        for ii in np.array(task_list)[0:task_id+1]:
            if jj < task_id:
                acc_matrix[task_id, jj] = acc_matrix[task_id-1, jj]
            else:
                xtest =data[ii]['test']['x']
                ytest =data[ii]['test']['y']
                #if task_id == 3:
                #    _, acc_matrix[task_id,jj], comp_ratio = test(args, model_, device, xtest, ytest,criterion, jj, curr_task_masks=per_task_masks_[jj], mode="test")
                #else:
                _, acc_matrix[task_id,jj], comp_ratio = test(args, model, device, xtest, ytest,criterion, jj, curr_task_masks=per_task_masks[jj], mode="test")
            jj +=1

        jj = task_id + 1
        for ii in range(task_id+1, 10):
            #if jj < task_id:
            #    acc_matrix[task_id, jj] = acc_matrix[task_id-1, jj]
            #else:
            #import pdb
            #pdb.set_trace()

            xtest = data[ii]['test']['x']
            ytest = data[ii]['test']['y']
            _, acc_matrix[task_id,jj], comp_ratio = test(args, model, device, xtest, ytest,criterion, jj, curr_task_masks=per_task_masks[task_id], mode="test")
            jj +=1

        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(i_a + 1):
                print('{:5.1f} '.format(acc_matrix[i_a,j_a]),end='')
            print()

        # update task id
        task_id +=1        

    model = model.to(device)
    # Test one more time
    test_acc_matrix=np.zeros((10,10))
    sparsity_matrix = []
    mask_comp_ratio = []
    sparsity_per_task = {}
    criterion = torch.nn.CrossEntropyLoss()

    task_list = []
    task_id=0
    for k, ncla in taskcla:
        print('*'*100)
        print('Task {:2d} ({:s})'.format(k,data[k]['name']))
        print('*'*100)
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =data[k]['test']['x']
        ytest =data[k]['test']['y']
        task_list.append(k)
        # Test
        print ('-'*40)
        test_loss, test_acc, comp_ratio = test(args, model, device, xtest, ytest,  criterion, task_id, curr_task_masks=per_task_masks[task_id], mode="test")
        print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))

        # save accuracy
        jj = 0
        for ii in np.array(task_list)[0:task_id+1]:
            if jj < task_id:
                test_acc_matrix[task_id, jj] = acc_matrix[task_id-1, jj]
            else:
                xtest = data[ii]['test']['x']
                ytest = data[ii]['test']['y']

                _, test_acc_matrix[task_id,jj], comp_ratio = test(args, model, device, xtest, ytest,criterion, jj, curr_task_masks=per_task_masks[jj], mode="test", comp_flag=False)
                mask_comp_ratio.append(comp_ratio/32)
            jj +=1
        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(i_a + 1):
                print('{:5.1f} '.format(test_acc_matrix[i_a,j_a]),end='')
            print()

        #model = activations_quant.get_transparent_model(model.cpu())
        #model.cuda()

        # update task id
        task_id +=1

    print('-'*50)

    print()

    sparsity_per_layer = print_sparsity(consolidated_masks)
    all_sparsity = global_sparsity(consolidated_masks)
    print("Global Sparsity: {}".format(all_sparsity))
    print("Bit Mask Capacity: {}%".format(np.sum(comp_ratio)))
    print('Model compressed with {} clusters: the model weigths are represented in codebook by {} bits. Including sparsity the percentage of original weights capacity is {} percentage.'.format(args.clusters, int(np.log2(args.clusters)), (100*all_sparsity)/(32/np.log2(args.clusters)))) 

    print ('Task Order : {}'.format(np.array(task_list)))
    print ('Diagonal Final Avg Accuracy: {:5.2f}%'.format( np.mean([test_acc_matrix[i,i] for i in range(len(taskcla))] )))
    print ('Final Avg accuracy: {:5.2f}%'.format( np.mean(test_acc_matrix[len(taskcla) - 1])))
    bwt=np.mean((test_acc_matrix[-1]-np.diag(acc_matrix))[:-1])
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    print('-'*50)
    print(args)


if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--quantization_mode', type=int, default=0, metavar='N',
                        help='quantization mode (default: 0)')
    parser.add_argument('--pruning_mode', type=int, default=0, metavar='N',
                        help='pruning mode (default: 0)')
    parser.add_argument('--batch_size_test', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=400, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--optim', type=str, default="adam", metavar='OPTIM',
                        help='optimizer choice')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    # CUDA parameters
    parser.add_argument('--gpu', type=str, default="0", metavar='GPU',
                        help="GPU ID for single GPU training")
    # CSNB parameters
    parser.add_argument('--sparsity', type=float, default=0.5, metavar='SPARSITY',
                        help="Target current sparsity for each layer")
    # Model parameters
    parser.add_argument('--model', type=str, default="alexnet", metavar='MODEL',
                        help="Models to be incorporated for the experiment")
    # Deep compression
    parser.add_argument("--deep_comp", type=str, default="", metavar='COMP',
                        help="Deep Compression Model")
    # Pruning threshold
    parser.add_argument("--prune_thresh", type=float, default=0.25, metavar='PRU_TH',
                        help="Pruning threshold for Deep Compression")
    # data parameters
    parser.add_argument('--loader', type=str,
                        default='task_incremental_loader',
                        help='data loader to use')
    # increment
    parser.add_argument('--increment', type=int, default=5, metavar='S',
                        help='(default: 5)')

    parser.add_argument('--data_path', default='./data/', help='path where data is located')
    parser.add_argument("--dataset",
                        default='mnist_permutations',
                        type=str,
                        required=True,
                        choices=['mnist_permutations', 'cifar100_100', 'cifar100_superclass', 'tinyimagenet', 'pmnist'],
                        help="Dataset to train and test on.")


    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')

    parser.add_argument("--workers", default=4, type=int, help="Number of workers preprocessing the data.")

    parser.add_argument("--class_order", default="random", type=str, choices=["random", "chrono", "old", "super"],
                        help="define classes order of increment ")


    # For cifar100
    parser.add_argument('--n_tasks', type=int, default=10,
                        help='total number of tasks, invalid for cifar100_superclass')
    parser.add_argument('--shuffle_task', default=False, action='store_true',
                        help='Invalid for cifar100_superclass')

    parser.add_argument('--encoding', type=str, default="huffman", metavar='',
                        help="")

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



