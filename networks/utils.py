
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import numpy as np
from copy import deepcopy

import math
from itertools import combinations, permutations

# ------------------prime generation and prime mod tables---------------------
# Find closest number in a list
def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def save_model_params(saver_dict, model, task_id):

    print ('saving model parameters ---')

    saver_dict[task_id]['model']={}
    for k_t, (m, param) in enumerate(model.named_parameters()):
        saver_dict[task_id]['model'][m] = param
        print (k_t,m,param.shape)
    print ('-'*30)

    return saver_dict

def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']=args.lr
        else:
            param_group['lr'] /= args.lr_factor

def is_prime(n):
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def get_primes(num_primes):
    primes = []
    for num in range(2,np.inf):
        if is_prime(num):
            primes.append(num)
            print(primes)

        if len(primes) >= num_primes:
            return primes

def checker(per_task_masks, consolidated_masks, task_id):
    # === checker ===
    for key in per_task_masks[task_id].keys():
        # Skip output head from other tasks
        # Also don't consolidate output head mask after training on new tasks; continue
        if "last" in key:
            if key in curr_head_keys:
                consolidated_masks[key] = deepcopy(per_task_masks[task_id][key])
            continue

        # Or operation on sparsity
        if 'weight' in key:
            num_cons = consolidated_masks[key].sum()
            num_prime = (prime_masks[key] > 0).sum()

            if num_cons != num_prime:
                print('diff.')

def print_sparsity(consolidated_masks, percent=1.0, item=False):
    sparsity_dict = {}
    for key in consolidated_masks.keys():
        # Skip output heads
        if "last" in key:
            continue

        mask = consolidated_masks[key]
        if mask is not None:
            sparsity = float(torch.sum(mask == 1)) / float(np.prod(mask.shape))
            print("{:>12} {:>2.4f}".format(key, sparsity ))

            if item :
                sparsity_dict[key] = sparsity.item() * percent
            else:            
                sparsity_dict[key] = sparsity * percent

    return sparsity_dict

def global_sparsity(consolidated_masks):
    denum, num = 0, 0
    for key in consolidated_masks.keys():
        # Skip output heads
        if "last" in key:
            continue

        mask = consolidated_masks[key]
        if mask is not None:
            num += torch.sum(mask == 1).item()
            denum += np.prod(mask.shape)

    return num / denum


def mask_projected (mask, feature_mat):

    # mask Projections
    for i, key in enumerate(mask.keys()):
        #for j in range(len(feature_mat)):
        if 'weight' in key:
            mask[key] = mask[key] - torch.mm(mask[key].float(), feature_mat[i])
        else:
            None

    return mask

## Define LeNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))
