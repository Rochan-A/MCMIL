from os import listdir
from os.path import isfile, join
import os.path as osp

onlyfiles = list(set([f.split('_')[0] for f in listdir('/content/train/') if isfile(join('/content/train/', f))]))

import pandas as pd

image_meta = pd.read_csv('train.csv')

scores = pd.unique(image_meta['gleason_score'])

sampled = {}
for score in scores:
    sampled[score] = image_meta[image_meta['gleason_score'] == score].sample(n=20)['image_id']

"""# Gleason pattern 3 or not"""

import os.path as osp

import pickle
import cv2
import numpy as np
import scipy.io as sio

import torch
from torch.utils import data
import torchvision.transforms as tvt

from PIL import Image
import tifffile as tiff

class PANDA_dataset(data.Dataset):
    """torch.data.Dataset for PANDA Dataset"""
    def __init__(self, files, scores, image_meta, size=128):
        super(PANDA_dataset, self).__init__()

        self.score = list(scores)
        self.files = files
        self.input_sz = size
        self.image_meta = image_meta

        self.label__ = ['0', 'negative', '3', '4', '5'] 

    def _load_data(self, image_id):

        # rand_sample = np.random.randint(0, 16)

        # print(image_meta.loc[image_meta['image_id'] == image_id]['gleason_score'])

        lbl = np.zeros(len(self.score), )

        for rand_sample in range(0, 16):
            image_path = osp.join('/content/train/' + image_id + "_" + str(rand_sample) + ".png")
            label_path = osp.join('/content/masks/' + image_id + "_" + str(rand_sample) + ".png")

            try:
                image = cv2.imread(image_path)
                label = cv2.imread(label_path)
            except:
                return None, None

            unique, counts = np.unique(label[:, :, 0], return_counts=True)

            unique, counts = list(unique), list(counts)

            if 0 in unique:
                ix = unique.index(0)
                unique.pop(ix)
                counts.pop(ix)

            if len(unique) == 0:
                return None, None

            if unique  == [1]:
                a = self.label__.index('0')
                hot = [0]*5
                hot[a] = 1
                return image, np.asarray([a, a+5])

            if 1 in unique:
                ix = unique.index(1)
                unique.pop(ix)
                counts.pop(ix)

            if unique  == [2]:
                a = self.label__.index('negative')
                hot = [0]*5
                hot[a] = 1
                return image, np.asarray([a, a+5])

            if 2 in unique:
                ix = unique.index(2)
                unique.pop(ix)
                counts.pop(ix)

            ma = max(unique)

            sorte = [x for _, x in sorted(zip(counts, unique), key=lambda pair: pair[0])]

            if len(sorte) >= 2:
                a = self.label__.index(str(sorte[1]))
                b = self.label__.index(str(sorte[0]))
                hot = [0]*5
                hot[a] = 1
                hot1 = [0]*5
                hot1[b] = 1
                return image, np.asarray([a, b+5])

            if len(sorte) == 1:
                a = self.label__.index(str(sorte[0]))
                hot = [0]*5
                hot[a] = 1

                return image, np.asarray([a, a+5])

    def _prepare_train(self, index, img, label):

        img1 = img.astype(np.uint8)

        img1 = img1.astype(np.float32) / 255.

        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        return img1, label

    def __getitem__(self, index):
        image_id = self.files[index]
        image, label = self._load_data(image_id)

        if image is None:
            return None

        return self._prepare_train(index, image, label)

    def __len__(self):
        return len(self.files)

    def _check_gt_k(self):
        raise NotImplementedError()

    def _filter_label(self):
        raise NotImplementedError()

    def search_by_id(self, id):
        image, label = self._load_data(id)
        print(label)

        return id

normalize = tvt.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_dataset = PANDA_dataset(onlyfiles, scores, image_meta, normalize)
val_dataset = PANDA_dataset(onlyfiles, scores, image_meta, normalize)

# for score in scores:
#     id = image_meta[image_meta['gleason_score'] == score].sample(n=1)['image_id'].to_string(index=False).strip()
#     print(train_dataset.search_by_id(id))
# train_dataset[85]

def my_collate(batch):
    len_batch = len(batch) # original batch length
    batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
    if len_batch > len(batch): # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
        diff = len_batch - len(batch)
        for i in range(diff):
            batch = batch + batch[:diff]
    return torch.utils.data.dataloader.default_collate(batch)

from torch.utils.data import DataLoader
import time

import torch
from torch import nn, optim
from torch import nn
from torch.nn import functional as F

from tqdm.notebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import metrics as mtx
from sklearn import model_selection as ms

import inspect

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn = my_collate)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, collate_fn = my_collate)

import random
import pickle
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import torch.nn.functional as F
import preconditioned_stochastic_gradient_descent as psgd#download psgd from https://github.com/lixilinx/psgd_torch 
import utilities as U

batch_size = 32
lr = 0.01#will aneal learning_rate to 0.001, 0.0001
num_epochs = 21#about two days on 1080ti with model ks_5_dims_128_192_256 (7.5M coefficients)
enable_dropout = False#our model is small (compared with others for this task), dropout may not be very helpful or may need more epochs to train
train_from_sketch = True
pre_check_svhn_reading = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if train_from_sketch:
    # model coefficients
    ks, dim1, dim2, dim3 = 5, 128, 192, 256
    dim0 = 3#RGB images
    W1 = torch.tensor(torch.randn(dim0*ks*ks+1, dim1)/(dim0*ks*ks)**0.5, requires_grad=True, device=device)
    W2 = torch.tensor(torch.randn(dim1*ks*ks+1, dim1)/(dim1*ks*ks)**0.5, requires_grad=True, device=device)
    W3 = torch.tensor(torch.randn(dim1*ks*ks+1, dim1)/(dim1*ks*ks)**0.5, requires_grad=True, device=device)
    # decimation by 2, i.e., stride=2
    W4 = torch.tensor(torch.randn(dim1*ks*ks+1, dim2)/(dim1*ks*ks)**0.5, requires_grad=True, device=device)
    W5 = torch.tensor(torch.randn(dim2*ks*ks+1, dim2)/(dim2*ks*ks)**0.5, requires_grad=True, device=device)
    W6 = torch.tensor(torch.randn(dim2*ks*ks+1, dim2)/(dim2*ks*ks)**0.5, requires_grad=True, device=device)
    # another three layers
    W7 = torch.tensor(torch.randn(dim2*ks*ks+1, dim3)/(dim2*ks*ks)**0.5, requires_grad=True, device=device)
    W8 = torch.tensor(torch.randn(dim3*ks*ks+1, dim3)/(dim3*ks*ks)**0.5, requires_grad=True, device=device)
    W9 = torch.tensor(torch.randn(dim3*ks*ks+1, dim3)/(dim3*ks*ks)**0.5, requires_grad=True, device=device)
    # detection layer
    W10 =torch.tensor(torch.randn(dim3*ks*ks+1, 10+1)/(dim3*ks*ks)**0.5, requires_grad=True, device=device)
else:
    with open('PANDA_MCMIL_1.pickle', 'rb') as f:
        Ws = pickle.load(f)
        W1,W2,W3,W4,W5,W6,W7,W8,W9,W10 = Ws
        ks, dim1, dim2, dim3 = int((W3.shape[0]/W3.shape[1])**0.5), W3.shape[1], W6.shape[1], W9.shape[1]
        dim0 = 3#RGB images

# CNN model
def model(x): 
    x = F.leaky_relu(F.conv2d(x, W1[:-1].view(dim1,dim0,ks,ks), bias=W1[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W2[:-1].view(dim1,dim1,ks,ks), bias=W2[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W3[:-1].view(dim1,dim1,ks,ks), bias=W3[-1], padding=ks//2), negative_slope=0.1)
    if enable_dropout:
        x = 2*torch.bernoulli(torch.rand(x.shape,device=device))*x  
    #print(x.shape)
    x = F.leaky_relu(F.conv2d(x, W4[:-1].view(dim2,dim1,ks,ks), bias=W4[-1], padding=ks//2, stride=2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W5[:-1].view(dim2,dim2,ks,ks), bias=W5[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W6[:-1].view(dim2,dim2,ks,ks), bias=W6[-1], padding=ks//2), negative_slope=0.1)
    if enable_dropout:
        x = 2*torch.bernoulli(torch.rand(x.shape,device=device))*x
    #print(x.shape)   
    x = F.leaky_relu(F.conv2d(x, W7[:-1].view(dim3,dim2,ks,ks), bias=W7[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W8[:-1].view(dim3,dim3,ks,ks), bias=W8[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W9[:-1].view(dim3,dim3,ks,ks), bias=W9[-1], padding=ks//2), negative_slope=0.1)
    if enable_dropout:
        x = 2*torch.bernoulli(torch.rand(x.shape,device=device))*x
    #print(x.shape)
    x = F.conv2d(x, W10[:-1].view(10+1,dim3,ks,ks), bias=W10[-1])
    #print(x.shape)
    return x

def train_loss(images, labels):
    y = model(images)
    # print(y[0], labels[0])
    y = F.log_softmax(y, 1).double()#really need double precision to calculate log Prb({lables}|Image)
    loss = 0.0
    for i in range(y.shape[0]):
        loss -= U.log_prb_labels(y[i], labels[i])    
    return loss/y.shape[0]/y.shape[2]/y.shape[3]

def test_acc():
    num_errors = 0
    for num_iter, data in enumerate(train_loader):
        im, test_labels = data[0][0], data[1].tolist()[0]

        image = torch.tensor(im/256, dtype=torch.float, device=device)
        y = model(image[None])[0]
        
        # plt_cnt += 1
        # if plt_cnt<=8:
        #     plt.subplot(8,2,2*plt_cnt)
        # else:
        #     break
        
        l = set()
        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                _, label = torch.max(y[:,i,j], dim=0)
                if label < 10:
                    l.add(str(label.item()))

        if set(test_labels) != l:
            num_errors += 1

    return num_errors/len(train_dataset)

# train our model; use PSGD-Newton for optimization (virtually tuning free)

Ws = [W1,W2,W3,W4,W5,W6,W7,W8,W9,W10]
Qs = [[torch.eye(W.shape[0], device=device), torch.eye(W.shape[1], device=device)] for W in Ws]
grad_norm_clip_thr = 0.05*sum(W.shape[0]*W.shape[1] for W in Ws)**0.5 
TrainLoss, TestLoss, BestTestLoss = [], [], 1e30
t0 = time.time()
for epoch in range(num_epochs):
    for num_iter, data in enumerate(train_loader):
        images, labels = data[0], data[1].tolist()
    # for num_iter in range(int((len_train+len_extra1+len_extra2)/batch_size)):
    #     images, labels = get_batch()            

        new_images = torch.tensor(images/256, dtype=torch.float, device=device)
        new_labels = []
        for label in labels:
            new_labels.append(U.remove_repetitive_labels(label))

        # print('Labels: ', labels, type(labels), type(labels[0]), type(labels[0][0]), 'New Labels: ', new_labels)

        # print(new_labels, labels)
        loss = train_loss(new_images, new_labels)

        grads = grad(loss, Ws, create_graph=True)
        TrainLoss.append(loss.item())     
    
        v = [torch.randn(W.shape, device=device) for W in Ws]
        Hv = grad(grads, Ws, v)  
        with torch.no_grad():
            Qs = [psgd.update_precond_kron(q[0], q[1], dw, dg) for (q, dw, dg) in zip(Qs, v, Hv)]
            pre_grads = [psgd.precond_grad_kron(q[0], q[1], g) for (q, g) in zip(Qs, grads)]
            grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
            lr_adjust = min(grad_norm_clip_thr/(grad_norm + 1.2e-38), 1.0)
            for i in range(len(Ws)):
                Ws[i] -= lr_adjust*lr*pre_grads[i]
            
        if num_iter%100==0:     
            print('epoch: {}; iter: {}; train loss: {}; elapsed time: {}'.format(epoch, num_iter, TrainLoss[-1], time.time()-t0))
       
    TestLoss.append(test_loss())
    if TestLoss[-1] < BestTestLoss:
        BestTestLoss = TestLoss[-1]
        with open('PANDA_MCMIL_1.pickle', 'wb') as f:
            pickle.dump(Ws, f)
    print('epoch: {}; best test loss: {}; learning rate: {}'.format(epoch, BestTestLoss, lr))
    if epoch+1==int(num_epochs/3) or epoch+1==int(num_epochs*2/3):
        lr *= 0.1

# with open('PANDA_MCMIL_1.pickle', 'wb') as f:
#         Ws = [W1,W2,W3,W4,W5,W6,W7,W8,W9,W10]
#         pickle.dump(Ws, f)
