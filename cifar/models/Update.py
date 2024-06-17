import copy
import math

import torch
import torchvision
from torch import nn, autograd
# from torch.distributed import tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
import numpy as np
import random
from sklearn import metrics
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from models.rdp_accountant import compute_rdp, get_privacy_spent
from itertools import cycle

class DatasetSplit(Dataset):

    """
    Class DatasetSplit - To get datasamples corresponding to the indices of samples a particular client has from the actual complete dataset

    """

    def __init__(self, dataset, idxs):

        """

        Constructor Function

        Parameters:

            dataset: The complete dataset

            idxs : List of indices of complete dataset that is there in a particular client

        """
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):

        """

        returns length of local dataset

        """

        return len(self.idxs)

    def __getitem__(self, item):

        """
        Gets individual samples from complete dataset

        returns image and its label

        """
        image, label = self.dataset[self.idxs[item]]
        return image, label
    

# function to train a client
# 参数上加噪
# def train_client_w(args, dp_epsilon, dataset,train_idx,net):
def train_client_w(args, dataset, train_idx, net):
    '''

    Train individual client models

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : Complete dataset loaded by the Dataloader

        args (dictionary) : The list of arguments defined by the user

        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client

    Returns:

        net.state_dict() (state_dict) : The updated weights of the client model

        train_loss (float) : Cumulative loss while training

    '''


    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()
    
    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []

    for iter in range(args.local_ep):   
        batch_loss = []
        
        for batch_idx, (images, labels) in enumerate(ldr_train):
            
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

# ====================================================batch更新一次参数(保底)====================================================
# def train_client_g(args, dp_epsilon, cookie, base_layers, t_e, dataset, train_idx, net):
#     total_grad = []
#     grad = {}
#     params = dict(net.named_parameters())
#     for name in params:
#         grad[name] = torch.zeros(params[name].shape).to(args.device)
#
#     # loss_func = nn.CrossEntropyLoss(reduction='none')
#     loss_func = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
#     epoch_loss = []
#
#     train_idx = list(train_idx)
#     ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
#
#     net.train()
#     data_size = len(train_idx)
#     # print(data_size)
#     clip = 0.1
#     delta = 1e-5
#
#     if args.model == 'ResNet':
#         if base_layers == 216:
#             tempt = 108
#         elif base_layers == 204:
#             tempt = 102
#         elif base_layers == 192:
#             tempt = 96
#         elif base_layers == 174:
#             tempt = 87
#     elif args.model == 'MobileNet':
#         if base_layers == 162:
#             tempt = 81
#         elif base_layers == 150:
#             tempt = 75
#         elif base_layers == 138:
#             tempt = 69
#     elif args.model == 'ResNet50':
#         if base_layers == 318:
#             tempt = 159
#         elif base_layers == 300:
#             tempt = 150
#         elif base_layers == 282:
#             tempt = 141
#         elif base_layers == 258:
#             tempt = 129
#
#     dp_epsilon = dp_epsilon / args.local_ep
#     sampling_prob = args.local_bs / data_size
#     steps = int(args.local_ep / sampling_prob)
#     z = np.sqrt(2 * np.log(1.25 / delta)) / dp_epsilon
#     sigma, eps = get_sigma(sampling_prob, steps, dp_epsilon, delta, z, rgp=True)
#     s = 2 * clip * args.lr / args.local_bs
#     noise_scale = s * sigma
#     print("noise_scale：", noise_scale)
#
#     for iter in range(args.local_ep):
#         optimizer.zero_grad()
#         batch_loss = []
#         t_e += eps
#         h = 0
#         clipped_grads = {name: torch.zeros_like(param) for name, param in net.named_parameters()}
#         for images, labels in ldr_train:
#             images, labels = images.to(args.device), labels.to(args.device)
#             log_probs = net(images.float())
#             loss = loss_func(log_probs, labels.long())
#
#             loss.backward(retain_graph=True)
#             torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip)
#             # if args.dp_mechanism == 'Laplace':
#             #     # add Laplace noise
#             #     count = 0
#             #     for name, param in net.named_parameters():
#             #         if count > tempt:
#             #             break
#             #         clipped_grads[name] += laplace_noise(args, clipped_grads[name].shape, clip, dp_epsilon,device=args.device)
#             #         count += 1
#             #
#             # elif args.dp_mechanism == 'Gaussian':
#             #     # add Gaussian noise
#             #     count = 0
#             #     for name, param in net.named_parameters():
#             #         if count > tempt:
#             #             break
#             #         clipped_grads[name] += gaussian_noise(args, clipped_grads[name].shape, clip, delta, dp_epsilon,device=args.device)
#             #         count += 1
#
#             if args.dp_mechanism == 'MA':
#                 # add Gaussian noise
#                 count = 0
#                 for name, param in net.named_parameters():
#                     if count > tempt:
#                         break
#                     param.grad += torch.normal(0, noise_scale, size=param.grad.shape, device=args.device)
#                     count += 1
#
#             for name, param in net.named_parameters():
#                 clipped_grads[name] += param.grad
#
#             # update local model
#             optimizer.step()
#             batch_loss.append(loss.item())
#             h += 1
#             net.zero_grad()
#         for name, param in net.named_parameters():
#             clipped_grads[name] /= h
#
#         list_grad = [value for value in clipped_grads.values()]
#         if len(total_grad) == 0:
#             total_grad = list(list_grad)
#         else:
#             total_grad = [x + y for x, y in zip(total_grad, list_grad)]
#
#         for name, param in net.named_parameters():
#             param.grad = clipped_grads[name]
#
#         epoch_loss.append(sum(batch_loss) / len(batch_loss))
#
#     return net.state_dict(), sum(epoch_loss) / len(epoch_loss), total_grad, t_e
def grad_clip(net, clip):
    # 逐样本裁剪
    grads_list = []
    for param in net.parameters():
        if param.grad is not None:
            grads_list.append(param.grad.clone())
        else:
            continue
    # 将梯度列表展平为一维张量
    flatten_grads = torch.cat([grad.flatten() for grad in grads_list])
    grads_clipped = flatten_grads / max(1.0, float(torch.norm(flatten_grads, p=2)) / clip)

    start = 0
    for param in net.parameters():
        size = param.numel()
        if param.grad is None:
            continue
        else:
            param.grad.copy_(grads_clipped[start:start + size].view_as(param.grad))
        start += size

# ====================================================逐样本裁剪(测试)=========================================================
def train_client_g(args, dp_epsilon, cookie, base_layers, t_e, dataset, train_idx, net):
    global_grads = [torch.zeros(size=param.shape).to(args.device) for param in net.parameters()]
    grad = {}

    params = dict(net.named_parameters())
    for name in params:
        grad[name] = torch.zeros(params[name].shape).to(args.device)

    loss_func = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []

    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)

    net.train()
    data_size = len(train_idx)
    # print(data_size)
    clip = 0.7
    delta = 1e-5

    if args.model == 'ResNet':
        if base_layers == 216:
            tempt = 108
        elif base_layers == 204:
            tempt = 102
        elif base_layers == 192:
            tempt = 96
        elif base_layers == 174:
            tempt = 87
    elif args.model == 'MobileNet':
        if base_layers == 162:
            tempt = 81
        elif base_layers == 150:
            tempt = 75
        elif base_layers == 138:
            tempt = 69
    elif args.model == 'ResNet50':
        if base_layers == 318:
            tempt = 159
        elif base_layers == 300:
            tempt = 150
        elif base_layers == 282:
            tempt = 141
        elif base_layers == 258:
            tempt = 129
    elif args.model == 'cnn':
        if args.dataset == 'cifar':
            if base_layers == 8:
                tempt = 8
            elif base_layers == 6:
                tempt = 6
        elif args.dataset == 'mnist':
            if base_layers == 6:
                tempt = 6
            elif base_layers == 4:
                tempt = 4
    elif args.model == 'ResNet18':
        if base_layers == 120:
            tempt = 60
        elif base_layers == 108:
            tempt = 54
        elif base_layers == 90:
            tempt = 45
        elif base_layers == 78:
            tempt = 39

    dp_epsilon = dp_epsilon / args.local_ep
    sampling_prob = args.local_bs / data_size
    steps = int(args.local_ep / sampling_prob)
    z = np.sqrt(2 * np.log(1.25 / delta)) / dp_epsilon
    sigma, eps = get_sigma(sampling_prob, steps, dp_epsilon, delta, z, rgp=True)
    s = 2 * clip * args.lr
    noise_scale = s * sigma
    print("noise_scale：", noise_scale)

    for iter in range(args.local_ep):
        optimizer.zero_grad()
        batch_loss = []
        t_e += eps

        for images, labels in ldr_train:
            h = 0  # 统计实际的batchsize大小
            single_loss = []
            clipped_grads = {name: torch.zeros_like(param) for name, param in net.named_parameters()}
            images, labels = images.to(args.device), labels.to(args.device)
            log_probs = net(images.float())
            loss = loss_func(log_probs, labels.long())

            # bound l2 sensitivity (gradient clipping)
            for i in range(loss.size()[0]):
                loss[i].backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip)
                # grad_clip(net, clip)

                for name, param in net.named_parameters():
                    clipped_grads[name] += param.grad
                net.zero_grad()
                h += 1
                single_loss.append(loss[i].item())

            if args.dp_mechanism == 'MA':
                # add Gaussian noise
                count = 0
                for name, param in net.named_parameters():
                    if count > tempt:
                        break
                    clipped_grads[name] += torch.normal(0, noise_scale, size=clipped_grads[name].shape, device=args.device)
                    count += 1

            for name, param in net.named_parameters():
                clipped_grads[name] /= h

            for name, param in net.named_parameters():
                param.grad = clipped_grads[name]

            for i, param in enumerate(net.parameters()):
                global_grads[i] += param.grad  # 累加当前epoch的梯度到全局梯度

            # update local model
            optimizer.step()
            batch_loss.append(sum(single_loss) / len(single_loss))
            net.zero_grad()

        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    return net.state_dict(), sum(epoch_loss) / len(epoch_loss), global_grads, t_e
# ====================================================逐样本裁剪=========================================================
# def train_client_g(args, dp_epsilon, cookie, base_layers, t_e, dataset, train_idx, net):
#     global_grads = [torch.zeros(size=param.shape).to(args.device) for param in net.parameters()]
#     optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
#     epoch_loss = []
#
#     train_idx = list(train_idx)
#     ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
#
#     net.train()
#     data_size = len(train_idx)
#     # print(data_size)
#     clip = 0.1
#     delta = 1e-5
#
#     if args.model == 'ResNet':
#         if base_layers == 216:
#             tempt = 108
#         elif base_layers == 204:
#             tempt = 102
#         elif base_layers == 192:
#             tempt = 96
#         elif base_layers == 174:
#             tempt = 87
#     elif args.model == 'MobileNet':
#         if base_layers == 162:
#             tempt = 81
#         elif base_layers == 150:
#             tempt = 75
#         elif base_layers == 138:
#             tempt = 69
#     elif args.model == 'ResNet50':
#         if base_layers == 318:
#             tempt = 159
#         elif base_layers == 300:
#             tempt = 150
#         elif base_layers == 282:
#             tempt = 141
#         elif base_layers == 258:
#             tempt = 129
#
#     dp_epsilon = dp_epsilon / args.local_ep
#     sampling_prob = args.local_bs / data_size
#     steps = int(args.local_ep / sampling_prob)
#     z = np.sqrt(2 * np.log(1.25 / delta)) / dp_epsilon
#     sigma, eps = get_sigma(sampling_prob, steps, dp_epsilon, delta, z, rgp=True)
#     s = 2 * clip * args.lr / args.local_bs
#     noise_scale = s * sigma
#     print("noise_scale：", noise_scale)
#
#     for iter in range(args.local_ep):
#         optimizer.zero_grad()
#         batch_loss = []
#         t_e += eps
#
#         for batch_idx,(images,labels) in enumerate(ldr_train):
#             single_loss = []
#             net.zero_grad()
#             total_grads = [torch.zeros(size=param.shape).to(args.device) for param in net.parameters()]
#             for id,(X_microbatch, y_microbatch) in enumerate(TensorDataset(images, labels)):
#                 X_microbatch, y_microbatch = X_microbatch.to(args.device), y_microbatch.to(args.device)
#                 net.zero_grad()
#                 output = net(torch.unsqueeze(X_microbatch.to(torch.float32), 0))
#                 loss = F.cross_entropy(output, torch.unsqueeze(y_microbatch.to(torch.long), 0))
#                 loss.backward()
#
#                 grad_clip(net, clip)
#
#                 # 逐样本累加
#                 grads = [param.grad.detach().clone() for param in net.parameters()]
#                 for idx, grad in enumerate(grads):
#                     total_grads[idx] += grad
#                 single_loss.append(loss.item())
#
#             avg_batch_loss = sum(single_loss) / len(single_loss)
#             for i, param in enumerate(net.parameters()):
#                 param.grad = total_grads[i]
#
#             # 添加噪声
#             if args.dp_mechanism == 'MA':
#                 # add noise
#                 count = 0
#                 for name, param in net.named_parameters():
#                     if count > tempt:
#                         break
#                     param.grad += torch.normal(0, noise_scale, size=param.grad.shape, device=args.device)
#                     count += 1
#
#             for i, param in enumerate(net.parameters()):
#                 param.grad /= args.local_bs
#
#             for i, param in enumerate(net.parameters()):
#                 global_grads[i] += param.grad  # 累加当前epoch的梯度到全局梯度
#
#             optimizer.step()
#             batch_loss.append(avg_batch_loss)
#
#         epoch_loss.append(sum(batch_loss) / len(batch_loss))
#
#     return net.state_dict(), sum(epoch_loss) / len(epoch_loss), global_grads, t_e

def finetune_client(args,dataset,train_idx,net):

    '''

    Train individual client models

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : Complete dataset loaded by the Dataloader

        args (dictionary) : The list of arguments defined by the user

        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client

    Returns:

        net.state_dict() (state_dict) : The updated weights of the client model

        train_loss (float) : Cumulative loss while training

    '''


    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()
    
    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []
    
    for iter in range(1):
        batch_loss = []
        
        for batch_idx, (images, labels) in enumerate(ldr_train):
            
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
    return net.state_dict(),sum(epoch_loss) / len(epoch_loss)


# function to test a client
def test_client(args,dataset,test_idx,net):

    '''

    Test the performance of the client models on their datasets

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : The data on which we want the performance of the model to be evaluated

        args (dictionary) : The list of arguments defined by the user

        test_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local dataset of this client

    Returns:

        accuracy (float) : Percentage accuracy on test set of the model

        test_loss (float) : Cumulative loss on the data

    '''
    
    data_loader = DataLoader(DatasetSplit(dataset, test_idx), batch_size=args.local_bs)  
    net.eval()
    #print (test_data)
    test_loss = 0
    correct = 0
    
    l = len(data_loader)
    
    with torch.no_grad():
                
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = net(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            
            correct += y_pred.eq(target.data.view_as(y_pred)).float().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)

        return accuracy, test_loss

def Dynamically_persion_layers(base_layers,negative_count, positive_count):
    base_layer = base_layers
    # result = [x - y for x, y in zip(new_acc, old_acc)]
    #
    # positive_count = 0
    # negative_count = 0
    #
    # for num in result:
    #     if num > 0:
    #         positive_count += 1
    #     elif num < 0:
    #         negative_count += 1
    #
    # print("Positive count:", positive_count)
    # print("Negative count:", negative_count)
    #==================================个性化层减少=====================================
    # positive_count > negative_count,准确率增加，基础层增加，个性化层减少
    # if positive_count > negative_count:
    #     if base_layer == 216:
    #         base_layer = 216
    #     elif base_layer == 204:
    #         base_layer =216
    #     elif base_layer == 192:
    #         base_layer = 204
    #     elif base_layer == 174:
    #         base_layer = 192
    # # positive_count > negative_count,准确率减少，基础层减少，个性化层增加
    # if positive_count < negative_count:
    #     if base_layer == 216:
    #         base_layer = 204
    #     elif base_layer == 204:
    #         base_layer = 192
    #     elif base_layer == 192:
    #         base_layer = 174
    #     elif base_layer == 174:
    #         base_layer = 174
    # ==================================个性化层增加=====================================
        # positive_count > negative_count,准确率增加，基础层减少，个性化层增加
    if positive_count > negative_count:
        if base_layer == 216:
            base_layer = 204
        elif base_layer == 204:
            base_layer = 192
        elif base_layer == 192:
            base_layer = 174
        elif base_layer == 174:
            base_layer = 174

        # positive_count > negative_count,准确率减少，基础层增加，个性化层减少
    if positive_count < negative_count:
        if base_layer == 216:
            base_layer = 216
        elif base_layer == 204:
            base_layer = 216
        elif base_layer == 192:
            base_layer = 204
        elif base_layer == 174:
            base_layer = 192

    return base_layer

def Dynamically_persion_layers1(base_layers,negative_count, positive_count):
    base_layer = base_layers
    # positive_count > negative_count,准确率增加，基础层减少，个性化层增加
    if positive_count > negative_count:
        if base_layer == 162:
            base_layer = 150
        elif base_layer == 150:
            base_layer = 138
        elif base_layer == 138:
            base_layer = 138

        # positive_count > negative_count,准确率减少，基础层增加，个性化层减少
    if positive_count < negative_count:
        if base_layer == 162:
            base_layer = 162
        elif base_layer == 150:
            base_layer = 162
        elif base_layer == 138:
            base_layer = 150

    return base_layer

def Dynamically_persion_layers2(base_layers,negative_count, positive_count):
    base_layer = base_layers
    # positive_count > negative_count,准确率增加，基础层减少，个性化层增加
    if positive_count > negative_count:
        if base_layer == 6:
            base_layer = 4
        elif base_layer == 4:
            base_layer = 4

        # positive_count > negative_count,准确率减少，基础层增加，个性化层减少
    if positive_count < negative_count:
        if base_layer == 6:
            base_layer = 6
        elif base_layer == 4:
            base_layer = 6

    return base_layer

def Dynamically_persion_layers3(base_layers,negative_count, positive_count):
    base_layer = base_layers
    # positive_count > negative_count,准确率增加，基础层减少，个性化层增加
    if positive_count > negative_count:
        if base_layer == 318:
            base_layer = 300
        elif base_layer == 300:
            base_layer = 282
        elif base_layer == 282:
            base_layer = 258
        elif base_layer == 258:
            base_layer = 258

        # positive_count > negative_count,准确率减少，基础层增加，个性化层减少
    if positive_count < negative_count:
        if base_layer == 318:
            base_layer = 318
        elif base_layer == 300:
            base_layer = 318
        elif base_layer == 282:
            base_layer = 318
        elif base_layer == 258:
            base_layer = 282

    return base_layer

def Dynamically_persion_layers4(base_layers,negative_count, positive_count):
    base_layer = base_layers
    # positive_count > negative_count,准确率增加，基础层减少，个性化层增加
    if positive_count > negative_count:
        if base_layer == 8:
            base_layer = 6
        elif base_layer == 6:
            base_layer = 6

        # positive_count > negative_count,准确率减少，基础层增加，个性化层减少
    if positive_count < negative_count:
        if base_layer == 8:
            base_layer = 8
        elif base_layer == 6:
            base_layer = 8

    return base_layer

def Dynamically_persion_layers5(base_layers,negative_count, positive_count):
    base_layer = base_layers
    # positive_count > negative_count,准确率增加，基础层减少，个性化层增加
    if positive_count > negative_count:
        if base_layer == 120:
            base_layer = 108
        elif base_layer == 108:
            base_layer = 90
        elif base_layer == 90:
            base_layer = 78
        elif base_layer == 78:
            base_layer = 78

        # positive_count > negative_count,准确率减少，基础层增加，个性化层减少
    if positive_count < negative_count:
        if base_layer == 120:
            base_layer = 120
        elif base_layer == 108:
            base_layer = 120
        elif base_layer == 90:
            base_layer = 108
        elif base_layer == 78:
            base_layer = 90

    return base_layer

def Distilling(args, net_glob, Ensemble_model, iter):
    if args.pb_dataset == "pbCifar10":
        # 加载CIFAR10数据集
        # 定义数据增强预处理
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

        # 加载CIFAR-10数据集
        trainset = torchvision.datasets.CIFAR10(root='..\data\pbCifar10', train=True, download=True, transform=transform)

        # 随机生成训练集和测试集的索引
        indices = list(range(len(trainset)))
        train_indices = random.sample(indices, 2560)
        test_indices = list(set(indices) - set(train_indices))
        test_indices = random.sample(test_indices, 1280)

        # 使用Subset函数和DataLoader函数导入数据集
        trainset = torch.utils.data.Subset(trainset, train_indices)
        testset = torch.utils.data.Subset(trainset, test_indices)
        trainloader1 = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader1 = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
        trainloader = trainloader1

    elif args.pb_dataset == "pbCifar100":
        # 加载CIFAR100数据集
        # 定义数据增强预处理
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

        # 加载CIFAR-100数据集
        trainset = torchvision.datasets.CIFAR100(root='..\data\pbCifar100', train=True, download=True,transform=transform)

        # 随机生成训练集和测试集的索引
        indices = list(range(len(trainset)))
        train_indices = random.sample(indices, 1280) #500/1280/3200
        test_indices = list(set(indices) - set(train_indices))
        test_indices = random.sample(test_indices, 1280)

        # 使用Subset函数和DataLoader函数导入数据集
        trainset = torch.utils.data.Subset(trainset, train_indices)
        testset = torch.utils.data.Subset(trainset, test_indices)
        trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader2 = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
        trainloader = trainloader2

    elif  args.pb_dataset == "SVHN":
        # 加载SVHN数据集
        # 定义数据增强预处理
        transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 加载SVHN数据集
        trainset = torchvision.datasets.SVHN(root='..\data\SVHN', split ="train", download=True,transform=transform)

        # 随机生成训练集和测试集的索引
        indices = list(range(len(trainset)))
        train_indices = random.sample(indices, 1280)  # 500/3200
        test_indices = list(set(indices) - set(train_indices))
        test_indices = random.sample(test_indices, 1280)

        # 使用Subset函数和DataLoader函数导入数据集
        trainset = torch.utils.data.Subset(trainset, train_indices)
        testset = torch.utils.data.Subset(trainset, test_indices)
        trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader2 = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
        trainloader = trainloader2

    elif args.pb_dataset == "Usps":
        # 定义数据变换，将数据转换为Tensor并将像素值标准化为[0, 1]之间的范围
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # 导入USPS数据集
        trainset = torchvision.datasets.USPS(root='..\\data\\Usps', train=True, download=True, transform=transform)

        # # 遍历数据集中的每个图像，并将其调整为28x28大小
        # resized_usps_train = [(resize_image(Image.fromarray(x.numpy().squeeze(), mode='L')), y) for x, y in trainset]
        #
        # # 将调整后的图像转换为torch张量
        # trainset = [(transforms.ToTensor()(x), y) for x, y in resized_usps_train]

        # 随机生成训练集和测试集的索引
        indices = list(range(len(trainset)))
        train_indices = random.sample(indices, 1280)  # 500/3200
        test_indices = list(set(indices) - set(train_indices))
        test_indices = random.sample(test_indices, 1280)

        # 使用Subset函数和DataLoader函数导入数据集
        trainset = torch.utils.data.Subset(trainset, train_indices)
        testset = torch.utils.data.Subset(trainset, test_indices)
        trainloader3 = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader2 = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
        trainloader = trainloader3

    elif args.pb_dataset == "FMnist":
        # 加载FashionMNIST数据集
        # 定义数据增强预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # 加载CIFAR-10数据集
        trainset = torchvision.datasets.FashionMNIST(root='..\data\FashionMNIST', train=True, download=True, transform=transform)

        # 随机生成训练集和测试集的索引
        indices = list(range(len(trainset)))
        train_indices = random.sample(indices, 500)
        test_indices = list(set(indices) - set(train_indices))
        test_indices = random.sample(test_indices, 1280)

        # 使用Subset函数和DataLoader函数导入数据集
        trainset = torch.utils.data.Subset(trainset, train_indices)
        testset = torch.utils.data.Subset(trainset, test_indices)
        trainloader4 = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader1 = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
        trainloader = trainloader4

    else:
        print("没有这个公共数据集")

    epochs = 2
    temp = 7
    alpha = 0.5
    hard_loss = nn.CrossEntropyLoss()
    soft_loss = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(net_glob.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for i, (x, y)  in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to('cuda')
            else:
                x = x.to('cuda')
            y = y.to('cuda')
            # 教师模型预测
            with torch.no_grad():
                teacher_preds = Ensemble_model(x)
            # 学生模型预测
            student_preds = net_glob(x)
            student_loss = hard_loss(student_preds, y)
            # 计算蒸馏后的预测结果及soft_loss
            distillation_loss = soft_loss(
                F.softmax(student_preds / temp, dim=1),
                F.softmax(teacher_preds / temp, dim=1)
            ).to('cuda')
            # 将 hard_loss 和 soft_loss 加权求和
            loss = alpha * student_loss + (1 - alpha) * distillation_loss
            # 反向传播,优化权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def resize_image(image):
    # 将PIL Image对象调整为28x28大小
    resized_image = image.resize((28, 28))
    return resized_image

def similarity(w_glob, params2, base_layers):
    glob_base_layer = {}
    for i in list(w_glob.keys())[0:base_layers]:
        glob_base_layer[i] = copy.deepcopy(w_glob[i])
    list_1 = [value for value in glob_base_layer.values()]
    # list_1 = [value for value in w_glob.values()]
    params1 = torch.cat([p.view(-1) for p in list_1])
    # print(params1.shape)
    # print(params2.shape)
    # 计算两个向量之间的余弦相似度
    result = torch.nn.functional.cosine_similarity(params1, params2, dim=0)
    # 归一化
    result = (result + 1) / 2
    # print(f"全局模型与本地模型[{idx}]的归一化相似度：{result}")

    return result

# def euclidean_distance(w_glob, base_layers, params2):
#     glob_base_layer = {}
#     for i in list(w_glob.keys())[0:base_layers]:
#         glob_base_layer[i] = copy.deepcopy(w_glob[i])
#     list_1 = [value for value in glob_base_layer.values()]
#     params1 = torch.cat([p.view(-1) for p in list_1])
#     # print(params1.shape)
#     # print(params2.shape)
#     # 计算两个向量之间的余弦相似度
#     result = np.linalg.norm(params1 - params2)
#     # 计算最大距离值
#     max_distance = np.linalg.norm(np.ones_like(params1) * 10)
#     # 归一化处理
#     result = result / max_distance
#     print(f"全局模型与本地模型的归一化相似度：{result}")
#     return result

def Dynamically_allocate(args, sim, minimum):
    lim_eps = args.dp_epsilon * 0.8
    # args.dp_epsilon /=200

    eps = []
    for idx in range(args.num_users):
        c_eps = args.dp_epsilon * (1 - sim[idx]) / (1 - minimum)
        if c_eps < lim_eps:                # 防止隐私预算过小
            c_eps = torch.tensor(lim_eps)
        # print("客户端{:.0f}的epsilon：{:.4f}".format(idx, c_eps))
        eps.append(c_eps)

    return eps

def add_noise(args, net, dp_epsilon):
    delta = 0.01
    dp_clip = 2
    sensitivity = 1.0

    if args.dp_mechanism == 'Laplace':
        with torch.no_grad():
            for k, v in net.named_parameters():
                noise = torch.from_numpy(np.random.laplace(loc=0, scale=sensitivity / dp_epsilon,size=v.shape)).to(args.device)
                v += noise

    elif args.dp_mechanism == 'Gaussian':
        c = np.sqrt(2 * np.log(1.25 / delta))
        sigma = c * sensitivity / dp_epsilon
        with torch.no_grad():
            for k, v in net.named_parameters():
                noise = torch.from_numpy(np.random.normal(loc=0, scale=sigma, size=v.shape)).to(args.device)
                v += noise

            print(noise)
            print(v)

def laplace_noise(args, data_shape, sensitivity, epsilon, device=None):
    """
       laplace noise
    """
    # noise_scale = 2*sensitivity / epsilon
    noise_scale = (2 * sensitivity / args.local_bs) / epsilon
    noise = np.random.laplace(0, noise_scale, data_shape)
    noise = torch.from_numpy(noise).to(device)

    return noise

def gaussian_noise(args, data_shape, clip, delta, epsilon, device=None):
    """
    Gaussian noise
    """
    # s = 2 * clip
    s = 2 * clip / args.local_bs
    z = np.sqrt(2 * np.log(1.25 / delta))
    print("z = ", format(z))
    sigma = z * s / epsilon
    # print("sigma = ",format(sigma))
    noise = torch.normal(0, sigma, data_shape).to(device)

    return noise

def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    average = total / count
    return average


def vector_variance(vectors):
    n = len(vectors)
    dim = len(vectors[0])
    # 计算每个维度上的平均值
    means = [sum(vec[i] for vec in vectors) / n for i in range(dim)]
    # 计算每个维度上的方差
    variances = [sum((vec[i] - means[i]) ** 2 for vec in vectors) / n for i in range(dim)]
    return variances

def loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rdp_orders=32, rgp=True):
    previous_eps=eps
    while True:
        orders = np.arange(2, rdp_orders, 0.1)
        steps = T
        if (rgp):
            rdp = compute_rdp(q, cur_sigma, steps,
                              orders) * 2  ## when using residual gradients, the sensitivity is sqrt(2)
        else:
            rdp = compute_rdp(q, cur_sigma, steps, orders)
        cur_eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)  # 根据目标delta值计算对应的epsilon值，并获取最优的阶数。
        if (cur_eps < eps and cur_sigma > interval): # 判断当前epsilon值是否小于目标epsilon，并且当前的sigma值是否大于间隔值
            cur_sigma -= interval
            previous_eps = cur_eps
        else:
            cur_sigma += interval
            break
    return cur_sigma, previous_eps


## interval: init search inerval
## rgp: use residual gradient perturbation or not
def get_sigma(q, T, eps, delta, init_sigma, interval=1., rgp=True):
    cur_sigma = init_sigma

    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, previous_eps = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    return cur_sigma, previous_eps
