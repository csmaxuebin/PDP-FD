import statistics
from itertools import combinations

import matplotlib
from torch import nn

matplotlib.use('Agg')
import matplotlib.pyplot as pltz
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torchsummary import summary
import time
import random
import logging
import json
from hashlib import md5
import copy
import easydict
import os
import sys
from collections import defaultdict, deque, OrderedDict
from torch.utils.tensorboard import SummaryWriter
import pickle

import datetime
# Directory where the json file of arguments will be present    存放json的参数文件的目录
directory = './Parse_Files'

# Import different files required for loading dataset, model, testing, training
from utility.LoadSplit import Load_Dataset, Load_Model, Load_Model1
# from utility.options import args_parser
from models.Update import train_client_w, test_client, finetune_client, Distilling, similarity, Dynamically_allocate, \
    train_client_g, Dynamically_persion_layers, Dynamically_persion_layers1, Dynamically_persion_layers2, \
    Dynamically_persion_layers3, Dynamically_persion_layers4, \
    calculate_average, vector_variance, Dynamically_persion_layers5
from models.Fed import FedAvg, DiffPrivFedAvg
from models.test import test_img

torch.manual_seed(0)


if __name__ == '__main__':
    
    # Initialize argument dictionary 初始化参数字典
    args = {}

    # From Parse_Files folder, get the name of required parse file which is provided while running this script from bash
    f = directory+'/'+str(sys.argv[1])
    print(f)
    with open(f) as json_file:  
        args = json.load(json_file)

    # Taking hash of config values and using it as filename for storing model parameters and logs
    param_str = json.dumps(args)
    file_name1 = md5(param_str.encode()).hexdigest()

    # Converting args to easydict to access elements as args.device rather than args[device]
    args = easydict.EasyDict(args)
    print(args)

    # Save configurations by making a file using hash value 通过使用哈希值创建文件来保存配置
    with open('./config/parser_{}.txt'.format(file_name1),'w') as outfile:
        json.dump(args,outfile,indent=4)

    # Setting the device - GPU or CPU
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # ================================================创建存放数据的文件==================================================
    # 定义一个.txt文件名
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    file_name = 'accuracy_{}.txt'.format(timestamp)

    # 指定文件路径和文件名
    file_path = './results.txt/'

    # 如果文件路径不存在，则创建它
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # 将文件路径和文件名合并起来
    file_path_name = os.path.join(file_path, file_name)
    #=================================================导入数据集以及初始化模型=============================================

    # Load the training and testing datasets
    dataset_train, dataset_test, dict_users = Load_Dataset(args=args)

    #splitting user data into training and testing parts 将用户数据拆分为训练和测试部分
    train_data_users = {}
    test_data_users = {}

    #=======计算第n项值=======
    def nth_term10(n, d):
        nth_term = 0.185 + (n - 1) * d
        return nth_term

    def nth_term5(n, d):
        nth_term = 0.082 + (n - 1) * d
        return nth_term

    def nth_term2(n, d):
        nth_term = 0.0219 + (n - 1) * d
        return nth_term

    for i in range(args.num_users):
        dict_users[i] = list(dict_users[i])
        train_data_users[i] = list(random.sample(dict_users[i],int(args.split_ratio*len(dict_users[i]))))
        test_data_users[i] = list(set(dict_users[i])-set(train_data_users[i]))
        # test_data_users[i] = random.sample(test_data_users[i], 500)
# =================================================初始化模型===========================================================
    # Initialize Global Server Model
    net_glob = Load_Model(args=args)

    # 打印每一层参数
    # for k,v in net_glob.state_dict().items():
    #     print(k)

    net_glob.train()

    # copy weights 复制权重
    w_glob = net_glob.state_dict()

    # local models for each client 每个客户端的本地模型
    local_nets = {}
    if args.pb_dataset == "pbCifar100":
        # 获取原模型的最后一层，并替换为10输出的全连接层，并初始化该层参数
        last_layer = list(net_glob.children())[-1]
        input_dim = last_layer.in_features
        new_last_layer = nn.Linear(input_dim, 10)
        new_last_layer.weight.data.normal_(0, 0.01)
        new_last_layer.bias.data.zero_()

        for i in range(0, args.num_users):
            local_nets[i] = Load_Model(args=args)
            local_nets[i].train()
            local_nets[i].load_state_dict(w_glob)
            list(local_nets[i].children())[-1] = new_last_layer
    else:
        for i in range(0,args.num_users):
            local_nets[i] = Load_Model(args=args)      #创建新的网络模型
            local_nets[i].train()                      #进入训练模式
            local_nets[i].load_state_dict(w_glob)      #全局模型的参数加载到本地模型

    #创建集成模型
    Ensemble_model = Load_Model1(args=args)
# =================================================开始联邦学习训练========================================================

    # Start training 开始训练

    print("Start Training")
    print("网络模型：",args.model)
    print("本地数据集：",args.dataset)
    print("服务器数据集：", args.pb_dataset)
    print("初始基础层个数：", args.base_layers)
    print("差分隐私机制：", args.dp_mechanism)
    print("总隐私预算：", args.dp_epsilon)

    start = time.time()
    all_accuracy = 0
    args.dp_epsilon /= args.epochs
    all_acc_list = []
    all_global_acc_list = []
    dis = []
    count_var = 0
    gradient_dict = {}
    cookie = 0
    count = [0, 0, 0, 0]
    count_noise = []
    total_epsilon = [0,0,0,0,0,0,0,0,0,0]
    for idx in range(0,args.num_users):
        gradient_dict[idx] = deque(maxlen=10)

    for iter in range(args.epochs):
        print("---------Round {}---------".format(iter))
#======================================================================客户端============================================
        w_locals, loss_locals = [], []
        acc_list = []
        average_value = []
        total_diff = []
        mean_total_diff = 0
        max_value = 0
        mean_mean = 0
        variance = 0

        if iter == 0:
            base_layers = args.base_layers
        # 隐私预算单调递减10
        # dp_epsilon = nth_term10(iter+1, -0.0017)
        # print("dp_epsilon:", dp_epsilon)

        # 隐私预算单调递减5
        # dp_epsilon = nth_term5(iter + 1, -0.000647)
        # print("dp_epsilon:", dp_epsilon)

        # 隐私预算单调递减2
        # dp_epsilon = nth_term2(iter + 1, -0.00004)
        # print("dp_epsilon:", dp_epsilon)

        for idx in range(0,args.num_users):
            t_e = total_epsilon[idx]

            # 隐私预算平均分配
            # dp_epsilon = args.dp_epsilon

            # 判断是否需要输出隐私预算的值
            if args.dp_mechanism != 'no_dp':
                if iter == 0:
                    dp_epsilon = args.dp_epsilon
                else:
                    dp_epsilon_value = eps_list[idx]
                    dp_epsilon = dp_epsilon_value.item()
            if args.dp_mechanism == 'no_dp':
                # 客户端模型进行训练（不加噪）
                w, loss = train_client_w(args, dataset_train, train_data_users[idx], net=local_nets[idx])
                w_locals.append(w)
                loss_locals.append(copy.deepcopy(loss))

            else :
                # 客户端模型进行训练（在梯度上加噪）
                w, loss, list_it_grad, total_epsilon[idx] = train_client_g(args, dp_epsilon, cookie, base_layers, t_e, dataset_train, train_data_users[idx], net=local_nets[idx])
                w_locals.append(w)
                loss_locals.append(copy.deepcopy(loss))

# =========================================跳出局部最优======================================================================
                if args.jump_lo_op:
                    # # 寻找趋于稳定的梯度轮次
                    if cookie == 0:
                        params = torch.cat([p.view(-1) for p in list_it_grad])
                        gradient_dict[idx].append(params)
                    elif cookie == 1:
                        if len(gradient_dict[idx]) > 0:
                            gradient_dict[idx].clear()
                    # 当梯度列表长度达到5时，计算所有向量之间的差异
                    # print("滑动窗口大小：", len(gradient_dict[idx]))
                    if len(gradient_dict[idx]) == 10:
                        gradients = list(gradient_dict[idx])  # 获取梯度列表中的向量

                        # 分割前四轮和所有五轮的梯度
                        gradients_first_four_rounds = gradients[:9]
                        gradients_all_rounds = gradients

                        # 计算前四轮和所有五轮的欧氏距离方差
                        # 计算所有向量之间的欧氏距离
                        euclidean_distances_first_four_rounds = [torch.norm(g1 - g2) for g1, g2 in
                                                                 combinations(gradients_first_four_rounds, 2)]
                        euclidean_distances_first_four_rounds = [tensor.item() for tensor in
                                                                 euclidean_distances_first_four_rounds]

                        euclidean_distances_all_rounds = [torch.norm(g1 - g2) for g1, g2 in
                                                          combinations(gradients_all_rounds, 2)]
                        euclidean_distances_all_rounds = [tensor.item() for tensor in euclidean_distances_all_rounds]

                        # 方差
                        variance_first_four_rounds = np.var(euclidean_distances_first_four_rounds)
                        variance_all_rounds = np.var(euclidean_distances_all_rounds)

                        if variance_all_rounds < variance_first_four_rounds:
                            c_var = 1
                        else:
                            c_var = 0
                        count_var += c_var
        print("累计消耗隐私预算：", total_epsilon)
        if cookie == 1:
            cookie = 0
        # print("方差稳定降低的个数：", count_var)
# ===============================================================================================================
        s = 0
        for i in range(args.num_users):
            # print("训练集数量：", len(train_data_users[i]))
            # print("测试集数量：", len(test_data_users[i]))
            acc_train, loss_train = test_client(args,dataset_train,train_data_users[i],local_nets[i])
            acc_test, loss_test = test_client(args,dataset_train,test_data_users[i],local_nets[i])
            s += acc_test
            acc_list.append(acc_test)

        s /= args.num_users
        all_acc_list.append(s)
        # all_accuracy += s

        iter_loss = sum(loss_locals) / len(loss_locals)
        # 将结果保存到文件中
        with open(file_path_name, 'a') as f:
            f.write("round: " + str(iter) + "  ")
            f.write('loss: {:.14f}'.format(iter_loss) + "  ")
            f.write('Accuracy: {:.4f}  '.format(s))
            f.write(f"累积消耗的隐私预算: {total_epsilon}")

        print("客户端的平均准确率: {: .3f}".format(s))

        # 统计准确率差值的正负======================================调整个性化层=================================
        new_acc = []
        if iter == 0:
            old_acc = acc_list
        else:
            new_acc = acc_list

            result = [x - y for x, y in zip(new_acc, old_acc)]
            positive_count = 0
            negative_count = 0

            for num in result:
                if num > 0:
                    positive_count += 1
                elif num < 0:
                    negative_count += 1
            old_acc = new_acc

# ======================================================================服务器=====================================================================
        # 将本地模型的前base_layers层参数提取出来
        w_ens = {}  # 保存每一个客户端的基础层参数
        w_Ens = []  # 将所有客户端的基础层参数添加到列表里
        sim = []   # 保存这一轮客户端的相似度

        for idx in range(args.num_users):
            for i in list(w_locals[idx].keys())[0:base_layers]:
                w_ens[i] = copy.deepcopy(w_locals[idx][i])
            w_Ens.append(w_ens)

        # 聚合基础层的参数
        w_Ensemble = FedAvg(w_Ens)

        if args.model == 'ResNet':
            Personal_layer = base_layers - 218
        elif args.model == 'cnn':
            if args.dataset == 'cifar':
                Personal_layer = base_layers - 10
            elif args.dataset == 'mnist':
                Personal_layer = base_layers - 8
        elif args.model == 'MobileNet':
            Personal_layer = base_layers - 164
        elif args.model == 'ResNet50':
            Personal_layer = base_layers - 320
        elif args.model == 'ResNet18':
            Personal_layer = base_layers - 122

        # 全局模型的个性化层
        new_personal_layer = {}                             # 全局模型的个性化层参数
        for i in list(w_glob.keys())[Personal_layer:]:
            new_personal_layer[i] = copy.deepcopy(w_glob[i])

        # 将个性化层参数添加到基础层参数后面
        w_Ensemble.update(new_personal_layer)

        # 集成模型的参数复制到集成模型上
        Ensemble_model.load_state_dict(w_Ensemble)

        #进行知识蒸馏
        Distilling(args, net_glob, Ensemble_model, iter)
        # print("知识蒸馏完成")

        # copy weights
        w_glob = net_glob.state_dict()

        # 如果不添加噪声，则不需要计算相似度
        if args.dp_mechanism != 'no_dp':
            # 计算相似度并分配隐私预算
            w_loc_base = {}
            for idx in range(args.num_users):
                for i in list(w_locals[idx].keys())[0:base_layers]:
                    w_loc_base[i] = copy.deepcopy(w_locals[idx][i])

                list_2 = [value for value in w_loc_base.values()]
                # list_2 = [value for value in w_locals[idx].values()]
                params2 = torch.cat([p.view(-1) for p in list_2])
                # 计算本地模型与全局模型的相似度
                similar = similarity(w_glob, params2, base_layers)
                sim.append(similar)

            minimum = min(sim)
            maximum = max(sim)
            # print("最低相似度值:{:.5f}".format(minimum))
            # print("最高相似度值:{:.5f}".format(maximum))
            # 动态分配隐私预算
            print("隐私预算分配")
            eps_list = Dynamically_allocate(args, sim, minimum)
            # print("nan之前的：", eps_list)
            eps_list = [torch.where(torch.isnan(x).cuda(), torch.tensor(args.dp_epsilon).cuda(), x.cuda()) for x in eps_list]
            # eps_list = np.nan_to_num(eps_list, nan=args.dp_epsilon)
            # print("nan之后的：", eps_list)

        # 跳出局部最优解
        if args.jump_lo_op:
            # 判断是否趋于稳定，更新大的隐私预算
            if count_var > 8:
                # 原方法
                # fixed_value = (8 / 9 + 0.1 - 8 * args.dp_epsilon / 0.9) * args.dp_epsilon
                # 新方法
                rate = (0.1-args.dp_epsilon)/(0.1-0.01)
                fixed_value = (1 - rate) * args.dp_epsilon + rate * 0.01 * 0.9
                # fixed_value = args.dp_epsilon * 0.5
                eps_list = [torch.tensor(fixed_value) for _ in eps_list]
                print("添加大噪声以跳出局部最优:", fixed_value)
                count_noise.append(iter + 1)
                count_var = 0
                cookie = 1
            else:
                count_var = 0
        # ======================================================动态调整个性化层数=========================================
        # 个性化层修改ResNet========================================
        # if args.model == 'ResNet':
        #     if iter != 0:
        #         lay = Dynamically_persion_layers(base_layers, negative_count, positive_count)
        #         base_layers = lay
        #     Personal_layer = base_layers - 218
        #     if Personal_layer == -2:
        #         l = 1
        #     elif Personal_layer == -14:
        #         l = 2
        #     elif Personal_layer == -26:
        #         l = 3
        #     elif Personal_layer == -44:
        #         l = 4
        # # 个性化层修改ResNet18========================================
        # if args.model == 'ResNet18':
        #     if iter != 0:
        #         lay = Dynamically_persion_layers5(base_layers, negative_count, positive_count)
        #         base_layers = lay
        #     Personal_layer = base_layers - 122
        #     if Personal_layer == -2:
        #         l = 1
        #     elif Personal_layer == -14:
        #         l = 2
        #     elif Personal_layer == -32:
        #         l = 3
        #     elif Personal_layer == -44:
        #         l = 4
        # # 个性化层修改MobileNet=================================
        # if args.model == 'MobileNet':
        #     if iter != 0:
        #         lay = Dynamically_persion_layers1(base_layers, negative_count, positive_count)
        #         base_layers = lay
        #     Personal_layer = base_layers - 164
        #     if Personal_layer == -2:
        #         l = 1
        #     elif Personal_layer == -14:
        #         l = 2
        #     elif Personal_layer == -26:
        #         l = 3
        #     elif Personal_layer == -38:
        #         l = 4
        # # 个性化层修改cnn=================================
        # if args.model == 'cnn':
        #     if args.dataset == 'cifar':
        #         if iter != 0:
        #             lay = Dynamically_persion_layers4(base_layers, negative_count, positive_count)
        #             base_layers = lay
        #         Personal_layer = base_layers - 10
        #         if Personal_layer == -2:
        #             l = 1
        #         elif Personal_layer == -4:
        #             l = 2
        #
        #     elif args.dataset == 'mnist':
        #         if iter != 0:
        #             lay = Dynamically_persion_layers2(base_layers, negative_count, positive_count)
        #             base_layers = lay
        #         Personal_layer = base_layers - 8
        #         if Personal_layer == -2:
        #             l = 1
        #         elif Personal_layer == -4:
        #             l = 2
        # # 个性化层修改ResNet50========================================
        # if args.model == 'ResNet50':
        #     if iter != 0:
        #         lay = Dynamically_persion_layers3(base_layers, negative_count, positive_count)
        #         base_layers = lay
        #     Personal_layer = base_layers - 320
        #     if Personal_layer == -2:
        #         l = 1
        #     elif Personal_layer == -20:
        #         l = 2
        #     elif Personal_layer == -38:
        #         l = 3
        #     elif Personal_layer == -62:
        #         l = 4
        #
        # print("动态选择后的基础层数:", base_layers)
        # print("个性化层数:{}".format(l))
        # count[l - 1] += 1
        if iter < args.epochs * 0.8 :
            if args.model == 'ResNet':
                if iter != 0:
                    lay = Dynamically_persion_layers(base_layers, negative_count, positive_count)
                    base_layers = lay
                Personal_layer = base_layers - 218
                if Personal_layer == -2:
                    l = 1
                elif Personal_layer == -14:
                    l = 2
                elif Personal_layer == -26:
                    l = 3
                elif Personal_layer == -44:
                    l = 4
            # 个性化层修改ResNet18========================================
            if args.model == 'ResNet18':
                if iter != 0:
                    lay = Dynamically_persion_layers5(base_layers, negative_count, positive_count)
                    base_layers = lay
                Personal_layer = base_layers - 122
                if Personal_layer == -2:
                    l = 1
                elif Personal_layer == -14:
                    l = 2
                elif Personal_layer == -32:
                    l = 3
                elif Personal_layer == -44:
                    l = 4
            # 个性化层修改MobileNet=================================
            if args.model == 'MobileNet':
                if iter != 0:
                    lay = Dynamically_persion_layers1(base_layers, negative_count, positive_count)
                    base_layers = lay
                Personal_layer = base_layers - 164
                if Personal_layer == -2:
                    l = 1
                elif Personal_layer == -14:
                    l = 2
                elif Personal_layer == -26:
                    l = 3
                elif Personal_layer == -38:
                    l = 4
            # 个性化层修改cnn=================================
            if args.model == 'cnn':
                if args.dataset == 'cifar':
                    if iter != 0:
                        lay = Dynamically_persion_layers4(base_layers, negative_count, positive_count)
                        base_layers = lay
                    Personal_layer = base_layers - 10
                    if Personal_layer == -2:
                        l = 1
                    elif Personal_layer == -4:
                        l = 2

                elif args.dataset == 'mnist':
                    if iter != 0:
                        lay = Dynamically_persion_layers2(base_layers, negative_count, positive_count)
                        base_layers = lay
                    Personal_layer = base_layers - 8
                    if Personal_layer == -2:
                        l = 1
                    elif Personal_layer == -4:
                        l = 2
            # 个性化层修改ResNet50========================================
            if args.model == 'ResNet50':
                if iter != 0:
                    lay = Dynamically_persion_layers3(base_layers, negative_count, positive_count)
                    base_layers = lay
                Personal_layer = base_layers - 320
                if Personal_layer == -2:
                    l = 1
                elif Personal_layer == -20:
                    l = 2
                elif Personal_layer == -38:
                    l = 3
                elif Personal_layer == -62:
                    l = 4
            # print("个性化层数:{}".format(l))
            count[l - 1] += 1
        elif iter == args.epochs * 0.8:
            max_value = max(count)  # 找到列表中的最大值
            max_index = [i for i, value in enumerate(count) if value == max_value]  # 查找最大值的索引

            # 根据索引输出对应的数值
            for index in max_index:
                if index == 0:
                    if args.model == 'ResNet':
                        base_layers = 216
                    elif args.model == 'ResNet18':
                        base_layers = 120
                    elif args.model == 'ResNet50':
                        base_layers = 318
                    elif args.model == 'cnn':
                        if args.dataset == 'cifar':
                            base_layers = 8
                        elif args.dataset == 'mnist':
                            base_layers = 6
                    # print("动态选择后的基础层数:", base_layers)
                elif index == 1:
                    if args.model == 'ResNet':
                        base_layers = 204
                    elif args.model == 'ResNet18':
                        base_layers = 108
                    elif args.model == 'ResNet50':
                        base_layers = 300
                    elif args.model == 'cnn':
                        if args.dataset == 'cifar':
                            base_layers = 6
                        elif args.dataset == 'mnist':
                            base_layers = 4
                    # print("动态选择后的基础层数:", base_layers)
                elif index == 2:
                    if args.model == 'ResNet':
                        base_layers = 192
                    elif args.model == 'ResNet18':
                        base_layers = 90
                    elif args.model == 'ResNet50':
                        base_layers = 282
                    # print("动态选择后的基础层数:", base_layers)
                elif index == 3:
                    if args.model == 'ResNet':
                        base_layers = 174
                    elif args.model == 'ResNet18':
                        base_layers = 78
                    elif args.model == 'ResNet50':
                        base_layers = 258
                    # print("动态选择后的基础层数:", base_layers)
        print("动态选择后的基础层数:", base_layers)
# ======================================================================客户端=====================================================================
        # 更新客户端的基础层并保持个性化层不变
        # 将全局模型的前base_layers层参数复制到本地模型
        for idx in range(args.num_users):
            for i in list(w_glob.keys())[0:base_layers]:     #第一个循环遍历所有客户端，而第二个循环则遍历全局模型的前base_layers层的参数
                w_locals[idx][i] = copy.deepcopy(w_glob[i]) 
            local_nets[idx].load_state_dict(w_locals[idx])

        ### FineTuning
        if args.finetune:
            # print("FineTuning")
            personal_params=list(w_glob.keys())[base_layers:]
            for idx in range(0,args.num_users):
                for i,param in enumerate(local_nets[idx].named_parameters()):
                    if param[0] not in personal_params:
                        param[1].requires_grad=False
                w,loss = finetune_client(args,dataset_train,train_data_users[idx],net = local_nets[idx])
                for i,param in enumerate(local_nets[idx].named_parameters()):
                    if param[0] not in personal_params:
                        param[1].requires_grad=True

            s = 0
            for i in range(args.num_users):
                logging.info("Client {}:".format(i))
                acc_train, loss_train = test_client(args,dataset_train,train_data_users[i],local_nets[i])
                acc_test, loss_test = test_client(args,dataset_train,test_data_users[i],local_nets[i])
                s += acc_test
            s /= args.num_users
            all_global_acc_list.append(s)

            print("全局精度: {: .3f}".format(s))
            with open(file_path_name, 'a') as f:
                f.write('Global_Accuracy: {:.4f}\n'.format(s))

    end = time.time()
    print("Training Time: {}s".format(end-start))
    print("End of Training")
# ======================================================================联邦学习结束=====================================================================

    max_acc = max(all_acc_list)
    max_acc1 = max(all_global_acc_list)
    print("本地最高准确率: {: .4f}".format(max_acc))
    print("全局最高准确率: {: .4f}".format(max_acc1))
    print("所有添加大噪的轮次：", count_noise)
    print("累计消耗隐私预算：", total_epsilon)

    for i in range(4):
        print(f"个性化层为{i + 1}：{count[i]}次。")

    with open(file_path_name, 'a') as f:
        f.write('本地最高准确率: {: .4f}\n'.format(max_acc))
        f.write('全局最高准确率: {: .4f}\n'.format(max_acc1))
        f.write(f"累积消耗的隐私预算: {total_epsilon}\n")
        f.write(f"所有添加大噪的轮次: {count_noise}\n")
        for i in range(4):
            f.write(f"个性化层为{i + 1}：{count[i]}次。")

    # save model parameters 保存模型参数
    torch.save(net_glob.state_dict(),'./state_dict/server_{}.pt'.format(file_name1))
    for i in range(args.num_users):
        torch.save(local_nets[i].state_dict(),'./state_dict/client_{}_{}.pt'.format(i,file_name))

    # testing local models 测试本地模型
    # s = 0
    # for i in range(args.num_users):
    #     logging.info("Client {}:".format(i))
    #     acc_train, loss_train = test_client(args,dataset_train,train_data_users[i],local_nets[i])
    #     acc_test, loss_test = test_client(args,dataset_train,test_data_users[i],local_nets[i])
    #     s += acc_test
    # s /= args.num_users
    # print("测试训练完成后的客户端平均准确率: {: .3f}".format(s))
