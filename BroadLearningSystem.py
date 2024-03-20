# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:09:38 2018

@author: HAN_RUIZHI yb77447@umac.mo OR  501248792@qq.com

This code is the first version of BLS Python. 
If you have any questions about the code or find any bugs
   or errors during use, please feel free to contact me.
If you have any questions about the original paper, 
   please contact the authors of related paper.
"""

import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA 
import time


def show_accuracy(predictLabel, Label):
    # 求准确度
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count/len(Label),5))


def tansig(x):
    # tansig激活函数
    return (2/(1+np.exp(-2*x)))-1


def sigmoid(data):
    # sigmoid激活函数
    return 1.0/(1+np.exp(-data))
    

def linear(data):
    return data
    

def tanh(data):
    return (np.exp(data)-np.exp(-data))/(np.exp(data)+np.exp(-data))
    

def relu(data):
    return np.maximum(data, 0)


def pinv(A, reg):
    # 求伪逆公式
    return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)


def shrinkage(a, b):
    """
    软阈值处理
    软阈值处理可以将信号中的一些小的幅度信息压缩到 0，从而使得信号变得更加稀疏，去除噪声等不需要的信息。
    """
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)  # maximum用于获取两个数组中对应元素的最大值，并返回一个新的数组。
    return z


def sparse_bls(A, b):
    """
    稀疏自编码函数，参数A为FeatureOfEachWindowAfterPreprocess，b为FeatureOfInputDataWithBias
    稀疏自编码（sparse autoencoder）是一种常用的无监督学习方法，其主要作用包括特征提取、数据降维和去噪等方面。

    具体来说，稀疏自编码的主要作用如下：
    特征提取：稀疏自编码可以将原始数据中的一些高层次特征提取出来，并且可以学习到对这些特征的稀疏表示。
             这些特征可以用于后续的分类、聚类等任务中，从而达到更好的效果。
    数据降维：稀疏自编码可以将原始数据进行降维，从而减少数据的维度，提高计算效率。降维的同时还可以保留数据中的重要信息，从而避免信息丢失。
    去噪：稀疏自编码可以通过学习对数据的稀疏表示，对输入数据进行去噪。具体来说，
         稀疏自编码可以通过学习对于一组输入数据的稀疏表示，从而将输入数据中的噪声信息去除掉，从而得到更加干净的数据。
    """
    lam = 0.001  # L1 正则化参数 lam
    itrs = 50  # 迭代次数 itrs
    AA = A.T.dot(A)   # AA=A的转置乘以矩阵A
    m = A.shape[1]  # m=A的列数
    n = b.shape[1]  # n=b的列数
    x1 = np.zeros([m, n])  # x1存储一个全零矩阵
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I  # 将矩阵AA主对角线元素加1在求逆
    L2 = (L1.dot(A.T)).dot(b)  # 计算编码的目标值 = A的转置*L1*b
    for i in range(itrs):
        # 求解输入矩阵的稀疏编码结果
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk


def BLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3):
    """
    原始的BLS
    """

    """
    训练过程
    """
    # 1. 对训练数据进行标准化处理
    L = 0  # 初始步数为0
    train_x = preprocessing.scale(train_x, axis=1)  # 对train_x的每个样本标准化处理

    # 2. 生成特征映射层
    # FeatureOfInputDataWithBias 存储训练样本数据+偏置，为每一个训练样本设置偏置，以确保更好的拟合数据
    # OutputOfFeatureMappingLayer 存储特征映射层的输出，也就是提前给特征映射层占个位置
    # Beta1OfEachWindow 存储稀疏编码系数
    # weightOfEachWindow 存储每个样本特征的权重
    # FeatureOfEachWindow 存储单个特征映射数据
    # FeatureOfEachWindowAfterPreprocess 存储单个特征映射归一化后的数据
    # scaler1 对FeatureOfEachWindow中的特征数据进行归一化（使其在0-1内）
    # outputOfEachWindow 存储稀疏自编码后的数据
    # distOfMaxAndMin 在特征选择阶段中，需要计算每个特征的最大值和最小值之间的距离，通过计算这些距离，可以评估每个特征在区分不同类别的能力，
    #                 如果某个特征的最大值和最小值之间的距离太小，那么它可能不能很好地区分不同的类别，因此可以考虑将其从特征集中删除。
    # OutputOfFeatureMappingLayer 存储outputOfEachWindow的数据形成特征映射层

    #  拼接后的新矩阵 FeatureOfInputDataWithBias 包含了输入数据矩阵中的所有特征以及一个额外的全为 0.1 的偏置项
    #  这个偏置项用于确保模型可以学习到输入数据中的偏移量，从而更好地拟合训练数据
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])  # 将train_x后加一列0.1

    # 特征映射层输出
    # 创建全为0的数组，行为train_x的行，列为N2*N1（特征映射窗口数*每个窗口内节点数）
    # 第一维（行）表示输出矩阵中的样本数量，第二维（列）表示输出矩阵中的特征数量。
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2*N1])

    Beta1OfEachWindow = []  # 存储了特征映射层中每个窗口的稀疏编码系数
    distOfMaxAndMin = []  # 存储每个特征的最大值和最小值之间的距离
    minOfEachWindow = []  # 列表的长度等于特征映射层中窗口的数量，每个元素都是一个标量，用于存储对应窗口的最小值。

    # 存储每个训练周期的训练集和测试集准确率
    train_acc_all = np.zeros([1, L+1])  # 定义train_acc_all 是一个长度为 L+1 的一维数组
    test_acc = np.zeros([1, L+1])  # test_acc 是一个长度为 L+1 的一维数组

    train_time = np.zeros([1, L+1])  # train_time 是一个长度为 L+1 的一维数组
    test_time = np.zeros([1, L+1])  # test_time 是一个长度为 L+1 的一维数组
    time_start = time.time()  # 计时开始

    for i in range(N2):  # 遍历特征映射层（Feature Mapping Layer）中的每一个窗口，窗口数为N2=10（在每个窗口内做下面的操作）
        random.seed(i)  # seed（）在循环内每次随机生成的随机数序列中的数都是相同的，每次随机生成10个相同的数

        # 定义每个窗口权重
        # 先随机生成一个大小为 (train_x.shape[1]+1, N1=10) 的随机数矩阵，在乘2减1（控制在1以内）
        # train_x.shape[1]+1 表示输入数据中每个样本的特征数量（列数）加上一个偏置项的数量
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1, N1)-1

        # FeatureOfEachWindow 可以看作是一个包含了所有样本在特征映射层中的输出特征向量的矩阵
        # 这些特征向量可以用于进一步训练分类器或者回归器等模型，以完成具体的任务。
        # 每个特征映射数据 = 特征偏置*权重
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)

        # 对FeatureOfEachWindow中的每个特征进行归一化，数据存储在FeatureOfEachWindowAfterPreprocess中
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)

        # 将稀疏编码系数存在Beta1OfEachWindow中
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)

        # 带偏置的矩阵数据被稀疏处理后存入outputOfEachWindow
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)

        # print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))

        # np.max(outputOfEachWindow, axis=0) 和np.min(outputOfEachWindow, axis=0) 分别计算矩阵 outputOfEachWindow 中每列的最大值和最小值
        # distOfMaxAndMin存储每个特征中最大值减最小值
        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))

        # 利用最大值最小值对outputOfEachWindow归一化
        minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]

        # outputOfEachWindow被赋值给OutputOfFeatureMappingLayer的对应位置
        OutputOfFeatureMappingLayer[:, N1*i:N1*(i+1)] = outputOfEachWindow

        del outputOfEachWindow
        del FeatureOfEachWindow 
        del weightOfEachWindow

    # 3. 生成增强层
    # 在特征映射层加偏置形成增强层带偏置
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])

    # 保证 weightOfEnhanceLayer的列空间包含输入数据的列空间
    # 正交化随机生成增强结点的权重
    if N1*N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1, N3))-1
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1, N3).T-1).T

    # tempOfOutputOfEnhanceLayer存储增强层输出临时文件夹
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    # print('Enhance nodes: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))

    # 计算收敛参数parameterOfShrink：避免过拟合
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)

    # OutputOfEnhanceLayer存储增强层输出
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

    # 4. 生成最终结果
    # 生成最终输入InputOfOutputLayer为OutputOfFeatureMappingLayer+OutputOfEnhanceLayer拼接而成
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])

    # 求InputOfOutputLayer的伪逆矩阵pinvOfInput
    # 伪逆矩阵的作用是在线性模型中用于计算最小二乘解，即用于求解模型中的权重参数
    pinvOfInput = pinv(InputOfOutputLayer, c)

    # 求输出权重OutputWeight，用于模型的预测计算，train_y 是训练数据的目标值矩阵
    OutputWeight = np.dot(pinvOfInput, train_y)

    # 计时结束，得出训练时间
    time_end = time.time()
    trainTime = time_end - time_start

    # 存储训练结果
    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)

    # 5. 模型评估
    # 得出训练准确度和训练时间
    trainAcc = show_accuracy(OutputOfTrain, train_y)
    print('Training accurate is', trainAcc*100, '%')
    print('Training time is ', trainTime, 's')
    train_acc_all[0][0] = trainAcc
    train_time[0][0] = trainTime

    """
    测试过程
    """
    # 1. 数据预处理
    test_x = preprocessing.scale(test_x, axis=1)

    # 2. 生成特征映射层
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2*N1])

    # 在测试过程中控制特征映射层大小
    ymin = 0
    ymax = 1

    time_start = time.time()  # 测试时间开始

    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        # OutputOfFeatureMappingLayerTest中存储的数据在ymin和ymax之间
        OutputOfFeatureMappingLayerTest[:, N1*i:N1*(i+1)] = (ymax-ymin)*(outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i]-ymin

    # 3. 生成增强层
    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)

    print(parameterOfShrink)
    # 4. 生成最终输出
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    print(OutputOfTest)
    time_end = time.time()  # 测试时间结束
    testTime = time_end - time_start

    # 5.模型评估
    testAcc = show_accuracy(OutputOfTest, test_y)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime

    return test_acc, test_time, train_acc_all, train_time


'''
增加强化层节点版---BLS

参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内特征节点数
N2-----映射层窗口数
N3-----强化层节点数
l------步数
M------步长
'''


def BLS_AddEnhanceNodes(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, L, M):
    """
    训练过程    两个参数最重要，1）y;2)Beta1OfEachWindow
    """

    # 1. 数据预处理
    train_x = preprocessing.scale(train_x, axis=1)

    # 2. 生成特征映射层
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])

    # 定义几种容器
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2*N1])
    distOfMaxAndMin = []
    minOfEachWindow = []
    Beta1OfEachWindow = []

    train_acc = np.zeros([1, L+1])
    test_acc = np.zeros([1, L+1])

    train_time = np.zeros([1, L+1])
    test_time = np.zeros([1, L+1])
    time_start = time.time()  # 计时开始

    u = 0  # 保证每次循环生成序列的数都是不一样的

    for i in range(N2):
        random.seed(i+u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1, N1)-1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1*i:N1*(i+1)] = outputOfEachWindow
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 
        
    # 3. 生成增强层
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
    if N1*N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1, N3)-1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1, N3).T-1).T
    
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
    # 4. 生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer, c)
    OutputWeight = pinvOfInput.dot(train_y) 

    time_end = time.time()
    trainTime = time_end - time_start

    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain, train_y)
    print('Training accurate is', trainAcc*100, '%')
    print('Training time is ', trainTime, 's')
    train_acc[0][0] = trainAcc
    train_time[0][0] = trainTime

    """
    测试过程
    """
    test_x = preprocessing.scale(test_x, axis=1) 

    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2*N1])
    time_start = time.time()
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:, N1*i:N1*(i+1)] = (outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i]

    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)

    time_end = time.time()  # 测试完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest, test_y)
    print('Testing accurate is', testAcc*100, '%')
    print('Testing time is ', testTime, 's')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime

    '''
    增量增加强化节点
    '''
    parameterOfShrinkAdd = []  # 存储增量增加强化节点过程中每个阶段的缩放参数

    # L 是增加强化节点的次数，循环语句会执行 L 次，每次增加一个强化节点
    # list(range(L))每次循环会生成e值来表示当前元素的值
    for e in list(range(L)):
        time_start = time.time()
        if N1*N2 >= M:  # 首先判断当前强化层的节点数是否大于等于预设的节点数M
            random.seed(e)
            # 如果是，则生成一个大小为 (N2*N1+1) * M 的随机权重矩阵 weightOfEnhanceLayerAdd，
            # 然后对该矩阵进行正交化处理，使其列向量单位正交，以便提高模型的预测能力
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2*N1+1, M)-1)
        else:
            random.seed(e)
            # 如果当前强化层的节点数小于预设的节点数 M，则生成一个大小为 M * (N2*N1+1) 的随机权重矩阵 weightOfEnhanceLayerAdd，
            # 然后对该矩阵进行转置和正交化处理，得到一个大小为 (N2*N1+1) * M 的权重矩阵，同样使其列向量单位正交
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2*N1+1, M).T-1).T
        tempOfOutputOfEnhanceLayerAdd = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayerAdd)
        parameterOfShrinkAdd.append(s/np.max(tempOfOutputOfEnhanceLayerAdd))
        OutputOfEnhanceLayerAdd = tansig(tempOfOutputOfEnhanceLayerAdd*parameterOfShrinkAdd[e])
        tempOfLastLayerInput = np.hstack([InputOfOutputLayer, OutputOfEnhanceLayerAdd])

        # 伪逆算法 动态逐步算法的一部分
        D = pinvOfInput.dot(OutputOfEnhanceLayerAdd)
        C = OutputOfEnhanceLayerAdd - InputOfOutputLayer.dot(D)
        if C.all() == 0:
            w = D.shape[1]
            B = np.mat(np.eye(w) - np.dot(D.T, D)).I.dot(np.dot(D.T, pinvOfInput))
        else:
            B = pinv(C, c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)), B])

        OutputWeightEnd = pinvOfInput.dot(train_y)
        InputOfOutputLayer = tempOfLastLayerInput

        Training_time = time.time() - time_start
        train_time[0][e+1] = Training_time

        OutputOfTrain1 = InputOfOutputLayer.dot(OutputWeightEnd)
        TrainingAccuracy = show_accuracy(OutputOfTrain1, train_y)
        train_acc[0][e+1] = TrainingAccuracy
        print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %')

        # 测试
        time_start = time.time()
        OutputOfEnhanceLayerAddTest = tansig(InputOfEnhanceLayerWithBiasTest.dot(weightOfEnhanceLayerAdd) * parameterOfShrinkAdd[e])
        InputOfOutputLayerTest = np.hstack([InputOfOutputLayerTest, OutputOfEnhanceLayerAddTest])

        OutputOfTest1 = InputOfOutputLayerTest.dot(OutputWeightEnd)
        TestingAcc = show_accuracy(OutputOfTest1, test_y)
        
        Test_time = time.time() - time_start
        test_time[0][e+1] = Test_time
        test_acc[0][e+1] = TestingAcc
        print('Incremental Testing Accuracy is : ', TestingAcc * 100, ' %')
        
    return test_acc, test_time, train_acc, train_time


'''
增加强化层节点版---BLS

参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内节点数
N2-----映射层窗口数
N3-----强化层节点数
L------步数

M1-----增加映射节点数
M2-----与增加映射节点对应的强化节点数
M3-----新增加的强化节点
'''


def BLS_AddFeatureEnhanceNodes(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, L, M1, M2, M3):
    """
    训练过程
    """
    train_x = preprocessing.scale(train_x, axis=1)

    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2*N1])
    Beta1OfEachWindow = list()
    distOfMaxAndMin = []
    minOfEachWindow = []

    train_acc = np.zeros([1, L+1])
    test_acc = np.zeros([1, L+1])
    train_time = np.zeros([1, L+1])
    test_time = np.zeros([1, L+1])
    time_start = time.time()#计时开始

    u = 0
    for i in range(N2):
        random.seed(i+u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1, N1)-1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.mean(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1*i:N1*(i+1)] = outputOfEachWindow
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow
 
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
    if N1*N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1, N3)-1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1, N3).T-1).T
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

    InputOfOutputLayerTrain = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayerTrain, c)
    OutputWeight =pinvOfInput.dot(train_y)  # 全局违逆
    time_end = time.time()  # 训练完成
    trainTime = time_end - time_start
    OutputOfTrain = np.dot(InputOfOutputLayerTrain, OutputWeight)

    trainAcc = show_accuracy(OutputOfTrain, train_y)
    print('Training accurate is', trainAcc*100, '%')
    print('Training time is ', trainTime, 's')
    train_acc[0][0] = trainAcc
    train_time[0][0] = trainTime

    """
       测试过程
    """
    test_x = preprocessing.scale(test_x, axis=1)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2*N1])
    time_start = time.time()
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:, N1*i:N1*(i+1)] = (outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i]
    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    time_end = time.time()
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest, test_y)
    print('Testing accurate is', testAcc*100, '%')
    print('Testing time is ', testTime, 's')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime

    '''
    增加特征映射 和 强化节点
    '''
    WeightOfNewFeature2 = list()
    WeightOfNewFeature3 = list()
    for e in list(range(L)):  # 每次增加一个节点
        time_start = time.time()
        random.seed(e+N2+u)  # 保证每次随机数不一样

        # 生成新的特征映射层
        weightOfNewMapping = 2 * random.random([train_x.shape[1]+1, M1]) - 1  # 随机生成新的权重
        NewMappingOutput = FeatureOfInputDataWithBias.dot(weightOfNewMapping)

        scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(NewMappingOutput)
        FeatureOfEachWindowAfterPreprocess = scaler2.transform(NewMappingOutput)

        betaOfNewWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfNewWindow)
   
        TempOfFeatureOutput = FeatureOfInputDataWithBias.dot(betaOfNewWindow)

        distOfMaxAndMin.append(np.max(TempOfFeatureOutput, axis=0) - np.min(TempOfFeatureOutput, axis=0))
        minOfEachWindow.append(np.mean(TempOfFeatureOutput, axis=0))
        outputOfNewWindow = (TempOfFeatureOutput-minOfEachWindow[N2+e])/distOfMaxAndMin[N2+e]

        OutputOfFeatureMappingLayer = np.hstack([OutputOfFeatureMappingLayer, outputOfNewWindow])

        # 更新增强节点
        NewInputOfEnhanceLayerWithBias = np.hstack([outputOfNewWindow, 0.1 * np.ones((outputOfNewWindow.shape[0], 1))])

        if M1 >= M2:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2*random.random([M1+1, M2])-1)
        else:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2*random.random([M1+1, M2]).T-1).T
        WeightOfNewFeature2.append(RelateEnhanceWeightOfNewFeatureNodes)

        tempOfNewFeatureEhanceNodes = NewInputOfEnhanceLayerWithBias.dot(RelateEnhanceWeightOfNewFeatureNodes)
        
        parameter1 = s/np.max(tempOfNewFeatureEhanceNodes)

        outputOfNewFeatureEhanceNodes = tansig(tempOfNewFeatureEhanceNodes * parameter1)

        # 增加增强节点
        if N2*N1+e*M1 >= M3:
            random.seed(67797325+e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2*N1+(e+1)*M1+1, M3) - 1)
        else:
            random.seed(67797325+e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2*N1+(e+1)*M1+1, M3).T-1).T
        WeightOfNewFeature3.append(weightOfNewEnhanceNodes)

        InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])

        tempOfNewEnhanceNodes = InputOfEnhanceLayerWithBias.dot(weightOfNewEnhanceNodes)
        parameter2 = s/np.max(tempOfNewEnhanceNodes)
        OutputOfNewEnhanceNodes = tansig(tempOfNewEnhanceNodes * parameter2)

        OutputOfTotalNewAddNodes = np.hstack([outputOfNewWindow, outputOfNewFeatureEhanceNodes, OutputOfNewEnhanceNodes])
        tempOfInputOfLastLayers = np.hstack([InputOfOutputLayerTrain, OutputOfTotalNewAddNodes])

        D = pinvOfInput.dot(OutputOfTotalNewAddNodes)
        C = OutputOfTotalNewAddNodes - InputOfOutputLayerTrain.dot(D)
        if C.all() == 0:
            w = D.shape[1]
            B = (np.eye(w) - D.T.dot(D)).I.dot(D.T.dot(pinvOfInput))
        else:
            B = pinv(C, c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)), B])
        OutputWeight = pinvOfInput.dot(train_y)        
        InputOfOutputLayerTrain = tempOfInputOfLastLayers
        
        time_end = time.time()
        Train_time = time_end - time_start
        train_time[0][e+1] = Train_time

        predictLabel = InputOfOutputLayerTrain.dot(OutputWeight)
        TrainingAccuracy = show_accuracy(predictLabel, train_y)
        train_acc[0][e+1] = TrainingAccuracy
        print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %')
        
        # 测试过程
        time_start = time.time() 
        WeightOfNewMapping = Beta1OfEachWindow[N2+e]
        outputOfNewWindowTest = FeatureOfInputDataWithBiasTest.dot(WeightOfNewMapping)
        outputOfNewWindowTest = (outputOfNewWindowTest-minOfEachWindow[N2+e])/distOfMaxAndMin[N2+e]
        OutputOfFeatureMappingLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, outputOfNewWindowTest])
        InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1*np.ones([OutputOfFeatureMappingLayerTest.shape[0], 1])])
        NewInputOfEnhanceLayerWithBiasTest = np.hstack([outputOfNewWindowTest, 0.1*np.ones([outputOfNewWindowTest.shape[0], 1])])
        weightOfRelateNewEnhanceNodes = WeightOfNewFeature2[e]
        OutputOfRelateEnhanceNodes = tansig(NewInputOfEnhanceLayerWithBiasTest.dot(weightOfRelateNewEnhanceNodes) * parameter1)
        weightOfNewEnhanceNodes = WeightOfNewFeature3[e]
        OutputOfNewEnhanceNodes = tansig(InputOfEnhanceLayerWithBiasTest.dot(weightOfNewEnhanceNodes)*parameter2)
        InputOfOutputLayerTest = np.hstack([InputOfOutputLayerTest, outputOfNewWindowTest, OutputOfRelateEnhanceNodes,OutputOfNewEnhanceNodes])
        predictLabel = InputOfOutputLayerTest.dot(OutputWeight)
        TestingAccuracy = show_accuracy(predictLabel, test_y)
        time_end = time.time()
        Testing_time = time_end - time_start
        test_time[0][e+1] = Testing_time
        test_acc[0][e+1] = TestingAccuracy
        print('Testing Accuracy is : ', TestingAccuracy * 100, ' %')

    return test_acc, test_time, train_acc, train_time