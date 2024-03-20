import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
import time
from BLS_Preprocessing import show_accuracy, sparse_bls, pinv, tansig


def CFBLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3):
    """
    CFBLS
    """
    """
    训练过程
    """
    # 1. 对训练数据进行标准化处理
    L = 0  # 初始步数为0
    train_x = preprocessing.scale(train_x, axis=1)  # 对train_x的每个样本标准化处理

    # 2. 生成特征映射层
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])  # 加偏置
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])  # 特征映射层输出
    Beta1OfEachWindow = []  # 存储了特征映射层中每个窗口的稀疏编码系数
    distOfMaxAndMin = []  # 存储每个特征的最大值和最小值之间的距离
    minOfEachWindow = []  # 列表的长度等于特征映射层中窗口的数量，每个元素都是一个标量，用于存储对应窗口的最小值。

    # 存储每个训练周期的训练集和测试集准确率
    train_acc_all = np.zeros([1, L + 1])  # 定义train_acc_all 是一个长度为 L+1 的一维数组
    test_acc = np.zeros([1, L + 1])  # test_acc 是一个长度为 L+1 的一维数组

    train_time = np.zeros([1, L + 1])  # train_time 是一个长度为 L+1 的一维数组
    test_time = np.zeros([1, L + 1])  # test_time 是一个长度为 L+1 的一维数组
    time_start = time.time()  # 计时开始
    weightOfEachWindowOfCascades_list = []
    for i in range(N2):
        random.seed(i)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, N1) - 1   # 定义每个窗口权重
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)  # 归一化
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)  # 稀疏化

        # 特征映射层级联
        weightOfEachWindowOfCascades = LA.orth(2.0 * np.random.rand(N1, N1) - 1.0)
        weightOfEachWindowOfCascades_list.append(weightOfEachWindowOfCascades)
        if i == 0:
            outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
            tempOfoutputOfEachWindow = outputOfEachWindow
        else:
            tempOfoutputOfEachWindow = np.dot(tempOfoutputOfEachWindow, weightOfEachWindowOfCascades)
            outputOfEachWindow = tempOfoutputOfEachWindow

        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i]) / distOfMaxAndMin[i]  # 利用最大值最小值归一化
        OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow

        del outputOfEachWindow
        del FeatureOfEachWindow
        del weightOfEachWindow

    # 3. 生成增强层
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
    if N1 * N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3)) - 1
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

    # 4. 生成最终结果
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer, c)
    OutputWeight = np.dot(pinvOfInput, train_y)

    # 计时结束，得出训练时间
    time_end = time.time()
    trainTime = time_end - time_start

    # 存储训练结果
    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)

    # 5. 模型评估
    # 得出训练准确度和训练时间
    trainAcc = show_accuracy(OutputOfTrain, train_y)
    print('Training accurate is', trainAcc * 100, '%')
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
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])

    # 在测试过程中控制特征映射层大小
    ymin = 0
    ymax = 1

    time_start = time.time()  # 测试时间开始

    for i in range(N2):
        betaOfEachWindow = Beta1OfEachWindow[i]
        # 级联
        weightOfEachWindowOfCascades = weightOfEachWindowOfCascades_list[i]
        if i == 0:
            outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, betaOfEachWindow)
            tempOfoutputOfEachWindowTest = outputOfEachWindowTest
        else:
            tempOfoutputOfEachWindowTest = np.dot(tempOfoutputOfEachWindowTest, weightOfEachWindowOfCascades)
            outputOfEachWindowTest = tempOfoutputOfEachWindowTest
        # OutputOfFeatureMappingLayerTest中存储的数据在ymin和ymax之间
        OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (ymax - ymin) * (outputOfEachWindowTest - minOfEachWindow[i]) / distOfMaxAndMin[i] - ymin

    # 3. 生成增强层
    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)

    # 4. 生成最终输出
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)

    time_end = time.time()  # 测试时间结束
    testTime = time_end - time_start

    # 5.模型评估
    testAcc = show_accuracy(OutputOfTest, test_y)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime

    return test_acc, test_time, train_acc_all, train_time
