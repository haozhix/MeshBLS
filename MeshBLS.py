from BLS_functions import *
import numpy as np
from numpy import random
from scipy import linalg as LA
import time
center_c = 0.001
center_s = 0.8
center_e = 8000

curvs_c = 0.001
curvs_s = 0.8
curvs_e = 1000

angle1_c = 0.001
angle1_s = 0.8
angle1_e = 1000

angle2_c = 0.001
angle2_s = 0.8
angle2_e = 1000

area_c = 0.001
area_s = 0.8
area_e = 1000
# center_c = 0.001
# center_s = 1
# center_e = 1000
#
# curvs_c = 0.001
# curvs_s = 0.8
# curvs_e = 500
#
# angle1_c = 0.001
# angle1_s = 0.8
# angle1_e = 1000
#
# angle2_c = 0.001
# angle2_s = 1
# angle2_e = 1000
#
# area_c = 0.001
# area_s = 0.8
# area_e = 1000

fusion_c = 1  # Regularization coefficient


def Enlayer(Train, shape, e, s, train_y):

    InOfEnhLayerWithBias = np.hstack([Train, 0.1 * np.ones((Train.shape[0], 1))])
    if shape >= center_e:
        random.seed(67797325)
        weiOfEnhLayer = LA.orth(2 * random.randn(shape + 1, e)) - 1
    else:
        random.seed(67797325)
        weiOfEnhLayer = LA.orth(2 * random.randn(shape + 1, e).T - 1).T
    tempOfOutOfEnhLayer = np.dot(InOfEnhLayerWithBias, weiOfEnhLayer)
    parameterOfShrink = s / np.max(tempOfOutOfEnhLayer)
    OutOfEnhLayer = tansig(tempOfOutOfEnhLayer * parameterOfShrink)

    InputOfCLayer = np.hstack([Train, OutOfEnhLayer])
    pinvOfInputC = pinv(InputOfCLayer, center_c)
    CWeight = np.dot(pinvOfInputC, train_y)
    OutC = np.dot(InputOfCLayer, CWeight)
    return OutC, weiOfEnhLayer, parameterOfShrink, CWeight


def EnlayerTest(Test, weiOfEnhLayer, parameterOfShrink, CWeight):
    InOfEnhLayerWithBiasTest = np.hstack([Test, 0.1 * np.ones((Test.shape[0], 1))])
    tempOfOutOfEnhLayerTest = np.dot(InOfEnhLayerWithBiasTest, weiOfEnhLayer)
    OutOfEnhLayerTest = tansig(tempOfOutOfEnhLayerTest * parameterOfShrink)
    InputOfCLayerTest = np.hstack([Test, OutOfEnhLayerTest])
    OutCTest = np.dot(InputOfCLayerTest, CWeight)
    return OutCTest


def MeshBLS(centerTrain, centerTest, normalTrain, normalTest, curvsTrain, curvsTest, angle1Train, angle1Test,
            angle2Train, angle2Test, areaTrain, areaTest, train_y, test_y):

    center_shape = centerTrain.shape[1]
    triangle_shape = curvsTrain.shape[1]
    angle1_shape = angle1Train.shape[1]
    angle2_shape = angle2Train.shape[1]
    area_shape = areaTrain.shape[1]

    time_start = time.time()

    OutC1, weiOfEnhLayer1, parameterOfShrink1, C1Weight = Enlayer(centerTrain, center_shape, center_e, center_s, train_y)
    OutC3, weiOfEnhLayer3, parameterOfShrink3, C3Weight = Enlayer(curvsTrain, triangle_shape, curvs_e, curvs_s, train_y)
    OutC4, weiOfEnhLayer4, parameterOfShrink4, C4Weight = Enlayer(angle1Train, angle1_shape, angle1_e, angle1_s, train_y)
    OutC5, weiOfEnhLayer5, parameterOfShrink5, C5Weight = Enlayer(angle2Train, angle2_shape, angle2_e, angle2_s, train_y)
    OutC6, weiOfEnhLayer6, parameterOfShrink6, C6Weight = Enlayer(areaTrain, area_shape, area_e, area_s, train_y)

    OutC1_N = softmax(OutC1)
    OutC3_N = softmax(OutC3)
    OutC4_N = softmax(OutC4)
    OutC5_N = softmax(OutC5)
    OutC6_N = softmax(OutC6)

    # 生成最终输入
    InputOfOutputLayer = np.hstack([OutC1_N, OutC3_N, OutC4_N, OutC5_N, OutC6_N])
    pinvOfInput = pinv(InputOfOutputLayer, fusion_c)
    OutputWeight = np.dot(pinvOfInput, train_y)  # 全局违逆
    time_end = time.time()  # 训练完成
    trainTime = time_end - time_start

    # 训练输出
    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain, train_y)
    print('Training accurate is', trainAcc * 100, '%')
    print('Training time is ', trainTime, 's')

    # 测试过程
    time_start = time.time()  # 测试计时开始

    #  强化层
    OutC1Test = EnlayerTest(centerTest, weiOfEnhLayer1, parameterOfShrink1, C1Weight)
    OutC3Test = EnlayerTest(curvsTest, weiOfEnhLayer3, parameterOfShrink3, C3Weight)
    OutC4Test = EnlayerTest(angle1Test, weiOfEnhLayer4, parameterOfShrink4, C4Weight)
    OutC5Test = EnlayerTest(angle2Test, weiOfEnhLayer5, parameterOfShrink5, C5Weight)
    OutC6Test = EnlayerTest(areaTest, weiOfEnhLayer6, parameterOfShrink6, C6Weight)

    OutC1Test_N = softmax(OutC1Test)
    OutC3Test_N = softmax(OutC3Test)
    OutC4Test_N = softmax(OutC4Test)
    OutC5Test_N = softmax(OutC5Test)
    OutC6Test_N = softmax(OutC6Test)

    #  最终层输入
    InputOfOutputLayerTest = np.hstack([OutC1Test_N, OutC3Test_N, OutC4Test_N, OutC5Test_N, OutC6Test_N])

    #  最终测试输出
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    time_end = time.time()  # 训练完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest, test_y)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')
    return trainAcc, testAcc
