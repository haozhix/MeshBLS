
from code_shrec_16.BLS_functions import one_hot_m
from code_shrec_16.GetShrecdata import get_dataset, getFeatures1, getFeatures2, getFeatures3
from code_shrec_16.MeshBLS import MeshBLS

# 数据载入
dataset_dir = 'shrec_16'  # 读取数据集
train_data_array, train_labels_array, test_data_array, test_labels_array = get_dataset(dataset_dir)
print(train_data_array.shape)

centerTrainData = train_data_array[:, :, 0:3]
normalTrainData = train_data_array[:, :, 3:6]
curvsTrainData = train_data_array[:, :, 6:9]
angle1TrainData = train_data_array[:, :, 9:12]
angle2TrainData = train_data_array[:, :, 12:18]
areaTrainData = train_data_array[:, :, 18:19]

centerTestData = test_data_array[:, :, 0:3]
normalTestData = test_data_array[:, :, 3:6]
curvsTestData = test_data_array[:, :, 6:9]
angle1TestData = test_data_array[:, :, 9:12]
angle2TestData = test_data_array[:, :, 12:18]
areaTestData = test_data_array[:, :, 18:19]

trainlabel = one_hot_m(train_labels_array, 30)
testlabel = one_hot_m(test_labels_array, 30)

print('================extract the feas =======================')
centerTrain, centerTest = getFeatures1(centerTrainData, centerTestData)
normalTrain, normalTest = getFeatures1(normalTrainData, normalTestData)
curvsTrain, curvsTest = getFeatures1(curvsTrainData, curvsTestData)
angle1Train, angle1Test = getFeatures1(angle1TrainData, angle1TestData)
angle2Train, angle2Test = getFeatures2(angle2TrainData, angle2TestData)
areaTrain, areaTest = getFeatures3(areaTrainData, areaTestData)

print('================run meshbls=======================')
MeshBLS(centerTrain, centerTest,
        normalTrain, normalTest,
        curvsTrain, curvsTest,
        angle1Train, angle1Test,
        angle2Train, angle2Test,
        areaTrain, areaTest,
        trainlabel, testlabel)
