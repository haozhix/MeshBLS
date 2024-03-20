import numpy as np
import scipy.io as scio
from BroadLearningSystem import BLS, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes
from Demo4BLS import testdata, testlabel, trainlabel, traindata


teA = list() #Testing ACC
tet = list() #Testing Time
trA = list() #Training ACC
trt = list() #Training Time
t0 = 0
t2 =[]
t1 = 0
tt1 = 0
tt2 = 0
tt3 = 0

# BLS parameters
s = 0.8  #reduce coefficient
C = 2**(-30) #Regularization coefficient
N1 = 22  #Nodes for each feature mapping layer window
N2 = 20  #Windows for feature mapping layer
N3 = 540 #Enhancement layer nodes


#  bls-网格搜索 找N1, N2, N3
for N1 in range(8, 25, 2):
    r1 = len(range(8, 25, 2))
    for N2 in range(10, 21, 2):
        r2 = len(range(10, 21, 2))
        for N3 in range(600, 701, 10):
            r3 = len(range(600, 701, 10))
            a, b, c, d = BLS(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)
            t0 += 1
            if a > t1:
                tt1 = N1
                tt2 = N2
                tt3 = N3
                t1 = a
            teA.append(a)
            tet.append(b)
            trA.append(c)
            trt.append(d)
            print('当前进度:', round(t0/(r1*r2*r3)*100, 4), '%', 'The best result:', t1, 'N1:', tt1, 'N2:', tt2, 'N3:', tt3)
meanTeACC = np.mean(teA)
meanTrTime = np.mean(trt)
maxTeACC = np.max(teA)


# # BLS随机种子搜索
# teA = list() #Testing ACC
# tet = list() #Testing Time
# trA = list() #Training ACC
# trt = list() #Train Time
# t0 = 0
# t = 0
# t2 =[]
# t1 = 0
# tt1 = 0
# tt2 = 0
# tt3 = 0
#
# ## BLS parameters
# s = 0.8 #reduce coefficient
# C = 2**(-30) #Regularization coefficient
#
# #N1 = 10  #Nodes for each feature mapping layer window
# #N2 = 10  #Windows for feature mapping layer
# #N3 = 500 #Enhancement layer nodes
#
# u = 45
# i = 0
# L = 5
# M = 50
# for N1 in range(10, 21, 20):
#     r1 = len(range(10, 21, 20))
#     for N2 in range(12, 21, 10):
#         r2 = len(range(12, 21, 10))
#         for N3 in range(4000, 4001, 500):
#             r3 = len(range(4000, 4001, 500))
#             for i in range(-28, -27, 2):
#                 r4 = len(range(-28, -27, 2))
#                 C = 2**(i)
# #               a, b, c, d = BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,u)
#                 a, b, c, d = BLS_AddEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M)
#                 t0 += 1
#             if a > t1:
#                 tt1 = N1
#                 tt2 = N2
#                 tt3 = N3
#                 t1 = a
#                 t = u
#                 i1 = i
#                 tet.append(b)
#                 teA.append(a)
#                 trA.append(c)
#                 trt.append(d)
#                 print('NO.', t0, 'total:', r1*r2*r3, 'ACC:', a*100, 'Pars:', N1, ',', N2, ',', N3, 'C', i)
#                 print('The best so far:', t1*100, 'N1:', tt1, 'N2:', tt2, 'N3:', tt3, 'C:', i1)


# 网格搜索查找c u
teA = list()
tet = list()
trA = list()
trt = list()
s = 0.8
#C = 2**(-30)
N1 = 10
N2 = 100
N3 = 8000
L = 5
M1 = 20
M2 = 20
M3 = 50
t0 = 0
t1 = 0
t2 = 0
for i in range(-30,-21,5):
    r1 = len(range(-30,-21,5))
    for u in range(10,50,1):
        r2 = len(range(10,50,1))
        C = 2**i
        t0 += 1
#    a,b,c,d = BLS_AddEnhanceNodes(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,L,M1)
        a,b,c,d = BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,u)
        teA.append(a)
        tet.append(b)
        trA.append(c)
        trt.append(d)
        if a > t1:
            t1 = a
            t2 = i
            t = u
        print(t0,'当前进度为:',round(t0/(r1*r2)*100,4),'%','The best result:', t1,'C',t2,'u:',t)



