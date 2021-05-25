from mnist import MNIST
import numpy as nump
import sys

nump.set_printoptions(threshold=sys.maxsize)
nump.set_printoptions(precision=20)


data = MNIST("MNIST/")

imgs, lbls = data.load_training()
test_imgs, test_lbls = data.load_testing()

numList = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
for i in range(10):
    numList[i] = nump.array([imgs[key] for (key, label) in enumerate(lbls) if int(label) == i])
fourSevenEight = nump.array([imgs[key] for (key, label) in enumerate(lbls) if int(label) == 4 or int(label) == 7 or int(label) == 8])
fourSevenEightLbl = nump.array([lbls[key] for (key, label) in enumerate(lbls) if int(label) == 4 or int(label) == 7 or int(label) == 8])

numTestList = ["tZero", "tOne", "tTwo", "tThree", "tFour", "tFive", "tSix", "tSeven", "tEight", "tEine"]
for j in range(10):
    numTestList[j] = nump.array([test_imgs[key] for (key, label) in enumerate(test_lbls) if int(label) == j])

allData = nump.array([imgs[key] for (key, label) in enumerate(lbls)])
allLbl = nump.array([lbls[key] for (key, label) in enumerate(lbls)])


zeroOneTrain = nump.array([imgs[key] for (key, label) in enumerate(lbls) if int(label) == 0 or int(label) == 1])
zeroOneTrainLbl = nump.array([lbls[key] for (key, label) in enumerate(lbls) if int(label) == 0 or int(label) == 1])

zeroOneTest = nump.array([test_imgs[key] for (key, label) in enumerate(test_lbls) if int(label) == 0 or int(label) == 1])
zeroOneTestLbl = nump.array([test_lbls[key] for (key, label) in enumerate(test_lbls) if int(label) == 0 or int(label) == 1])

fourSevenEightTestLbl = nump.array([test_lbls[key] for (key, label) in enumerate(test_lbls) if int(label) == 4 or int(label) == 7 or
                                    int(label) == 8])
fourSevenEightTest = nump.array([test_imgs[key] for (key, label) in enumerate(test_lbls) if int(label) == 4 or int(label) == 7 or
                                 int(label) == 8])
zeroTwoTrain = nump.array([imgs[key] for (key, label) in enumerate(lbls) if int(label) == 0 or int(label) == 2])
zeroTwoTrainLbl = nump.array([lbls[key] for (key, label) in enumerate(lbls) if int(label) == 0 or int(label) == 2])
zeroTwoTest = nump.array([test_imgs[key] for (key, label) in enumerate(test_lbls) if int(label) == 0 or int(label) == 2])
oneTwoTrain = nump.array([imgs[key] for (key, label) in enumerate(lbls) if int(label) == 1 or int(label) == 2])

oneTwoTrainLbl = nump.array([lbls[key] for (key, label) in enumerate(lbls) if int(label) == 1 or int(label) == 2])

oneTwoTest = nump.array([test_imgs[key] for (key, label) in enumerate(test_lbls) if int(label) == 1 or int(label) == 2])

oneTwoTestLbl = nump.array([test_lbls[key] for (key, label) in enumerate(test_lbls) if int(label) == 1 or int(label) == 2])
zeroTwoTestLbl = nump.array([test_lbls[key] for (key, label) in enumerate(test_lbls) if int(label) == 0 or int(label) == 2])
oneTwoThreeTrain = nump.array([imgs[key] for (key, label) in enumerate(lbls) if int(label) == 1 or int(label) == 2 or int(label) == 3])

oneTwoThreeTrainLbl = nump.array([lbls[key] for (key, label) in enumerate(lbls) if int(label) == 1 or int(label) == 2 or int(label) == 3])

oneTwoThreeTest = nump.array([test_imgs[key] for (key, label) in enumerate(test_lbls) if int(label) == 1 or int(label) == 2 or int(label) == 3])

oneTwoThreeTestLbl = nump.array([test_lbls[key] for (key, label) in enumerate(test_lbls) if int(label) == 1 or int(label) == 2 or int(label) == 3])

numFeature = 784
img = nump.array(imgs)
img_length = int(len(img))
img = img[0:img_length, 0:numFeature]
img = img / 2550
lbl = nump.array(lbls)
test_img = nump.array(test_imgs)
test_img = test_img[:, 0:numFeature]
test_img = test_img / 2550
test_lbl = nump.array(test_lbls)
learnR1 = 3.5
learnR2 = 2.5
Hidden_Layer_Nodes = 86
numClass = 10
Weights1 = nump.random.rand(numFeature, Hidden_Layer_Nodes)
Weights2 = nump.random.rand(Hidden_Layer_Nodes, numClass)
Layer0 = nump.ones((numFeature, 1))
Layer1 = nump.ones((Hidden_Layer_Nodes, 1))
Layer2 = nump.ones((numClass, 1))
In1 = nump.random.rand(Hidden_Layer_Nodes, 1)
Out1 = nump.random.rand(Hidden_Layer_Nodes, 1)
In2 = nump.random.rand(numClass, 1)
Out2 = nump.random.rand(numClass, 1)
EPOCH = 9
for e in range(EPOCH):
    print("Fully Connected Neural Network running on EPOCH " + str(e) + "...")
    for i in range(len(img)):
        Error = nump.zeros((numClass, 1))
        derivatives1 = nump.zeros((numFeature, Hidden_Layer_Nodes))
        derivatives2 = nump.zeros((Hidden_Layer_Nodes, numClass))
        # forward
        Layer0 = img[i, :].reshape(numFeature, 1)
        In1 = nump.dot(Layer0.T, Weights1).T
        Out1 = 1 / (1 + nump.exp(-In1))
        In2 = nump.dot(Out1.T, Weights2).T
        Out2 = 1 / (1 + nump.exp(-In2))

        # backward
        for j in range(10):
            if lbl[i] == j:
                Error[j] = Out2[j] - 1
            else:
                Error[j] = Out2[j]
        pList = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"]
        for n in range(10):
            pList[n] = Error[n] * Out2[n] * (1 - Out2[n])
        for d in range(len(derivatives2)):
            for e in range(10):
                derivatives2[d, e] = pList[e] * Out1[d]
        pdf = nump.zeros(Hidden_Layer_Nodes)
        for f in range(Hidden_Layer_Nodes):
            pdf[f] = (pList[0] * Weights2[f][0] + pList[1] * Weights2[f][1] + pList[2] * Weights2[f][2] + pList[3] *
                      Weights2[f][3] + pList[4] * Weights2[f][4] + pList[5] * Weights2[f][5] + pList[6] * Weights2[f][6] +
                      pList[7] * Weights2[f][7] + pList[8] * Weights2[f][8] + pList[9] * Weights2[f][9]) * Out1[f] * (1 - Out1[f])
        for m in range(numFeature):
            derivatives1[m, :] = pdf * img[i][m]

        Weights2 = Weights2 - learnR2 * derivatives2
        Weights1 = Weights1 - learnR1 * derivatives1

    S = 0
    F = 0
    for i in range(len(test_img)):
        Layer0 = test_img[i, :].reshape(numFeature, 1)
        In1 = nump.dot(Layer0.T, Weights1).T
        Out1 = 1 / (1 + nump.exp(-In1))
        In2 = nump.dot(Out1.T, Weights2).T
        Out2 = 1 / (1 + nump.exp(-In2))
        if test_lbl[i] == 0 and Out2[0] == nump.max(Out2):
            S += 1
        elif test_lbl[i] == 1 and Out2[1] == nump.max(Out2):
            S += 1
        elif test_lbl[i] == 2 and Out2[2] == nump.max(Out2):
            S += 1
        elif test_lbl[i] == 3 and Out2[3] == nump.max(Out2):
            S += 1
        elif test_lbl[i] == 4 and Out2[4] == nump.max(Out2):
            S += 1
        elif test_lbl[i] == 5 and Out2[5] == nump.max(Out2):
            S += 1
        elif test_lbl[i] == 6 and Out2[6] == nump.max(Out2):
            S += 1
        elif test_lbl[i] == 7 and Out2[7] == nump.max(Out2):
            S += 1
        elif test_lbl[i] == 8 and Out2[8] == nump.max(Out2):
            S += 1
        elif test_lbl[i] == 9 and Out2[9] == nump.max(Out2):
            S += 1
        else:
            F += 1
    print("EPOCH " + str(e))
    print("Success: " + str(S))
    print("Failure: " + str(F))
    print("Success Rate: " + str(S / (S + F)))
    learnR1 += 1
    learnR2 += 1
