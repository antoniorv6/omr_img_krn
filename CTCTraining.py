from utils.utils import loadImages, loadDataY, DATA_TYPE, check_and_retrieveVocabulary, data_preparation_CTC, getCTCValidationData, get_statistic_data, load_model_fromfile, loadVocabulary, getCTCTestData, save_image_asResult
from model.ctc_keras import get_model
from sklearn.utils import shuffle
import numpy as np
import cv2
import sys


vocabularyNames = ["paec","agnostic", "skern", "kern"]
outputNames = [".pae",".agnostic", ".krn", ".skm"]

def CTCTraining(output, data_path):
    fixed_height = 64

    print("Training CTC with - " + str(DATA_TYPE(output)))

    print("Loading training data...")
    XTrain = loadImages(data_path, data_path + "/train.lst", 100)
    YTrain = loadDataY(data_path, data_path + "/train.lst", output, 100)
    print(XTrain.shape)
    print(YTrain.shape)

    print("Loading validation data...")
    XValidate = loadImages(data_path, data_path + "/validation.lst", 100)
    YValidate = loadDataY(data_path, data_path + "/validation.lst", output, 100)
    print(XValidate.shape)
    print(YValidate.shape)

    XTrain, YTrain = shuffle(XTrain, YTrain)
    XValidate, YValidate = shuffle(XValidate, YValidate)

    w2i, i2w = check_and_retrieveVocabulary([YTrain, YValidate], "./vocabulary", vocabularyNames[output-1])

    print("Vocabulary size: " + str(len(w2i)))

    for i in range(min(len(XTrain), len(YTrain))):
        img = (255. - XTrain[i]) / 255.
        width = int(float(fixed_height * img.shape[1]) / img.shape[0])
        XTrain[i] = cv2.resize(img, (width, fixed_height))
        for idx, symbol in enumerate(YTrain[i]):
            YTrain[i][idx] = w2i[symbol]
    
    for i in range(min(len(XValidate), len(YValidate))):
        img = (255. - XValidate[i]) / 255.
        width = int(float(fixed_height * img.shape[1]) / img.shape[0])
        XValidate[i] = cv2.resize(img, (width, fixed_height))
        for idx, symbol in enumerate(YValidate[i]):
            YValidate[i][idx] = w2i[symbol]

    vocabulary_size = len(w2i)
    model_tr, model_pr = get_model(input_shape=(fixed_height,None,1),vocabulary_size=vocabulary_size)

    X_train, Y_train, L_train, T_train = data_preparation_CTC(XTrain, YTrain, fixed_height)

    print('Training with ' + str(X_train.shape[0]) + ' samples.')
    
    inputs = {'the_input': X_train,
                 'the_labels': Y_train,
                 'input_length': L_train,
                 'label_length': T_train,
                 }
    
    outputs = {'ctc': np.zeros([len(X_train)])}
    best_cer = 10000

    for super_epoch in range(50):
       model_tr.fit(inputs,outputs, batch_size = 16, epochs = 5, verbose = 2)
       ser, cer = getCTCValidationData(model_pr, XValidate, YValidate, i2w)
       if cer < best_cer:
           best_cer = cer
           model_pr.save("model/checkpoints/" + vocabularyNames[output-1] + "_model.h5")
           print('CER Improved -> Saving model to {}'.format("model/checkpoints/" + vocabularyNames[output-1] + "_model.h5"))

def CTCTest(output, data_path):
    fixed_height = 64
    print("Testing CTC with - " + str(DATA_TYPE(output)))
    print("Loading test data...")
    XTest = loadImages(data_path, data_path + "/test.lst", 100)
    YTest = loadDataY(data_path, data_path + "/test.lst", output, 100)
    print(XTest.shape)
    print(YTest.shape)

    #XTest, YTest = shuffle(XTest, YTest)

    for i, image in enumerate(XTest):
        save_image_asResult(image, str(i))
    
    for i in range(len(XTest)):
        img = (255. - XTest[i]) / 255.
        width = int(float(fixed_height * img.shape[1]) / img.shape[0])
        XTest[i] = cv2.resize(img, (width, fixed_height))

    i2w = loadVocabulary("./vocabulary/"+ vocabularyNames[output-1]+"i2w.npy")
    modelToTest = load_model_fromfile(vocabularyNames[output-1])

    ser, cer = getCTCTestData(modelToTest, XTest, YTest, i2w, outputNames[output-1])

    #PAEC
    #print("Testing CTC with - " + str(DATA_TYPE(1)))
    #YTest = loadDataY(data_path, data_path + "/test.lst", 1, 100)
    #i2w = loadVocabulary("./vocabulary/"+ vocabularyNames[0]+"i2w.npy")
    #modelToTest = load_model_fromfile(vocabularyNames[0])
    #ser, cer = getCTCTestData(modelToTest, XTest, YTest, i2w, outputNames[0])


def ShowStatsofCodification(data_path, outputName, outputType):
    wordsNumber = 0
    lengthList  = []
    maxElementLen = 0
    minElementLen = 1000000
    YTrain = loadDataY(data_path, data_path + "/train.lst", outputType, 100)
    YVal = loadDataY(data_path, data_path + "/validation.lst", outputType, 100)
    #YTest = loadDataY(data_path, data_path + "/test.lst", output, 100)
    w2i, i2w = check_and_retrieveVocabulary([YTrain, YVal], "./vocabulary", outputName)
    for element in YTrain:
        lengthList.append(len(element))
        wordsNumber+= len(element)
        if maxElementLen < len(element):
            maxElementLen = len(element)
        if minElementLen > len(element):
            minElementLen = len(element)

    for element in YVal:
        lengthList.append(len(element))
        wordsNumber+= len(element)
        if maxElementLen < len(element):
            maxElementLen = len(element)
        if minElementLen > len(element):
            minElementLen = len(element)
    
    median, mode = get_statistic_data(lengthList)

    print("Running words: " + str(wordsNumber))
    print("Vocabulary size: " + str(len(w2i)))
    print("Sequence elements median: " + str(median))
    print("Sequence elements mode: " + str(mode))
    print("Longest sequence: " + str(maxElementLen))
    print("Shortest sequence: " + str(minElementLen))
    print()
    print("##################")
    print()

def RetrieveStats(output,data_path):
    print("CODIFICATION STATS")
    print()
    print("##### **KERN #####")
    ShowStatsofCodification(data_path, vocabularyNames[1], 2)
    print("##### **SKM #####")
    ShowStatsofCodification(data_path, vocabularyNames[2], 3)
    print("##### PAEC #####")
    ShowStatsofCodification(data_path, vocabularyNames[0], 1)


def RetrieveImageStats(data_path):
    images = loadImages(data_path, data_path + "/test.lst", 100)

    for i in range(len(images)):
        img = (255. - images[i]) / 255.
        width = int(float(64 * img.shape[1]) / img.shape[0])
        images[i] = cv2.resize(img, (width, 64))
    
    widthList = []
    maxElementLen = 0
    minElementLen = 100000
    for img in images:
        width = img.shape[1]
        widthList.append(width)
        if maxElementLen < width:
            maxElementLen = width
        if minElementLen > width:
            minElementLen = width
    
    median, mode = get_statistic_data(widthList)

    print("Frames median: " + str(median))
    print("Frames mode: " + str(mode))
    print("Widest image: " + str(maxElementLen))
    print("Narrowest image: " + str(minElementLen))




