import numpy as np
from random import seed
from random import randint
from keras import backend as K
from keras.models import load_model

from .test import test_sequence2

import tensorflow as tf
# ====================
w2ikern = {}
i2wkern = {}
w2iagnostic = {}
i2wagnostic = {}
ALPHABETLENGTHKERN = 0
ALPHABETLENGTHAGNOSTIC = 0
BATCH_SIZE = 8
TRAINEPOCHS = 30
EVAL_EPOCH_STRIDE = 2
# ====================

def edit_distance(a, b):
    n, m = len(a), len(b)

    if n > m:
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def batch_confection(batchX, batchY):
    max_batch_input_len = max([len(sequence) for sequence in batchX])
    max_batch_output_len = max([len(sequence) for sequence in batchY])

    encoder_input = np.zeros((len(batchX), max_batch_input_len), dtype=np.float)
    decoder_input = np.zeros((len(batchY), max_batch_output_len + 1), dtype=np.float)
    decoder_output = np.zeros((len(batchY), max_batch_output_len + 1, ALPHABETLENGTHKERN), dtype=np.float)

    putNoise = randint(0, 100)
    noiseWord = randint(0, ALPHABETLENGTHKERN)

    for i, sequence in enumerate(batchX):
        for j, char in enumerate(sequence):
            encoder_input[i][j] = w2iagnostic[char]

    for i, sequence in enumerate(batchY):
        for j, char in enumerate(sequence):
            
           # if putNoise < 70:
           #     decoder_input[i][j] = w2ikern[char]
           # else:
           #     decoder_input[i][j] = noiseWord
           
           decoder_input[i][j] = 0
            
           if j > 0:
               decoder_output[i][j - 1][w2ikern[char]] = 1.

    return encoder_input, decoder_input, decoder_output


def batch_generator(X, Y, batch_size):
    index = 0
    while True:
        BatchX = X[index:index + batch_size]
        BatchY = Y[index:index + batch_size]

        encoder_input, decoder_input, decoder_output = batch_confection(BatchX, BatchY)

        yield [encoder_input, decoder_input], decoder_output

        index = (index + batch_size) % len(X)


def save_checkpoint(model, editionVal):
    model.save("model/checkpointsEncDull/model" + str(int(editionVal)) + ".h5")
    print("Saved checkpoint at - " + str(int(editionVal)))

def load_preTrainModel(path):
    return load_model(path)

def trainloop(model, w2ik, w2ia, i2wk, i2wa, lenAgnostic, lenKern, X, Y, validation_split):
    global ALPHABETLENGTHKERN, ALPHABETLENGTHAGNOSTIC
    global w2iagnostic, w2ikern, i2wagnostic, i2wkern

    seed(1)

    w2iagnostic = w2ia
    w2ikern = w2ik
    i2wkern = i2wk
    i2wagnostic = i2wa

    ALPHABETLENGTHAGNOSTIC = lenAgnostic
    ALPHABETLENGTHKERN = lenKern

    ##GLOBAL VARIABLES SET UP, TIME TO TRAIN

    split_index = int(len(X) * validation_split)
    X_train = X[split_index:]
    Y_train = Y[split_index:]

    X_validation = X[:split_index]
    Y_validation = Y[:split_index]

    X_trainTest = X[split_index:split_index + len(X_validation)]
    Y_trainTest = Y[split_index:split_index + len(Y_validation)]

    generator = batch_generator(X_train, Y_train, BATCH_SIZE)
    X_Val, Y_Val, T_validation = batch_confection(X_validation, Y_validation)
    X_TrainTest, Y_TrainTest, T_TrainTest = batch_confection(X_trainTest, Y_trainTest)

    print("--- TRAIN RELEVANT DATA ---")
    print()
    print("Training sequences to eval: ", X_trainTest.shape[0])
    print()
    print("Validation sequences to eval: ", X_Val.shape[0])
    print()
    print("---------------------------")

    bestesditionVal = 1000
    for epoch in range(TRAINEPOCHS):
        print()
        print('----> Epoch', epoch * EVAL_EPOCH_STRIDE)
        history = model.fit_generator(generator,
                                      steps_per_epoch=len(X_train)//BATCH_SIZE,
                                      verbose=2,
                                      epochs=EVAL_EPOCH_STRIDE)

        predTest = model.predict([X_Val, Y_Val], batch_size=BATCH_SIZE)
        predTrain = model.predict([X_TrainTest, Y_TrainTest], batch_size=BATCH_SIZE)

        current_edition_val = 0
        current_edition_trainTest = 0

        iToEval = randint(0, BATCH_SIZE - 1)

        for i, prediction in enumerate(predTest):
            raw_sequence = [i2wkern[char] for char in np.argmax(prediction, axis=1)]
            raw_trueseq = [i2wkern[char] for char in np.argmax(T_validation[i], axis=1)]

            predictionSequence = []
            truesequence = []

            for char in raw_sequence:
                predictionSequence += [char]
                if char == '</s>':
                    break
            
            for char in raw_trueseq:
                truesequence += [char]
                if char == '</s>':
                    break

            
            if i == iToEval:
                print("Showing result: " + str(iToEval))
                print("Prediction: " + str(predictionSequence))
                print("True: " + str(truesequence))

            current_edition_val += edit_distance(truesequence, predictionSequence) / len(truesequence)
        
        for i, prediction in enumerate(predTrain):
            raw_sequence = [i2wkern[char] for char in np.argmax(prediction, axis=1)]
            raw_trueseq = [i2wkern[char] for char in np.argmax(T_TrainTest[i], axis=1)]

            predictionSequence = []
            truesequence = []

            for char in raw_sequence:
                predictionSequence += [char]
                if char == '</s>':
                    break
            
            for char in raw_trueseq:
                truesequence += [char]
                if char == '</s>':
                    break
            
            current_edition_trainTest += edit_distance(truesequence, predictionSequence) / len(truesequence) 

        
        
        current_edition_val = (100. * current_edition_val) / len(X_validation)
        current_edition_trainTest = (100. * current_edition_trainTest) / len(X_trainTest)

        print()
        print()
        print('Epoch ' + str(((epoch + 1) * EVAL_EPOCH_STRIDE) - 1))
        print()
        print('SER avg train cheating: ' + str(current_edition_trainTest))
        print()
        print('SER avg validation cheating: ' + str(current_edition_val))
        print()
        
        edition_nocheating_val = 0
        for i, sequenceToTest in enumerate(X_Val):
            trueseq = [char for char in np.argmax(T_validation[i], axis=1)]
            prediction, true = test_sequence2(sequenceToTest,model, w2ikern, i2wkern, trueseq)

            predictionSequence = []
            truesequence = []

            for char in prediction:
                predictionSequence += [char]
                if char == '</s>':
                    break
            
            for char in true:
                truesequence += [char]
                if char == '</s>':
                    break
           
            if i == iToEval:
                print("Showing nocheating " + str(iToEval))
                print("Had to predict = " + str(truesequence))
                print("Predicted = " + str(prediction))
                
            edition_nocheating_val += edit_distance(truesequence, predictionSequence) / len(truesequence) 
        
        edition_nocheating_testTrain = 0
        for i, sequenceToTestTrain in enumerate(X_TrainTest):
            trueseq = [char for char in np.argmax(T_TrainTest[i], axis=1)]
            prediction, true = test_sequence2(sequenceToTestTrain, model, w2ikern, i2wkern, trueseq)

            predictionSequence = []
            truesequence = []

            for char in prediction:
                predictionSequence += [char]
                if char == '</s>':
                    break
            
            for char in true:
                truesequence += [char]
                if char == '</s>':
                    break

            edition_nocheating_testTrain += edit_distance(truesequence, predictionSequence) / len(truesequence) 
        
        
        current_edition_val_nocheat = (100. * edition_nocheating_val) / len(X_validation)
        current_edition_trainTest_nocheat = (100. * edition_nocheating_testTrain) / len(X_trainTest)
        
        if current_edition_val_nocheat < bestesditionVal:
            save_checkpoint(model, current_edition_val_nocheat)
            bestesditionVal = current_edition_val_nocheat

        print('SER avg train no cheating: ' + str(current_edition_trainTest_nocheat))
        print()
        print('SER avg validation no cheating: ' + str(current_edition_val_nocheat))
        print()
        print()

    return model, X_Val, Y_Val






