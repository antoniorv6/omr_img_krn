import tqdm
import numpy as np
from model.S2SModel import createS2SModel, createS2SModelAttention, createTranslationModels, createModelS2SAttentionGRU
from model.train import trainloop, load_preTrainModel
from model.test import test_sequence, test_sequence2, test_asTrain, eval_model
import os
import sys
# ====================
PATHKERN = "data/kern/"
PATHAGNOSTIC = "data/agnostic/"
NUMSAMPLES = 20000
w2ikern = {}
i2wkern = {}
w2iagnostic = {}
i2wagnostic = {}
ALPHABETLENGTHKERN = 0
ALPHABETLENGTHAGNOSTIC = 0
# ====================

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def fillWithIndent(vectortofill, character):
    filledVector = []

    for i in range(len(vectortofill)):
        filledVector.append(vectortofill[i])
        if i < len(vectortofill)-1:
            filledVector.append(character)

    return filledVector


def codeSpaces(kernvector, agnosticvector):
    finalEncodedAgnostic = []
    finalEncodedKern = []

    for i in range(len(agnosticvector)):
        subAgnostic = agnosticvector[i].split()
        for string in subAgnostic:
            finalEncodedAgnostic.append(string)

    for i in range(len(kernvector)):
        subKern = kernvector[i].split()
        for string in subKern:
            finalEncodedKern.append(string)

    return finalEncodedAgnostic, finalEncodedKern


def prepareVocabularyandOutput(raw, agnostic):
    global ALPHABETLENGTHKERN, ALPHABETLENGTHAGNOSTIC
    global w2iagnostic, w2ikern, i2wagnostic, i2wkern

    output_sos = '<s>'
    output_eos = '</s>'

    Y = [[output_sos] + sequence + [output_eos] for sequence in raw]

    # Setting up the vocabulary with positions and symbols
    vocabulary = set()

    for sequence in Y:
        vocabulary.update(sequence)

    if agnostic:
        if os.path.exists("vocabulary/w2iagnostic.npy"):
            print("Agnostic vocabulary exists, loading it")
            w2iagnostic = np.load("vocabulary/w2iagnostic.npy", allow_pickle=True).item()
            i2wagnostic = np.load("vocabulary/i2wagnostic.npy", allow_pickle=True).item()
            ALPHABETLENGTHAGNOSTIC = len(vocabulary) + 1
        else:
            w2iagnostic = dict([(char, i+1) for i, char in enumerate(vocabulary)])
            i2wagnostic = dict([(i+1, char) for i, char in enumerate(vocabulary)])
            w2iagnostic["<PAD>"] = 0
            i2wagnostic[0] = "<PAD>"

            ALPHABETLENGTHAGNOSTIC = len(vocabulary) + 1

            np.save("vocabulary/w2iagnostic.npy", w2iagnostic)
            np.save("vocabulary/i2wagnostic.npy", i2wagnostic)

    else:
        if os.path.exists("vocabulary/w2ikern.npy"):
            print("Kern vocabulary exists, loading it")
            w2ikern = np.load("vocabulary/w2ikern.npy", allow_pickle=True).item()
            i2wkern = np.load("vocabulary/i2wkern.npy", allow_pickle=True).item()
            ALPHABETLENGTHKERN = len(vocabulary) + 1 
        else:    
            w2ikern = dict([(char, i+1) for i, char in enumerate(vocabulary)])
            i2wkern = dict([(i+1, char) for i, char in enumerate(vocabulary)])
            w2ikern["<PAD>"] = 0
            i2wkern[0] = "<PAD>"
            ALPHABETLENGTHKERN = len(vocabulary) + 1

            np.save("vocabulary/w2ikern.npy", w2ikern)
            np.save("vocabulary/i2wkern.npy", i2wkern)

    return np.array(Y)


def prepareDataset():

    X = []
    Y = []

    for i in tqdm.tqdm(range(1, NUMSAMPLES + 1)):
        ##Load agnostic in X
        agnosticFile = open(PATHAGNOSTIC + "cod1_" + str((i-1)) + ".txt")
        agnosticSequence = agnosticFile.read()
        kernFile = open(PATHKERN + str((i-1)) + ".kern")
        kernSequence = kernFile.read()

        agnosticPlusSplit = agnosticSequence.split("+")
        kernBRSplit = kernSequence.split("\n")

        agnosticcodedPlus = fillWithIndent(agnosticPlusSplit, "+")
        kerncodedBR = fillWithIndent(kernBRSplit, str("<br>"))

        finalAgnostic, finalKern = codeSpaces(kerncodedBR, agnosticcodedPlus)

        X.append(finalAgnostic)
        Y.append(finalKern)

    return X, Y


if __name__ == '__main__':
    RawX, RawY = prepareDataset()
    RawX = prepareVocabularyandOutput(RawX, True)
    RawY = prepareVocabularyandOutput(RawY, False)

    print("Vocabulary length kern: " + str(ALPHABETLENGTHKERN))
    print("Vocabulary length agnostic: " + str(ALPHABETLENGTHAGNOSTIC))

    #print(RawX[0])
    #print(RawY[0])

    #print(i2wkern[0])

    modelToTrain = createS2SModelAttention(ALPHABETLENGTHKERN, ALPHABETLENGTHAGNOSTIC)
    
   # modelToTrain = load_preTrainModel("modelToLoad/model5.h5")
    
    modelToTest, X_Val, Y_Val = trainloop(modelToTrain, w2ikern, w2iagnostic, i2wkern, i2wagnostic, ALPHABETLENGTHAGNOSTIC, ALPHABETLENGTHKERN, RawX, RawY, 0.01)
    
    #_, X_Val, Y_Val = trainloop(modelToTrain, w2ikern, w2iagnostic, i2wkern, i2wagnostic, ALPHABETLENGTHAGNOSTIC, ALPHABETLENGTHKERN, RawX, RawY, 0.05)

    #model_encoder, model_decoder = createTranslationModels(modelToTrain)

    #test_asTrain(X_Val[0], modelToTrain, i2wkern, Y_Val[0])

    #test_sequence(X_Val[0], model_encoder, model_decoder, ALPHABETLENGTHAGNOSTIC, ALPHABETLENGTHKERN, w2iagnostic, w2ikern, i2wkern, Y_Val[0])

    #secondtest(modelToTrain, X_Val[0], Y_Val[0], w2iagnostic, w2ikern, i2wkern)
    #eval_model(X_Val, Y_Val, modelToTrain, w2ikern, i2wkern)
    test_sequence2(X_Val[0], modelToTrain, w2ikern, i2wkern, Y_Val[0])