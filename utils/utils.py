import numpy as np
from enum import Enum
import cv2
import sys
import tqdm
from os import path

class DATA_TYPE(Enum):
    AGNOSTIC = 1
    KERN = 2
    SKM = 3


#It only loads our images, so we can dynamically load the Y making no more functions
def loadImages(dataFile, samples):
    X = []
    loadedSamples = 0
    with open(dataFile) as paths:
        line = paths.readline()
        while line:
            imagePath = line.split("\t")[0]
            image = cv2.imread(imagePath)
            X.append(image)
            line = paths.readline()
            loadedSamples+=1
            if loadedSamples == samples:
                break
    
    return np.array(X)

#Inputs
#@dataFile -> Input File to get the data
#@dataToLoadY -> Value from the enum
def loadDataY(dataFile, type, samples):
    Y = [] 
    YSequence = []
    loadedSamples = 0
    with open(dataFile) as paths:
        line = paths.readline()
        while line:
            encodingPath = line.split("\t")[type]
            encodingPath = encodingPath.split("\n")[0]
            yfile = open(encodingPath)
            #KRN and SKM files. Process it as a block.
            if(type == (DATA_TYPE.KERN).value or type == (DATA_TYPE.SKM).value):
                krnLines = yfile.readlines()
                for i, line in enumerate(krnLines):
                    line = line.split("\n")[0] #Dumb trick to get the characters without breaks
                    krnLines[i] = line
                YSequence = krnLines
            else:
                YSequence = yfile.readline().split("\t")

            Y.append(YSequence)
            YSequence = []
            yfile.close()
            line = paths.readline()

            loadedSamples += 1
            if loadedSamples == samples:
                break
    
    return np.array(Y)

def levenshtein(a,b):
    "Computes the Levenshtein distance between a and b."
    n, m = len(a), len(b)

    if n > m:
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def validateCTC(model, X, Y, i2w):
    acc_ed = 0
    acc_len = 0

    for i in range(len(X)):
        pred = model.predict(np.expand_dims(np.expand_dims(X[i],axis=0),axis=-1))[0]

        out_best = np.argmax(pred,axis=1)

        # Greedy decoding (TODO Cambiar por la funcion analoga del backend de keras)
        out_best = [k for k, g in itertools.groupby(list(out_best))]
        decoded = []
        for c in out_best:
            if c < len(i2w):  # CTC Blank must be ignored
                decoded.append(i2w[c])

        acc_ed += levenshtein(decoded,[i2w[label] for label in Y[i]])
        acc_len += len(Y[i])

    ser = 100. * acc_ed / acc_len
    print("Validating with {} samples: {} SER".format(len(Y), str(ser)))
    return ser
# Dados vectores de X (imagenes) e Y (secuencia de etiquetas numéricas -no one hot- devuelve los 4 vectores necesarios para CTC)
def data_preparation_CTC(X, Y, height):
    # X_train, L_train
    max_image_width = max([img.shape[1] for img in X])

    X_train = np.zeros(shape=[len(X), height, max_image_width, 1], dtype=np.float32)
    L_train = np.zeros(shape=[len(X),1])

    for i, img in enumerate(X):
        X_train[i, 0:img.shape[0], 0:img.shape[1],0] = img
        L_train[i] = img.shape[1] // 2 # TODO Calcular el width_reduction de la CRNN

    # Y_train, T_train
    max_length_seq = max([len(w) for w in Y])

    Y_train = np.zeros(shape=[len(X),max_length_seq])
    T_train = np.zeros(shape=[len(X),1])
    for i, seq in enumerate(Y):
        Y_train[i, 0:len(seq)] = seq
        T_train[i] = len(seq)


    return X_train, Y_train, L_train, T_train

def check_and_retrieveVocabulary(YSequences, pathOfSequences, nameOfVoc):
    w2ipath = pathOfSequences + "/" + nameOfVoc + "w2i.npy"
    i2wpath = pathOfSequences + "/" + nameOfVoc + "i2w.npy"

    w2i = []
    i2w = []

    if path.isfile(w2ipath):
        w2i = np.load(w2ipath, allow_pickle=True).items()
        i2w = np.load(i2wpath, allow_pickle=True).items()
    else:
        w2i, i2w = make_vocabulary(YSequences, pathOfSequences, nameOfVoc)

    return w2i, i2w

def make_vocabulary(YSequences, pathToSave, nameOfVoc):
    vocabulary = set()
    for samples in YSequences:
        for sequence in samples:
            for element in sequence:
                vocabulary.update(element)
    
    #Vocabulary created
    w2i = {symbol:idx for idx,symbol in enumerate(vocabulary)}
    i2w = {idx:symbol for idx,symbol in enumerate(vocabulary)}

    #Save the vocabulary
    np.save(pathToSave + "/" + nameOfVoc + "w2i.npy", w2i)
    np.save(pathToSave + "/" + nameOfVoc + "i2w.npy", i2w)

    return w2i, i2w