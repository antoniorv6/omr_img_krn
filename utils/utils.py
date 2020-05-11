import numpy as np
from enum import Enum
import cv2
import sys
import tqdm
import itertools
from os import path
from collections import Counter
from keras.models import load_model
import io
from shutil import move

kernCharsToErase = ['y', '/', '\'', 'k', 'kk', 'K', 'KK', 'J', 'JJ', 'L', 'LL', '=', '==', '_']

class DATA_TYPE(Enum):
    PAEC = 1
    KERN = 2
    SKM = 3
    AGNOSTIC = 4


#It only loads our images, so we can dynamically load the Y making no more functions
def loadImages(dataLoc, dataFile, samples):
    X = []
    loadedSamples = 0
    with open(dataFile) as paths:
        line = paths.readline()
        while line:
            imagePath = dataLoc + line.split("\t")[0]
            image = cv2.imread(imagePath, False)
            X.append(image)
            line = paths.readline()
            #loadedSamples+=1
            #if loadedSamples == samples:
            #    break

    return np.array(X)

#Inputs
#@dataFile -> Input File to get the data
#@dataToLoadY -> Value from the enum
def loadDataY(dataLoc, dataFile, type, samples):
    Y = []
    YSequence = []
    loadedSamples = 0
    with open(dataFile) as paths:
        line = paths.readline()
        while line:
            encodingPath = dataLoc + line.split("\t")[type]
            encodingPath = encodingPath.split("\n")[0]
            yfile = open(encodingPath)
            #KRN and SKM files. Process it as a block.
            if(type == (DATA_TYPE.KERN).value or type == (DATA_TYPE.SKM).value):
                krnLines = yfile.readlines()
                for i, line in enumerate(krnLines):
                    line = line.split("\n")[0] #Dumb trick to get the characters without breaks
                    krnLines[i] = line
                YSequence = krnLines
            elif (type == (DATA_TYPE.PAEC).value):
                #Load PAE file
                YSequence = [char for char in yfile.readline()]
                YSequence.remove(YSequence[-1]) #Erase the \n character which does not bring any relevant information
            else:
                YSequence = yfile.readline().split("\t") #Load the agnostic file

            Y.append(YSequence)
            YSequence = []
            yfile.close()
            line = paths.readline()
            
            #loadedSamples += 1
            #if loadedSamples == samples:
            #    break

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

def getCTCValidationData(model, X, Y, i2w):
    acc_ed_ser = 0
    acc_ed_cer = 0
    acc_len_ser = 0
    acc_len_cer = 0

    for i in range(len(X)):
        pred = model.predict(np.expand_dims(np.expand_dims(X[i],axis=0),axis=-1))[0]

        out_best = np.argmax(pred,axis=1)

        # Greedy decoding (TODO Cambiar por la funcion analoga del backend de keras)
        out_best = [k for k, g in itertools.groupby(list(out_best))]
        decoded = []
        for c in out_best:
            if c < len(i2w):  # CTC Blank must be ignored
                decoded.append(i2w[c])

        groundtruth = [i2w[label] for label in Y[i]]
        acc_len_ser += len(Y[i])
        acc_ed_ser += levenshtein(decoded, groundtruth)

        separator = ""
        concatPrediction = separator.join(decoded)
        concatGT = separator.join(groundtruth)

        acc_len_cer += len(concatGT)
        acc_ed_cer += levenshtein(concatPrediction, concatGT)


    ser = 100. * acc_ed_ser / acc_len_ser
    cer = 100. * acc_ed_cer / acc_len_cer
    print("Validating with {} samples: {} SER".format(len(Y), str(ser)))
    print("Validating with {} samples: {} CER".format(len(Y), str(cer)))
    return ser, cer

def writeList(file, array):
    for element in array:
        file.write(element + " ")

def save_image_asResult(image, name):
    cv2.imwrite("test_results/"+name+".jpg", image)

def save_test_results(file, groundtruth, prediction, name, SER, CER):
    file.write(" " + name + ".jpg \n")
    file.write("SER - " + SER)
    file.write("\n")
    file.write("CER - " + CER)
    file.write("\n")
    file.write("Ground truth:" + "\n")
    writeList(file, groundtruth)
    file.write("\n")
    file.write("Prediction:" + "\n")
    writeList(file, prediction)
    file.write("\n")
    file.write("---------------------------------------\n")
    #file.close()
    #file = open("./test_results/" + name + "-prediction" + extension_name, "w")
    #writeList(file, prediction)
    #file.close()

def clean_sequence(sequence):
    clean_sequence = []
    for i, element in enumerate(sequence):
        if i > 0:
            for character in kernCharsToErase:
                element = element.replace(character, '')
            clean_sequence.append(element)
        else:
            clean_sequence.append(element)

    return clean_sequence

def search_confusions(groundtruth, prediction, file, image):
    file.write(str(image) + ".jpg\n")
    try:
        for i, element in enumerate(prediction):
            edit_distance = levenshtein(element, groundtruth[i])
            if edit_distance > 0:
                file.write(groundtruth[i] + "\t" + element + "\t" + str(edit_distance) + "\n")
    except IndexError:
        return
    file.write("---------------------------------------\n")

def getCTCTestData(model, X, Y, i2w, output_name):
    acc_ed_ser = 0
    acc_ed_cer = 0
    acc_len_ser = 0
    acc_len_cer = 0
    sequence_error = 0
    sequence_error_clean = 0

    acc_ed_ser_clean = 0
    acc_ed_cer_clean = 0
    acc_len_ser_clean = 0
    acc_len_cer_clean = 0

    file = open("./test_results/results_raw"+ output_name + ".txt" , "w")
    file_clean = open("./test_results/results_clean"+ output_name + ".txt" , "w")
    file_confusion = open("./test_results/confusion_raw.txt", "w")
    file_confusion_clean = open("./test_results/confusion_clean.txt", "w")


    for i in tqdm.tqdm(range(len(X))):
        pred = model.predict(np.expand_dims(np.expand_dims(X[i],axis=0),axis=-1))[0]

        out_best = np.argmax(pred,axis=1)

        # Greedy decoding (TODO Cambiar por la funcion analoga del backend de keras)
        out_best = [k for k, g in itertools.groupby(list(out_best))]
        decoded = []
        for c in out_best:
            if c < len(i2w):  # CTC Blank must be ignored
                decoded.append(i2w[c])
        
        ##Save files to test results

        groundtruth = Y[i]
        acc_len_ser += len(Y[i])
        distance_edition = levenshtein(decoded, groundtruth)
        acc_ed_ser += distance_edition
        
        if distance_edition is not 0:
            sequence_error += 1
            search_confusions(groundtruth, decoded, file_confusion, i)
        
        local_ser = 100.*distance_edition / len(groundtruth)

        separator = ""
        concatPrediction = separator.join(decoded)
        concatGT = separator.join(groundtruth)

        acc_len_cer += len(concatGT)
        distance_characters = levenshtein(concatPrediction, concatGT)
        acc_ed_cer += distance_characters 

        local_cer = 100.*distance_characters/len(concatGT)

        save_test_results(file, groundtruth, decoded, str(i), str(local_ser), str(local_cer))

        ##PROCESSED WITH THE UNCLEAN ONE, now we get into cleaning

        clean_gt = clean_sequence(groundtruth)
        clean_prediction = clean_sequence(decoded)

        distance_edition = levenshtein(clean_gt, clean_prediction)
        acc_ed_ser_clean += distance_edition
        acc_len_ser_clean += len(clean_gt)

        if distance_edition is not 0:
            sequence_error_clean += 1
            search_confusions(clean_gt, clean_prediction, file_confusion_clean, i)
        
        local_ser = 100.*distance_edition / len(clean_gt)

        separator = ""
        concatPrediction = separator.join(clean_prediction)
        concatGT = separator.join(clean_gt)

        acc_len_cer_clean += len(concatGT)
        distance_characters = levenshtein(concatPrediction, concatGT)
        acc_ed_cer_clean += distance_characters 

        local_cer = 100.*distance_characters/len(concatGT)

        save_test_results(file_clean, clean_gt, clean_prediction, str(i), str(local_ser), str(local_cer))

    ser = 100. * acc_ed_ser / acc_len_ser
    cer = 100. * acc_ed_cer / acc_len_cer
    sequence_error = 100. * sequence_error / len(Y)
    
    ser_clean = 100. * acc_ed_ser_clean / acc_len_ser_clean
    cer_clean = 100. * acc_ed_cer_clean / acc_len_cer_clean
    sequence_error_disp = 100. * sequence_error_clean / len(Y)
    
    print("{} SER".format(str(ser)))
    print("{} CER".format(str(cer)))
    print("{} Sequence Error Rate".format(str(sequence_error)))
    print()
    print("{} SER CLEAN".format(str(ser_clean)))
    print("{} CER CLEAN".format(str(cer_clean)))
    print("{} Sequence Error Rate clean".format(str(sequence_error_disp)))

    file.close()
    file_clean.close()
    file_confusion.close()
    file_confusion_clean.close()

    return ser, cer


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
        w2i = np.load(w2ipath, allow_pickle=True).item()
        i2w = np.load(i2wpath, allow_pickle=True).item()
    else:
        w2i, i2w = make_vocabulary(YSequences, pathOfSequences, nameOfVoc)

    return w2i, i2w

def make_vocabulary(YSequences, pathToSave, nameOfVoc):
    vocabulary = set()
    for samples in YSequences:
        for sequence in samples:
            vocabulary.update(sequence)

    #Vocabulary created
    w2i = {symbol:idx for idx,symbol in enumerate(vocabulary)}
    i2w = {idx:symbol for idx,symbol in enumerate(vocabulary)}

    #Save the vocabulary
    np.save(pathToSave + "/" + nameOfVoc + "w2i.npy", w2i)
    np.save(pathToSave + "/" + nameOfVoc + "i2w.npy", i2w)

    return w2i, i2w

def batch_confection_encoder(batchX, batchY, targetLength, w2iagnostic, w2ikern):
    max_batch_input_len = max([len(sequence) for sequence in batchX])
    max_batch_output_len = max([len(sequence) for sequence in batchY])

    encoder_input = np.zeros((len(batchX), max_batch_input_len), dtype=np.float)
    decoder_input = np.zeros((len(batchY), max_batch_output_len + 1), dtype=np.float)
    decoder_output = np.zeros((len(batchY), max_batch_output_len + 1, targetLength), dtype=np.float)

    for i, sequence in enumerate(batchX):
        for j, char in enumerate(sequence):
            encoder_input[i][j] = w2iagnostic[char]

    for i, sequence in enumerate(batchY):
        for j, char in enumerate(sequence):
           
           decoder_input[i][j] = 0
            
           if j > 0:
               decoder_output[i][j - 1][w2ikern[char]] = 1.

    return encoder_input, decoder_input, decoder_output

def batch_generator_encoder(X, Y, batch_size, targetLength, w2iagnostic, w2itarget):
    index = 0
    while True:
        BatchX = X[index:index + batch_size]
        BatchY = Y[index:index + batch_size]

        encoder_input, decoder_input, decoder_output = batch_confection_encoder(BatchX, BatchY, targetLength, w2iagnostic, w2itarget)

        yield [encoder_input, decoder_input], decoder_output

        index = (index + batch_size) % len(X)

def test_encoderSequence(sequence, model, w2itarget, i2wtarget, trueSequence):
    decoded = [0] #Dummy vector
    #decoded = [w2itarget['<s>']]
    predicted = []

    #trueSequence = [[i2wtarget[i] for i in trueSequence]]

    for i in range(1, 500):
        decoder_input = np.asarray([decoded])
        
        prediction = model.predict([[sequence], decoder_input])
        decoded.append(0) #[0,0]
        #decoded.append(i2wtarget[np.argmax(prediction[0][-1])]) [<s>, **kern]

        if i2wtarget[np.argmax(prediction[0][-1])] == '</s>':
            break
        
        predicted.append(i2wtarget[np.argmax(prediction[0][-1])])

    return predicted, trueSequence

def get_statistic_data(dataList):
    
    listLength = len(dataList)
    #Median
    median = np.median(dataList)

    #Mode
    data = Counter(dataList)
    get_mode = dict(data)
    mode = [k for k, v in get_mode.items() if v == max(list(data.values()))] 

    return median, mode

def load_model_fromfile(outputName):
    path = "./model/checkpoints/" + outputName + "_model.h5"
    model = load_model(path)
    return model

def loadVocabulary(path):
    vocabulary = np.load(path, allow_pickle=True).item()
    return vocabulary
