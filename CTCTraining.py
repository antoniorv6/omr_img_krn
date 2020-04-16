from utils.utils import loadImages, loadDataY, DATA_TYPE, check_and_retrieveVocabulary, data_preparation_CTC, validateCTC
from model.ctc_keras import get_model
from sklearn.utils import shuffle
import numpy as np
import cv2


vocabularyNames = ["agnostic", "kern", "skern"]

def CTCTraining(output, data_path):
    fixed_height = 32

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
    best_ser = 100

    for super_epoch in range(50):
       model_tr.fit(inputs,outputs, batch_size = 16, epochs = 5, verbose = 2)
       ser = validateCTC(model_pr, XValidate, YValidate, i2w)
       if ser < best_ser:
           best_ser = ser
           model_pr.save("model/checkpoints/" + vocabularyNames[output-1] + "_model")
           print('SER Improved -> Saving model to {}'.format("model/checkpoints/" + vocabularyNames[output-1] + "_model"))
           #print('SER Improved -> Saving model to')