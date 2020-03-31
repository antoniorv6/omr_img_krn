from utils.utils import loadImages, loadDataY, DATA_TYPE
import numpy as np


def CTCKRN():
    fixed_height = 32
    XTrain = loadImages("./Dataset/train.lst")
    YTrain = loadDataY("./Dataset/train.lst", (DATA_TYPE.KERN).value)
    print(XTrain.shape)
    print(YTrain.shape)
    
# ==============================================================
#
#                           MAIN
#
# ==============================================================

#if __name__ == "__main__":
#
#    # ========
#    # TODO Editar segun el problema
#    val_split = 0.1
#    fixed_height = 32
#    # ========
#
#    # Cargar datos
#    X, Y = load_data(_) # TODO [X son imagenes e Y son secuencias de palabras/caracteres]
#    X, Y = shuffle(X, Y)
#
#    # TODO Gestionar vocabulario
#    vocabulary = None # TODO
#    w2i = {symbol:idx for idx,symbol in enumerate(vocabulary)}
#    i2w = {idx:symbol for idx,symbol in enumerate(vocabulary)}
#
#    # Normalización
#    for i in range(min(len(X),len(Y))):
#        img = (255. - X[i]) / 255.
#        width = int(float(fixed_height * img.shape[1]) / img.shape[0])
#        X[i] = cv2.resize(img, (width, fixed_height))
#
#        for idx, symbol in enumerate(Y[i]):
#            Y[i][idx] = w2i[symbol]
#
#
#    # Data split
#    val_idx = int(val_split * len(X))
#    X_train, Y_train = X[val_idx:], Y[val_idx:]
#    X_val, Y_val = X[:val_idx], Y[:val_idx]
#
#    # ===============================================
#    # CRNN
#
#    model_tr, model_pr = get_model(input_shape=(fixed_height,None,1),vocabulary_size=vocabulary_size)
#
#    # ===============================================
#    # Data preparation
#    X_train, Y_train, L_train, T_train = data_preparation(X_train, Y_train, fixed_height)
#
#    # ===============================================
#    # Training
#
#    print('Training with ' + str(X_train.shape[0]) + ' samples.')
#
#    inputs = {'the_input': X_train,
#                  'the_labels': Y_train,
#                  'input_length': L_train,
#                  'label_length': T_train,
#                  }
#    outputs = {'ctc': np.zeros([len(X_train)])}
#
#
#    best_ser = 100
#
#    # TODO Poner el validate en un callback para no tener que hacer super epocas
#    for super_epoch in range(50):
#        model_tr.fit(inputs,outputs, batch_size = mini_batch_size, epochs = 5, verbose = 2)
#        ser = validate(model_pr,X_val,Y_val,i2w)
#        if ser < best_ser:
#            best_ser = ser
#            model_pr.save(args.save_model)
#            print('SER Improved -> Saving model to {}'.format(args.save_model))