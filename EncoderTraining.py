from model.s2s_attention_encoder import get_encoderModel
from utils.utils import DATA_TYPE, loadDataY, data_preparation_CTC, check_and_retrieveVocabulary, batch_generator_encoder, batch_confection_encoder, levenshtein, test_encoderSequence
from sklearn.utils import shuffle
import numpy as np

vocabularyNames = ["enc_paec", "enc_kern", "enc_skern"]

BATCH_SIZE = 8
TRAINEPOCHS = 30

#This encoder is only for agnostic to (kern - SKM)
def EncoderTrain(output, data_path):
     
     print("Training Encoder with - " + str(DATA_TYPE(output)))

     #Load train data
     XTrain = loadDataY(data_path, data_path + "/train.lst", (DATA_TYPE.AGNOSTIC).value, 100)
     YTrain = loadDataY(data_path, data_path + "/train.lst", output, 100)

     #Load test data
     XValidate = loadDataY(data_path, data_path + "/validation.lst", (DATA_TYPE.AGNOSTIC).value, 100)
     YValidate = loadDataY(data_path, data_path + "/validation.lst", output, 100)

     #Prepare the vocabulary
     XTrain, YTrain = shuffle(XTrain, YTrain)
     XValidate, YValidate = shuffle(XValidate, YValidate)

     w2iagnostic, i2wagnostic = check_and_retrieveVocabulary([XTrain, XValidate], "./vocabulary", vocabularyNames[(DATA_TYPE.AGNOSTIC).value - 1])
     w2itarget, i2wtarget = check_and_retrieveVocabulary([YTrain, YValidate], "./vocabulary", vocabularyNames[output-1])

     print("DICTIONARIES LENGTH")
     lenAgnostic = len(w2iagnostic)
     lenTarget   = len(w2itarget)
     print("Agnostic - " + str(lenAgnostic))
     print("Target   - " + str(lenTarget))

     model_t = get_encoderModel(lenTarget, lenAgnostic)

     generator = batch_generator_encoder(XTrain, YTrain, BATCH_SIZE, lenTarget, w2iagnostic, w2itarget)
     X_Val, Y_Val, T_validation = batch_confection_encoder(XValidate, YValidate, lenTarget, w2iagnostic, w2itarget)

     for epoch in range(TRAINEPOCHS):
          print()
          print('----> Epoch', epoch * 2)

          _ = model_t.fit_generator(generator,
                                    steps_per_epoch=len(XTrain)//BATCH_SIZE,
                                    verbose=2,
                                    epochs=2)
          
          current_edition_val = 0
          best_edition_val = 1000000

          for i, sentence in enumerate(X_Val):
               print(i)
               trueseq = [char for char in np.argmax(T_validation[i], axis=1)]
               prediction, true = test_encoderSequence(sentence, model_t, w2itarget, i2wtarget, trueseq)

               predictionSequence = []
               truesequence = []

               for char in true:
                    predictionSequence += [char]
                    if char == '</s>':
                         break
            
               for char in prediction:
                    truesequence += [char]
                    if char == '</s>':
                         break
               
               current_edition_val += levenshtein(truesequence, predictionSequence) / len(truesequence)

          current_edition_val = (100. * current_edition_val) / len(XValidate)

          if current_edition_val < best_edition_val:
               print("Saving model")
               best_edition_val = current_edition_val
               model_t.save("model/checkpoints/" + vocabularyNames[output-1])

          print()
          print()
          print('Epoch ' + str(((epoch + 1) * 2) - 1))
          print()
          print('SER avg validation: ' + str(current_edition_val))
          print()






