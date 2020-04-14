import numpy as np


def test_sequence(sequence, model_enc, model_dec, lenangostic, lenkern, w2iagnostic, w2ikern, i2wkern, trueSequence):
    
    encoder_out, h, c = model_enc.predict(sequence)
    encoder_states = [h, c]

    input_decoder_seq = trueSequence
    #input_decoder_seq[0,0] = w2ikern['<s>']

    decoded_sentence = []

    trueseq = []
    for char in trueSequence:
        trueseq += [char]
        if char == '</s>':
            break


    i = 0

    prediction, _ , _ = model_dec.predict([[input_decoder_seq], encoder_out] + encoder_states, batch_size=1)

    raw_sequence = [i2wkern[char] for char in np.argmax(prediction[0], axis=1)]
    raw_trueseq = [i2wkern[char] for char in trueSequence]
    
    predictionSequence = []
    gt = []
    
    for char in raw_sequence:
        predictionSequence += [char]
        if char == '</s>':
            break
    
    for char in raw_trueseq:
        gt += [char]
        if char == '</s>':
            break
    
    print(gt)
    print(predictionSequence)

    #while True:
    #    decoder_prediction, newh, newc = model_dec.predict([input_decoder_seq, encoder_out] + encoder_states)
    #    print(decoder_prediction)
    #    prediction = np.argmax(decoder_prediction[0], axis=1)[0]
    #    predicted_character = i2wkern[prediction]
    #    print(predicted_character)
    #    if predicted_character == '</s>': 
    #        break
    #    if len(decoded_sentence) > 1000:
    #        print("Reached limit")
    #        break
#
    #    decoded_sentence += [predicted_character]
#
    #    encoder_states = [newh, newc]
#
    #    input_decoder_seq = np.zeros((1,1))
    #    input_decoder_seq[0, 0] = w2ikern[trueseq[i]]
#
    #    i+=1



    #print("Sentence to predict => " + str(trueseq))
    #print("Predicted sentence =>" + str(decoded_sentence))

def test_asTrain(sequence, model, i2wkern, trueSequence):  
    prediction = model.predict([[sequence], [trueSequence]], batch_size=1)[0]

    raw_sequence = [i2wkern[char] for char in np.argmax(prediction, axis=1)]
    raw_trueseq = [i2wkern[char] for char in trueSequence]
    
    predictionSequence = []
    gt = []
    
    for char in raw_sequence:
        predictionSequence += [char]
        if char == '</s>':
            break
    
    for char in raw_trueseq:
        gt += [char]
        if char == '</s>':
            break
    
    print(gt)
    print(predictionSequence)


def decode(i2wkern, sequence, numIt):
    text = []
    for idx in range(numIt):
        text += [i2wkern[sequence[idx]]]
    
    print(text)


def eval_model(X, Y, model, w2ikern, i2wkern):
    
    print("Evaluating on: " + str(len(X)) + " samples")

    edition_val = 0

    for i, sequence in enumerate(X):
        prediction, true = test_sequence2(sequence, model, w2ikern, i2wkern, Y[i])
        edition_val += edit_distance(true, prediction) / len(true)

    SER = (100. * edition_val) / len(X)

    print("Evaluation SER: " + str(SER))



def test_sequence2(sequence, model, w2ikern, i2wkern, trueSequence):
    decoded = [0]
    predicted = []

    trueSequence = [[i2wkern[i] for i in trueSequence]]
    
    for i in range(1, 500):
        decoder_input = np.asarray([decoded])
        
        prediction = model.predict([[sequence], decoder_input])
        decoded.append(0)

        if i2wkern[np.argmax(prediction[0][-1])] == '</s>':
            break
        
        predicted.append(i2wkern[np.argmax(prediction[0][-1])])
    
   # print(predicted)
   # print(trueSequence)

    return predicted, trueSequence[0]