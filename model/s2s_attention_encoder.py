from keras.layers import Input, Dense, LSTM, Embedding, Activation, Concatenate, TimeDistributed, Dot, GRU, Bidirectional
from keras.models import Model
from keras.layers import dot
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

#Cool constants
RNN_NEURONS = 512
EMBEDDINGSIZE = 64

def get_encoderModel(KERNSIZE, AGNOSTICSIZE):

    initSession()

    input_layer = Input(shape=(None,), name="encoder_input")
    embedding_input_layer = Embedding(input_dim=AGNOSTICSIZE, output_dim=EMBEDDINGSIZE)(input_layer)
    encoder_output, stateh, statec = LSTM(RNN_NEURONS, return_state=True, return_sequences=True, recurrent_dropout=0.1,  name="encoder_output")(embedding_input_layer)
    initial_decoder_state = [stateh, statec]

    input_decoder = Input(shape=(None,), name="decoder_input")
    embedding_output_layer = Embedding(input_dim=KERNSIZE, output_dim=EMBEDDINGSIZE, name="decoder_embedding")(input_decoder)
    decoder_output, _, _, = LSTM(RNN_NEURONS, return_sequences=True, return_state=True, recurrent_dropout=0.1, name="decoder_lstm")(embedding_output_layer, initial_state=initial_decoder_state)

    #Attention model implementation
    attention_product = Dot(axes=[2,2], name="dot1")([decoder_output, encoder_output]) #dot([decoder_output, encoder_output], axes=[2,2])
    attention_mask = Activation('softmax', name="activation")(attention_product)

    context = Dot(axes=[2,1], name="dot2")([attention_mask, encoder_output])#dot([attention_mask, encoder_output], axes=[2,1])
    decoder_combined_context = Concatenate(name="concat")([context, decoder_output])

    output = TimeDistributed(Dense(64, activation="tanh"), name="firstTD")(decoder_combined_context)
    output = TimeDistributed(Dense(KERNSIZE, activation="softmax"), name="decoder_output")(output)

    model = Model([input_layer, input_decoder], output)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.summary()

    return model


def initSession():
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth= True
    sess = tf.Session(config=conf)
    set_session(sess)