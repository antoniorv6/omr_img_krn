from keras.layers import Input, Dense, LSTM, Embedding, Activation, Concatenate, TimeDistributed, Dot, GRU, Bidirectional
from keras.models import Model
from keras.layers import dot
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

#Cool constants
RNN_NEURONS = 512
EMBEDDINGSIZE = 64

#MODEL FOR OUR BASELINE
def createS2SModel(KERNSIZE, AGNOSTICSIZE):
    input_layer = Input(shape=(None, AGNOSTICSIZE))
    encoder_output, stateh, statec = LSTM(RNN_NEURONS, return_sequences=True, return_state=True)(input_layer)
    initial_decoder_state = [stateh, statec]

    input_decoder = Input(shape=(None,KERNSIZE))
    decoder_output, _, _ = LSTM(RNN_NEURONS, return_sequences=True, return_state=True)(input_decoder, initial_state=initial_decoder_state)
    output = Dense(KERNSIZE, activation="softmax")(decoder_output)

    model = Model([input_layer, input_decoder], output)
    model.compile(optimizer="rmsprop",loss="categorical_crossentropy")
    model.summary()
    return model


def createS2SModelAttention(KERNSIZE, AGNOSTICSIZE):

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

def createModelS2SAttentionGRU(KERNSIZE, AGNOSTICSIZE):
    initSession()

    input_layer = Input(shape=(None,), name="encoder_input")
    embedding_input_layer = Embedding(input_dim=AGNOSTICSIZE, output_dim=EMBEDDINGSIZE)(input_layer)
    encoder = GRU(RNN_NEURONS, return_sequences=True)(embedding_input_layer)
    encoder_output, initial_decoder_state = GRU(RNN_NEURONS, return_sequences=True, return_state=True)(encoder)
    input_decoder = Input(shape=(None,), name="decoder_input")
    embedding_output_layer = Embedding(input_dim=KERNSIZE, output_dim=EMBEDDINGSIZE, name="decoder_embedding")(input_decoder)
    decoder = GRU(RNN_NEURONS, return_sequences=True)(embedding_output_layer, initial_state=initial_decoder_state)
    decoder_output, _ = GRU(RNN_NEURONS, return_sequences=True, return_state=True)(decoder)

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



def createTranslationModels(model):
    #WE CREATE A SEPPARATE ENCODER
    encoder_model = Model(model.get_layer('encoder_input').input, model.get_layer('encoder_output').output)

    #AND NOW WE GO WITH THE DECODER
    encoder_outputs = Input(shape=(None, RNN_NEURONS))
    decoder_state_input_h = Input(shape=(RNN_NEURONS,))
    decoder_state_input_c = Input(shape=(RNN_NEURONS,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_inputs = Input(shape=(None,))
    decoder_embedding = model.get_layer("decoder_embedding")(decoder_inputs)
    decoder_lstm_output, state_h, state_c = model.get_layer("decoder_lstm")(decoder_embedding, initial_state=decoder_state_inputs)
    decoder_state_output = [state_h, state_c]

    attention_product = model.get_layer("dot1")([decoder_lstm_output, encoder_outputs])
    attention_mask = model.get_layer("activation")(attention_product)
    context = model.get_layer("dot2")([attention_mask, encoder_outputs])
    decoder_combined_context = model.get_layer("concat")([context, decoder_lstm_output])
    firstoutput = model.get_layer("firstTD")(decoder_combined_context)
    output = model.get_layer("decoder_output")(firstoutput)

    decoder_model = Model([decoder_inputs, encoder_outputs] + decoder_state_inputs, [output] + decoder_state_output)

    encoder_model.summary()
    decoder_model.summary()

    return encoder_model, decoder_model


def initSession():
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth= True
    sess = tf.Session(config=conf)
    set_session(sess)