from __future__ import print_function

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras_text_summarization.library.applications.fake_news_loader import load_data, fit_text, cleantext, parsetext
from keras_text_summarization.library.rnn import RecursiveRNN3, RecursiveRNN2
import matplotlib.pyplot as plt
import glob

MAX_INPUT_LENGTH = 50
MAX_OUTPUT_LENGTH = 30
LATENT_DIM = 100
EMBEDDING_DIM = 100
LOAD_EXISTING_WEIGHTS = False

# Function takes a tokenized sentence and returns the words

def main():
    np.random.seed(42)
    data_dir_path = './demo/data'
    very_large_data_dir_path = './very_large_data'
    report_dir_path = './reports'
    model_dir_path = './models'
    print('loading csv file ...')
    df = pd.read_csv(data_dir_path + "/Reviews.csv").dropna()
    X = df['Text']
    Y = df['Summary']
    print(len(X))
    '''fp = open('deepmind_news_training.pickle','rb')
    data = pickle.load(fp)
    fp.close()
    X = data['articles'][:500]
    Y = data['summaries'][:500]'''

    '''for i, value in enumerate(X):
        X[i] = cleantext(str(value))
    for i,value in enumerate(Y):
        Y[i] = cleantext(str(value))
    '''
    '''articles = glob.glob('BBC_1/Articles/*')
    summaries = glob.glob('BBC_1/Summaries/*')

    documents = []
    sums = []
    titles = []
    for folder in articles:
        docs = glob.glob(folder + '/*')
        #summaries path from doc's
        sumpath = folder.split(sep='/')
        sumpath = sumpath[0] + '/Summaries/' + sumpath[2]

        sumpaths = glob.glob(sumpath + '/*')
        for i,article in enumerate(docs):
            try:
                with open(article,'r') as fp:
                    d=fp.readlines()
                    with open(sumpaths[i],'r') as sp:
                        s = sp.read()

                        documents.append(''.join(d[1:]))
                        sums.append(s)
                        titles.append(d[0])

            except Exception as e:
                print('{}'.format(i))
                continue

    X = np.array(documents)
    Y = np.array(titles)
    print('X: {} Y:{}'.format(X.shape,Y.shape))
    print(titles[0])
    print(documents[0])'''
    config = fit_text(X,Y)
    #print(config)

    # Preparing GloVe
    '''embeddings_index = {}
    f = open(os.path.join(very_large_data_dir_path, 'glove.6B.{}d.txt'.format(EMBEDDING_DIM)))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((len(config['input_word2idx']) , EMBEDDING_DIM), dtype='float32')
    for word, i in config['input_word2idx'].items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in glove will be zeros
            embedding_matrix[i] = embedding_vector
    print('Embedding Matrix: {}'.format(embedding_matrix.shape))

    embedding_matrix_target = np.zeros((len(config['target_word2idx']), EMBEDDING_DIM), dtype='float32')
    for word, i in config['target_word2idx'].items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in glove will be zeros
            embedding_matrix_target[i] = embedding_vector
    print('Embedding Matrix: {}'.format(embedding_matrix_target.shape))'''
    
    summarizer = RecursiveRNN3(config=config)


    if LOAD_EXISTING_WEIGHTS:
        weight_file_path = RecursiveRNN3.get_weight_file_path(model_dir_path=model_dir_path)
        print('Loading Weights:' + weight_file_path)
        summarizer.load_weights(weight_file_path=weight_file_path)
    #summarizer.load_glove(very_large_data_dir_path)

    '''vocabulary_size = 100
    tokenizer = Tokenizer(num_words=vocabulary_size, lower=True)
    tokenizer.fit_on_texts()

    Xseq = tokenizer.texts_to_sequences(X)
    Yseq = tokenizer.texts_to_sequences(Y)
    Xf = pad_sequences(Xseq, maxlen=MAX_INPUT_LENGTH)
    Yf = pad_sequences(Yseq, maxlen=MAX_OUTPUT_LENGTH)'''

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    history = summarizer.fit(Xtrain,Ytrain,Xtest,Ytest,epochs=20,batch_size=32)
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.show()


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()

    for i in range(10):
        print(Xtrain[i])
        print('-------')
        print(summarizer.summarize(Xtrain[i]))
        print(Ytrain[i])
        print("=====")

    '''
    tensoboard = TensorBoard(embeddings_layer_names=['emb_1', 'emb_2'], write_images=True)
    checkpoint = ModelCheckpoint('deep-seq2seq-glove.h5', save_best_only=True)

    encoder_inputs = Input(shape=(MAX_INPUT_LENGTH,), name='encoder_inputs')
    embedding_layer = Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                input_length=MAX_INPUT_LENGTH, trainable=False, name='embedding_layer')

    encoder_rnn = LSTM(units=LATENT_DIM, return_state=True, dropout=0.5, recurrent_dropout=0.5, name='lstm_enc1')
    encoder_output, state_h_f, state_c_f = encoder_rnn(embedding_layer(encoder_inputs))
    encoder_rnn2 = LSTM(units=LATENT_DIM, return_state=True, dropout=0.5, recurrent_dropout=0.5,
                        go_backwards=True, name='lstm_enc_back')
    encoder_output, state_h_b, state_c_b = encoder_rnn2(embedding_layer(encoder_inputs))

    state_h = Add()([state_h_f, state_h_b])
    state_c = Add()([state_c_f, state_c_b])

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(MAX_OUTPUT_LENGTH,), name='decoder_inputs')
    embedding_layer_dec = Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                input_length=MAX_INPUT_LENGTH, trainable=False, name='emb_2')
    decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True, dropout=0.5,
                        recurrent_dropout=0.5, name='lstm_decoder')
    decoder_outputs, state_h, state_c = decoder_lstm(embedding_layer_dec(decoder_inputs), initial_state=encoder_states)
    #decoder_outputs = Dropout(rate=0.5)(decoder_outputs)
    decoder_dense = TimeDistributed(Dense(EMBEDDING_DIM, name='decoder_dense'))
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    batch_size = 128

    history = model.fit([Xtrain, Ytrain], Ytrain, batch_size=batch_size, epochs=100, verbose=1, validation_split=0.2,
                        shuffle=True, callbacks=[tensoboard, checkpoint])

'''
if __name__ == '__main__':
    main()
