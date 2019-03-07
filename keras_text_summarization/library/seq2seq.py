from __future__ import print_function

from keras.models import Model
from keras.layers import Embedding, Dense, Input, Permute, Flatten, Multiply, Activation, Reshape, TimeDistributed, \
    Merge, merge, RepeatVector, Lambda, Add, BatchNormalization, concatenate, Average, Conv2D, MaxPooling2D, \
    MaxPooling1D, Conv1D, Dropout
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from scipy.stats.morestats import Mean

from keras_text_summarization.library.attention import GravesSequenceAttention
from keras_text_summarization.library.utility.glove_loader import load_glove, GLOVE_EMBEDDING_SIZE
import numpy as np
import os
from keras.optimizers import RMSprop, Adam, Nadam, Adadelta, SGD
from keras import backend as K, constraints, initializers, regularizers
from keras.engine import InputSpec
from keras.layers import LSTM, activations, Wrapper, Recurrent
from keras.engine.topology import Layer

# HIDDEN_UNITS= 100
HIDDEN_UNITS = 500
DEFAULT_BATCH_SIZE = 64
VERBOSE = 1
DEFAULT_EPOCHS = 10

class Seq2SeqSummarizer(object):
    model_name = 'seq2seq'

    def __init__(self, config):
        self.num_input_tokens = config['num_input_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.input_word2idx = config['input_word2idx']
        self.input_idx2word = config['input_idx2word']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.config = config

        self.version = 0
        if 'version' in config:
            self.version = config['version']

        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = Embedding(input_dim=self.num_input_tokens, output_dim=HIDDEN_UNITS,
                                      input_length=self.max_input_seq_length, name='encoder_embedding')
        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm', recurrent_dropout=0.2,
                            dropout=0.2)
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, self.num_target_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm',
                            recurrent_dropout=0.2, dropout=0.2)
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                         initial_state=encoder_states)
        decoder_dense = Dense(units=self.num_target_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = []
            for word in line.lower().split(' '):
                wid = 1
                if word in self.input_word2idx:
                    wid = self.input_word2idx[word]
                x.append(wid)
                if len(x) >= self.max_input_seq_length:
                    break
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        print(temp.shape)
        return temp

    def transform_target_encoding(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'START ' + line.lower() + ' END'
            for word in line2.split(' '):
                x.append(word)
                if len(x) >= self.max_target_seq_length:
                    break
            temp.append(x)

        temp = np.array(temp)
        print(temp.shape)
        return temp

    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_input_seq_length)
                decoder_target_data_batch = np.zeros(
                    shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                decoder_input_data_batch = np.zeros(
                    shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                            if idx > 0:
                                decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqSummarizer.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqSummarizer.model_name + '-config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqSummarizer.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, batch_size=None, model_dir_path=None):
        if epochs is None:
            epochs = DEFAULT_EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        self.version += 1
        self.config['version'] = self.version
        config_file_path = Seq2SeqSummarizer.get_config_file_path(model_dir_path)
        weight_file_path = Seq2SeqSummarizer.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = Seq2SeqSummarizer.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        Ytest = self.transform_target_encoding(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = self.generate_batch(Xtest, Ytest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        input_seq = []
        input_wids = []
        for word in input_text.lower().split(' '):
            idx = 1  # default [UNK]
            if word in self.input_word2idx:
                idx = self.input_word2idx[word]
            input_wids.append(idx)
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_input_seq_length)
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_target_tokens))
        target_seq[0, 0, self.target_word2idx['START']] = 1
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.max_target_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, self.num_target_tokens))
            target_seq[0, 0, sample_token_idx] = 1

            states_value = [h, c]
        return target_text.strip()


class Seq2SeqGloVeSummarizer(object):
    model_name = 'seq2seq-glove'

    def __init__(self, config, lr=0.001):
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.version = 0
        self.lr = lr

        if 'version' in config:
            self.version = config['version']

        self.word2em = dict()
        if 'unknown_emb' in config:
            self.unknown_emb = config['unknown_emb']
        else:
            self.unknown_emb = np.random.rand(1, GLOVE_EMBEDDING_SIZE)
            config['unknown_emb'] = self.unknown_emb

        self.config = config

        encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='encoder_lstm',
                            dropout=0.2, recurrent_dropout=0.2)
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, self.num_target_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm',
                            dropout=0.2, recurrent_dropout=0.2)
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                         initial_state=encoder_states)

        decoder_dense = Dense(units=self.num_target_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        optimizer = RMSprop(lr=self.lr, rho=0.9, epsilon=None, decay=0.0)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.model = model

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def load_glove(self, data_dir_path):
        self.word2em = load_glove(data_dir_path)

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = np.zeros(shape=(self.max_input_seq_length, GLOVE_EMBEDDING_SIZE))
            for idx, word in enumerate(line.lower().split(' ')):
                if idx >= self.max_input_seq_length:
                    break
                emb = self.unknown_emb
                if word in self.word2em:
                    emb = self.word2em[word]
                x[idx, :] = emb
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        print(temp.shape)
        return temp

    def transform_target_encoding(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'START ' + line.lower() + ' END'
            for word in line2.split(' '):
                x.append(word)
                if len(x) >= self.max_target_seq_length:
                    break
            temp.append(x)

        temp = np.array(temp)
        print(temp.shape)
        return temp

    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_input_seq_length)
                decoder_target_data_batch = np.zeros(
                    shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                decoder_input_data_batch = np.zeros(
                    shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                            if idx > 0:
                                decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizer.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizer.model_name + '-config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizer.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, batch_size=None, model_dir_path=None):
        if epochs is None:
            epochs = DEFAULT_EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        self.version += 1
        self.config['version'] = self.version
        config_file_path = Seq2SeqGloVeSummarizer.get_config_file_path(model_dir_path)
        weight_file_path = Seq2SeqGloVeSummarizer.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size)
        np.save(config_file_path, self.config)
        architecture_file_path = Seq2SeqGloVeSummarizer.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        Ytest = self.transform_target_encoding(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = self.generate_batch(Xtest, Ytest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint, tensorboard])
        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        input_seq = np.zeros(shape=(1, self.max_input_seq_length, GLOVE_EMBEDDING_SIZE))
        for idx, word in enumerate(input_text.lower().split(' ')):
            if idx >= self.max_input_seq_length:
                break
            emb = self.unknown_emb  # default [UNK]
            if word in self.word2em:
                emb = self.word2em[word]
            input_seq[0, idx, :] = emb
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_target_tokens))
        target_seq[0, 0, self.target_word2idx['START']] = 1
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.max_target_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, self.num_target_tokens))
            target_seq[0, 0, sample_token_idx] = 1

            states_value = [h, c]
        return target_text.strip()


class Seq2SeqGloVeSummarizerV2(object):

    model_name = 'seq2seq-glove-v2'

    def __init__(self, config):
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.version = 0
        if 'version' in config:
            self.version = config['version']

        self.word2em = dict()
        if 'unknown_emb' in config:
            self.unknown_emb = config['unknown_emb']
        else:
            self.unknown_emb = np.random.rand(1, GLOVE_EMBEDDING_SIZE)
            config['unknown_emb'] = self.unknown_emb

        self.config = config

        encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True,return_sequences=True, name='encoder_lstm',dropout=0.5,go_backwards=True)
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
        encoder_states = [encoder_state_h, encoder_state_c]

        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True,return_sequences=True, name='encoder_lstm2', dropout=0.5,go_backwards=True)
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_outputs)
        encoder_states = [encoder_state_h, encoder_state_c]

        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True,return_sequences=True, name='encoder_lstm3', dropout=0.5,go_backwards=True)
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_outputs)
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm',dropout=0.5)
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                         initial_state=encoder_states)
        decoder_outputs = Dropout(0.5)(decoder_outputs)
        decoder_dense = Dense(units=self.num_target_tokens, activation='sigmoid', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        sgd = Adam()
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.model = model
        self.model.summary()

        self.encoder_model = Model(encoder_inputs, encoder_states)
        self.encoder_model.summary()
        decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)
        self.decoder_model.summary()

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def load_glove(self, data_dir_path):
        self.word2em = load_glove(data_dir_path)

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = np.zeros(shape=(self.max_input_seq_length, GLOVE_EMBEDDING_SIZE))
            for idx, word in enumerate(line.lower().split(' ')):
                if idx >= self.max_input_seq_length:
                    break
                emb = self.unknown_emb
                if word in self.word2em:
                    emb = self.word2em[word]
                x[idx, :] = emb
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        print(temp.shape)
        return temp

    def transform_target_encoding(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'start ' + line.lower() + ' end'
            for word in line2.split(' '):
                x.append(word)
                if len(x) >= self.max_target_seq_length:
                    break
            temp.append(x)

        temp = np.array(temp)
        print(temp.shape)
        return temp

    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_input_seq_length)
                decoder_target_data_batch = np.zeros(shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                decoder_input_data_batch = np.zeros(shape=(batch_size, self.max_target_seq_length, GLOVE_EMBEDDING_SIZE))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.word2em:
                            emb = self.unknown_emb
                            decoder_input_data_batch[lineIdx, idx, :] = emb
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            if idx > 0:
                                decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizerV2.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizerV2.model_name + '-config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizerV2.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, batch_size=None, model_dir_path=None):
        if epochs is None:
            epochs = DEFAULT_EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        self.version += 1
        self.config['version'] = self.version
        config_file_path = Seq2SeqGloVeSummarizerV2.get_config_file_path(model_dir_path)
        weight_file_path = Seq2SeqGloVeSummarizerV2.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        tensorboard = TensorBoard()
        np.save(config_file_path, self.config)
        architecture_file_path = Seq2SeqGloVeSummarizerV2.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        Ytest = self.transform_target_encoding(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = self.generate_batch(Xtest, Ytest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint,tensorboard])
        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        input_seq = np.zeros(shape=(1, self.max_input_seq_length, GLOVE_EMBEDDING_SIZE))
        for idx, word in enumerate(input_text.lower().split(' ')):
            if idx >= self.max_input_seq_length:
                break
            emb = self.unknown_emb  # default [UNK]
            if word in self.word2em:
                emb = self.word2em[word]
            input_seq[0, idx, :] = emb
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, GLOVE_EMBEDDING_SIZE))
        target_seq[0, 0, :] = self.word2em['start']
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'start' and sample_word != 'end':
                target_text += ' ' + sample_word

            if sample_word == 'end' or target_text_len >= self.max_target_seq_length:
                terminated = True

            if sample_word in self.word2em:
                target_seq[0, 0, :] = self.word2em[sample_word]
            else:
                target_seq[0, 0, :] = self.unknown_emb

            states_value = [h, c]
        return target_text.strip()



class Seq2SeqGloVeAttentionSummarizer(object):
    model_name = 'seq2seq-glove-attention'

    '''def __init__(self, config, lr=0.001):
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.version = 0
        self.lr = lr

        if 'version' in config:
            self.version = config['version']

        self.word2em = dict()
        if 'unknown_emb' in config:
            self.unknown_emb = config['unknown_emb']
        else:
            self.unknown_emb = np.random.rand(1, GLOVE_EMBEDDING_SIZE)
            config['unknown_emb'] = self.unknown_emb

        self.config = config

        encoder_inputs = Input(shape=(self.max_input_seq_length, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='encoder_lstm',dropout=0.5)
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
        encoder_states = [encoder_state_h, encoder_state_c]

        # print(encoder_outputs.shape)
        # att_input = Reshape(target_shape=(self.num_target_tokens,HIDDEN_UNITS))(encoder_outputs)
        # attention_mul = self.attention_3d_block(att_input,timesteps=self.num_target_tokens)
        # print(attention_mul.shape)
        # attention_mul = Flatten()(attention_mul)
        # print(attention_mul.shape)

        decoder_inputs = Input(shape=(None, self.num_target_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm',dropout=0.5)
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                         initial_state=encoder_states)


        attention = TimeDistributed(Dense(1,activation='tanh'))(encoder_outputs)
        print(attention.shape)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(HIDDEN_UNITS)(attention)
        attention = Permute([2,1])(attention)
        print('final_attention layer:{}'.format(attention.shape))
        representation = merge([encoder_outputs,attention],mode='mul')
        print('representation shape:{}'.format(representation.shape))
        representation = Lambda(lambda xin: K.sum(xin, axis=1))(representation)
        print('representation shape:{}'.format(representation.shape))

        decoder_dense = Dense(units=self.num_target_tokens, activation='softmax', name='decoder_dense')
        #decoder_outputs = decoder_dense(decoder_outputs)
        probabilities = decoder_dense(representation)
        print(probabilities.shape)
        model = Model([encoder_inputs, decoder_inputs], probabilities)

        optimizer = Adam()

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.model = model

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)
    '''

    def __init__(self, config, lr=0.001):
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.version = 0
        self.lr = lr

        if 'version' in config:
            self.version = config['version']

        self.word2em = dict()
        if 'unknown_emb' in config:
            self.unknown_emb = config['unknown_emb']
        else:
            self.unknown_emb = np.random.rand(1, GLOVE_EMBEDDING_SIZE)
            config['unknown_emb'] = self.unknown_emb

        self.config = config
        # Encoder
        encoder_inputs = Input(shape=(self.max_input_seq_length, GLOVE_EMBEDDING_SIZE))
        cnn_out = Conv1D(64,(3),activation='relu')(encoder_inputs)
        cnn_out = MaxPooling1D(pool_size=(2))(cnn_out)
        #cnn_out = Flatten()(cnn_out)
        encoder_lstm = LSTM(HIDDEN_UNITS, dropout=0.2,  return_state=True,return_sequences=True,recurrent_dropout=0.2,
                            name='encoder')
        encoder_lstm_rev = LSTM(HIDDEN_UNITS, dropout=0.2, return_state=True,return_sequences=True,recurrent_dropout=0.2,
                                 go_backwards=True, name='encoder_rev')

        encoder_output, state_h, state_c = encoder_lstm(cnn_out)
        encoder_output_rev, state_h_rev, state_c_rev = encoder_lstm_rev(cnn_out)

        state_h_final = Add()([state_h, state_h_rev])
        state_c_final = Add()([state_c, state_c_rev])
        encoder_output_final = Add()([encoder_output, encoder_output_rev])

        encoder_lstm2 = LSTM(HIDDEN_UNITS, dropout=0.2,recurrent_dropout=0.2,  return_state=True,
                            name='encoder2')
        encoder_lstm_rev2 = LSTM(HIDDEN_UNITS, dropout=0.2, return_state=True,recurrent_dropout=0.2, go_backwards=True,
             name='encoder_rev2')

        encoder_output, state_h, state_c = encoder_lstm2(encoder_output_final)
        encoder_output_rev, state_h_rev, state_c_rev = encoder_lstm_rev2(encoder_output_final)

        state_h_final2 = Add()([state_h, state_h_rev])
        state_c_final2 = Add()([state_c, state_c_rev])

        # Adding weights
        state_h_final = Add()([state_h_final,state_h_final2])
        state_c_final = Add()([state_c_final, state_c_final2])
        encoder_output_final = Add()([encoder_output,encoder_output_rev])

        encoder_states = [state_h_final, state_c_final]
        encoder_output_final = RepeatVector(self.max_target_seq_length)(encoder_output_final)
        print(encoder_output_final.shape)
        # Decoder
        decoder_inputs = Input(shape=(None, self.num_target_tokens), name='decoder_inputs')
        print('decoder.input_shape: {}'.format(decoder_inputs.shape))
        decoder_lstm = LSTM(HIDDEN_UNITS, return_sequences=True, return_state=True, name='decoder', dropout=0.2,recurrent_dropout=0.2)

        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)


        #decoder_outputs = Reshape(target_shape=(self.max_target_seq_length,HIDDEN_UNITS))(decoder_outputs)
        #out = concatenate([encoder_output_final,decoder_outputs])

        #attention, scores = Attention(return_attention=True)(out)

        decoder_dense = TimeDistributed(Dense(self.num_target_tokens, activation='softmax', name='decoder_dense'))
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        optimizer = RMSprop(lr=self.lr)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.model = model

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        #decoder_outputs_rev, state_h_rev, state_c_rev = decoder_lstm(decoder_inputs,initial_state=decoder_state_inputs)

        #decoder_outputs = Reshape(target_shape=(self.max_target_seq_length, HIDDEN_UNITS))(decoder_outputs)

        #out = concatenate([encoder_output_final, decoder_outputs])
        #decoder_states = [d_state_h, d_state_c]
        #decoder_outputs,state_h,state_c = decoder_lstm(decoder_inputs,initial_state=decoder_state_inputs)
        decoder_states = [state_h,state_c]
        decoder_outputs = decoder_dense(decoder_outputs)

        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def load_glove(self, data_dir_path):
        self.word2em = load_glove(data_dir_path)

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = np.zeros(shape=(self.max_input_seq_length, GLOVE_EMBEDDING_SIZE))
            for idx, word in enumerate(line.lower().split(' ')):
                if idx >= self.max_input_seq_length:
                    break
                emb = self.unknown_emb
                if word in self.word2em:
                    emb = self.word2em[word]
                x[idx, :] = emb
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        print(temp.shape)
        return temp

    def transform_target_encoding(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'START ' + line.lower() + ' END'
            for word in line2.split(' '):
                x.append(word)
                if len(x) >= self.max_target_seq_length:
                    break
            temp.append(x)

        temp = np.array(temp)
        print(temp.shape)
        return temp

    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_input_seq_length)
                decoder_target_data_batch = np.zeros(
                    shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                decoder_input_data_batch = np.zeros(
                    shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                            if idx > 0:
                                decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizer.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizer.model_name + '-config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizer.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, batch_size=None, model_dir_path=None):
        if epochs is None:
            epochs = DEFAULT_EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        self.version += 1
        self.config['version'] = self.version
        config_file_path = Seq2SeqGloVeSummarizer.get_config_file_path(model_dir_path)
        weight_file_path = Seq2SeqGloVeSummarizer.get_weight_file_path(model_dir_path)

        checkpoint = ModelCheckpoint(weight_file_path, monitor='val_loss', save_best_only=True)
        stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size)

        np.save(config_file_path, self.config)
        architecture_file_path = Seq2SeqGloVeSummarizer.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        Ytest = self.transform_target_encoding(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = self.generate_batch(Xtest, Ytest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint, tensorboard])
        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        input_seq = np.zeros(shape=(1, self.max_input_seq_length, GLOVE_EMBEDDING_SIZE))
        for idx, word in enumerate(str(input_text).lower().split(' ')):
            if idx >= self.max_input_seq_length:
                break
            emb = self.unknown_emb  # default [UNK]
            if word in self.word2em:
                emb = self.word2em[word]
            input_seq[0, idx, :] = emb
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_target_tokens))
        target_seq[0, 0, self.target_word2idx['START']] = 1
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.max_target_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, self.num_target_tokens))
            target_seq[0, 0, sample_token_idx] = 1

            states_value = [h, c]
        return target_text.strip()

    '''@staticmethod
    def attention_3d_block(inputs, timesteps, single_attention_vector=False):
        from keras.layers import Lambda, RepeatVector,Permute

        # inputs.shape = (batch_size, time_steps, input_dim)
        input_dim = int(inputs.shape[2])
        print("input_dim : {}".format(input_dim))
        a = Permute((2, 1))(inputs)
        print(a.shape)
        a = Reshape((input_dim, timesteps))(a)  # this line is not useful. It's just to know which dimension is what.
        print(a.shape)
        a = Dense(timesteps, activation='softmax')(a)

        if single_attention_vector:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        print(a_probs.shape)

        output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
        return output_attention_mul'''
