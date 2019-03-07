from keras.callbacks import *
from keras.layers import *
from keras.layers.recurrent import *
from keras.models import Model


class Summarizer(object):

    def __init__(self, input_size,num_tokens,input_tokens=100,target_tokens=30,embedding_matrix=None):
        self.input_size = input_size
        self.embedding_matrix = embedding_matrix
        self.input_tokens = input_tokens
        self.target_tokens = target_tokens
        self.num_tokens = num_tokens
        self.model = self.build_model()


    def build_model(self):
        input1 = Input(shape=(self.input_tokens,))
        #TODO: inserire embedding layer trainable
        if self.embedding_matrix:
            x = Embedding(self.num_tokens,100,weights=[self.embedding_matrix],trainable=True)(input1)
        else:
            x = Embedding(self.num_tokens,100,input_length=self.input_tokens)(input1)

        num_filters = [256,256,256,256]
        kernel_size = [10,7,5,3]
        conv_out = []
        for i,num_filters in enumerate(num_filters):
            conv = Conv1D(num_filters,kernel_size=kernel_size[i])(x)
            pool = GlobalMaxPool1D()(conv)
            conv_out.append(pool)
        x = Concatenate()(conv_out)
        #art1,state_h,state_c = LSTM(300,return_state=True)(x)
        #art1 = Dropout(0.5)(art1)
        x = Dense(1024,activation='selu')(x)
        x = AlphaDropout(0.5)(x)

        #summary
        input2 = Input(shape=(self.target_tokens,))
        #TODO: inserire embedding layer trainable
        if self.embedding_matrix:
            y = Embedding(self.num_tokens,100,weights=[self.embedding_matrix],trainable=True)(input2)
        else:
            y = Embedding(self.num_tokens,100,input_length=self.target_tokens)(input2)

        num_filters = [256, 256, 256, 256]
        kernel_size = [10, 7, 5, 3]
        conv_out = []
        for i, num_filters in enumerate(num_filters):
            conv = Conv1D(num_filters, kernel_size=kernel_size[i])(y)
            pool = GlobalMaxPool1D()(conv)
            conv_out.append(pool)
        y = Concatenate()(conv_out)
        y = Dense(1024,activation='selu')(y)
        y = AlphaDropout(0.5)(y)
        #summ1,state_h2,state_c2 = LSTM(300,return_state=True)(input2_emb)
        #summ1 = Dropout(0.5)(summ1)

        #state_h = Multiply()([state_h,state_h2])
        #state_c = Multiply()([state_c,state_c2])
        #enc_states = [state_h,state_c]

        concat = concatenate([x,y])
        concat = RepeatVector(128)(concat)
        output = LSTM(300)(concat)
        #output = dec_lstm(concat,initial_state=enc_states)
        output = Dense(self.target_tokens,activation='softmax')(output)
        self.model = Model([input1,input2],output)

        self.model.summary()

        self.model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
        return self.model

    def fit(self,X,Y,batch_size = 50,epochs = 100,Xtest=None,Ytest=None):
        #TODO: add callbacks
        checkpoint = ModelCheckpoint('experiment.h5', monitor='val_loss', verbose=0, save_best_only=True)
        tensorboard = TensorBoard('./logdir',batch_size=batch_size)
        history = self.model.fit(X,Y,batch_size=batch_size,epochs=epochs,callbacks=[checkpoint,tensorboard], validation_split=0.2, shuffle=True)
        self.evaluate(X,Y,batch_size=batch_size)
        return history

    def evaluate(self,X,Y,batch_size=50):
        self.model.evaluate(X,Y,batch_size=batch_size)
