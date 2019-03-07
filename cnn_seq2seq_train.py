from __future__ import print_function
import pickle
import pandas as pd
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras_text_summarization.library.utility.plot_utils import plot_and_save_history
from keras_text_summarization.library.seq2seq import Seq2SeqGloVeAttentionSummarizer, Seq2SeqGloVeSummarizer
from keras_text_summarization.library.applications.fake_news_loader import load_data, fit_text, cleantext, parsetext
import numpy as np
from keras import backend as K
import deepdish as dd
import os
from rouge import Rouge
from keras.backend.tensorflow_backend import set_session

LOAD_EXISTING_WEIGHTS = False
NUM_THREADS = 32

config = K.tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS, inter_op_parallelism_threads=NUM_THREADS)
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

set_session(K.tf.Session(config=config))


data_categories = ["training", "validation", "test"]



def main():
    np.random.seed(42)
    data_dir_path = './demo/data'
    very_large_data_dir_path = './very_large_data'
    report_dir_path = './reports'
    model_dir_path = './models'

    '''filenames = load_data(data_dir_path, data_categories[0])
    print(len(filenames))
    data = {'articles': [], 'summaries': []}
    i =-1
    for x in sorted(filenames):
        i +=1
        if i%2 == 0:
            filename = x.split('.')[0]

            if os.path.exists(data_dir_path+data_categories[0]+'/'+filename+'.summ') and os.path.exists(data_dir_path+data_categories[0]+'/'+filename+'.sent'):
                try:
                    data['articles'].append(cleantext(parsetext(data_dir_path,data_categories[0],"{}".format(filename+'.sent'))))
                    data['summaries'].append(
                        cleantext(parsetext(data_dir_path, data_categories[0], "{}".format(filename + '.summ'))))
                except Exception as e:
                    print(e)
        else:
            continue

    # OBSOLETE
    # with open('deepmind_news_training.pickle', 'wb') as handle:
    #    pickle.dump(data,handle)
    dd.io.save('deepmind_training.h5',{'articles':data['articles'], 'summaries':data['summaries']},compression=None)
    print(len(data['articles']))
    print(len(data['summaries']))

    exit(0)'''

    #data = dd.io.load('deepmind_training.h5')
    print('loading csv file ...')
    df = pd.read_csv(data_dir_path + "/Reviews.csv").dropna()
    X = np.array(df['Text'].values)
    Y = np.array(df['Summary'].values)
    
    #with open('deepmind_news_training.pickle', 'rb') as handle:
    #    data = pickle.load(handle)


    # print('loading csv file ...')
    #df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")

    print('extract configuration from input texts ...')
    #Y = df.title
    #X = df['text']
    #Y = data['summaries'][:1000]
    #X = data['articles'][:1000]
    #del data

    config = fit_text(X, Y)
    print(config['max_target_seq_length'])
    print(config['max_input_seq_length'])
    print('configuration extracted from input texts ...')

    summarizer = Seq2SeqGloVeAttentionSummarizer(config,lr=1e-3)
    summarizer.load_glove(very_large_data_dir_path)

    if LOAD_EXISTING_WEIGHTS:
        summarizer.load_weights(
            weight_file_path=Seq2SeqGloVeSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(Xtrain.shape)
    print('training size: ', len(Xtrain))
    print('testing size: ', len(Xtest))

    print('start fitting ...')

    history = summarizer.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=500, batch_size=30)

    history_plot_file_path = report_dir_path + '/' + Seq2SeqGloVeAttentionSummarizer.model_name + '-history.png'
    if LOAD_EXISTING_WEIGHTS:
        history_plot_file_path = report_dir_path + '/' + Seq2SeqGloVeAttentionSummarizer.model_name + '-history-v' + str(
            summarizer.version) + '.png'
    plot_and_save_history(history, summarizer.model_name, history_plot_file_path, metrics={'loss', 'acc'})

    rouge = Rouge()
    scores = rouge.get_scores(hyps=summarizer.summarize(df['Text'][0]), refs=df['Text'][0])
    print(scores)
    #print(df['Text'][0])
    for i in range(10):
        print(summarizer.summarize(df['Text'][i]))
    print("=====================")

    for i in range(10):
        print(df['Summary'][i])
    exit(0)
if __name__ == '__main__':
    main()
