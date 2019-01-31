from __future__ import print_function
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from keras_text_summarization.library.utility.plot_utils import plot_and_save_history
from keras_text_summarization.library.seq2seq import Seq2SeqGloVeSummarizer
from keras_text_summarization.library.applications.cnn_news_loader import load_data, fit_text, cleantext, parsetext
import numpy as np

LOAD_EXISTING_WEIGHTS = False

data_categories = ["training", "validation", "test"]
data = {"articles": [], "summaries": []}


def main():
    np.random.seed(42)
    data_dir_path = 'cnn'
    very_large_data_dir_path = './very_large_data'
    report_dir_path = './reports'
    model_dir_path = './models'

    '''filenames = load_data(data_dir_path, data_categories[0])
    print(len(filenames))
    for k in range(len(filenames)):
        if k%2 == 0:
            try:
                data['articles'].append(cleantext(parsetext('cnn/',data_categories[0],"{}".format(filenames[k]))))
            except Exception as e:
                print(e)
        else:
            try:
                data['summaries'].append(cleantext(parsetext('cnn/',data_categories[0],"{}".format(filenames[k]))))
            except Exception as e:
                print(e)

    with open('deepmind_news_training.pickle', 'wb') as handle:
        pickle.dump(data,handle)
    print(len(data['articles']))
    print(len(data['summaries']))
    print(data['articles'][0][:100])
    print(data['summaries'][0][:100])
    
    exit(0)'''

    with open('deepmind_news_training.pickle', 'rb') as handle:
        data = pickle.load(handle)

    # exit(0)
    # print('loading csv file ...')
    # df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")

    print('extract configuration from input texts ...')
    # Y = df.title
    # X = df['text']
    Y = data['summaries']
    X = data['articles']
    config = fit_text(X, Y)

    print('configuration extracted from input texts ...')

    summarizer = Seq2SeqGloVeSummarizer(config)
    summarizer.load_glove(very_large_data_dir_path)

    if LOAD_EXISTING_WEIGHTS:
        summarizer.load_weights(
            weight_file_path=Seq2SeqGloVeSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    print('training size: ', len(Xtrain))
    print('testing size: ', len(Xtest))

    print('start fitting ...')
    history = summarizer.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=20, batch_size=16)

    history_plot_file_path = report_dir_path + '/' + Seq2SeqGloVeSummarizer.model_name + '-history.png'
    if LOAD_EXISTING_WEIGHTS:
        history_plot_file_path = report_dir_path + '/' + Seq2SeqGloVeSummarizer.model_name + '-history-v' + str(
            summarizer.version) + '.png'
    plot_and_save_history(history, summarizer.model_name, history_plot_file_path, metrics={'loss', 'acc'})


if __name__ == '__main__':
    main()
