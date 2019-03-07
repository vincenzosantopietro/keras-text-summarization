from collections import Counter
import os
import re

MAX_INPUT_SEQ_LENGTH = 50
MAX_TARGET_SEQ_LENGTH = 30
MAX_INPUT_VOCAB_SIZE = 5000
MAX_TARGET_VOCAB_SIZE = 2000


def parsetext(dire, category, filename):
    with open("%s/%s" % (dire + category, filename), 'r', encoding="Latin-1") as readin:
        #print("file read successfully")
        text = readin.read()
    return text.lower()


def cleantext(text):
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'s", "s", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"can't", " cannot ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"9\\/11", " 911 ", text)
    text = re.sub(r" u.s", " american ", text)
    text = re.sub(r" u.n", " united nations ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"\_", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[$#@%&*!~?%{}()]", " ", text)
    return text


def load_data(dire, category):
    filenames = []
    """category refers to either training, test or validation"""
    for dirs, subdr, files in os.walk(dire +'/'+ category):
        filenames = files
    return filenames


def fit_text(X, Y, input_seq_max_length=None, target_seq_max_length=None):
    if input_seq_max_length is None:
        input_seq_max_length = MAX_INPUT_SEQ_LENGTH
    if target_seq_max_length is None:
        target_seq_max_length = MAX_TARGET_SEQ_LENGTH
    input_counter = Counter()
    target_counter = Counter()

    max_input_seq_length = 0
    max_target_seq_length = 0

    # for each line
    for line in X:
        text = [word.lower() for word in line.split(' ')]  # tokenize - case unsensitive
        seq_length = len(text)  # num of words in the sequence
        if seq_length > input_seq_max_length:
            text = text[0:input_seq_max_length]  # crop
            seq_length = len(text)
        for word in text:  # words counter
            input_counter[word] += 1
        max_input_seq_length = max(max_input_seq_length, seq_length)

    for line in Y:
        line2 = 'START ' + line.lower() + ' END'  # added start-end tokens
        text = [word for word in line2.split(' ')]
        seq_length = len(text)
        if seq_length > target_seq_max_length:
            text = text[0:target_seq_max_length]
            seq_length = len(text)
        for word in text:
            target_counter[word] += 1
            max_target_seq_length = max(max_target_seq_length, seq_length)

    # building input dictionary
    input_word2idx = dict()
    for idx, word in enumerate(input_counter.most_common(MAX_INPUT_VOCAB_SIZE)):
        input_word2idx[word[0]] = idx + 2  # Adding 2 because of PAD and UNK
    input_word2idx['PAD'] = 0
    input_word2idx['UNK'] = 1
    # reverse dict
    input_idx2word = dict([(idx, word) for word, idx in input_word2idx.items()])

    target_word2idx = dict()
    for idx, word in enumerate(target_counter.most_common(MAX_TARGET_VOCAB_SIZE)):
        target_word2idx[word[0]] = idx + 1
    target_word2idx['UNK'] = 0

    target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])

    num_input_tokens = len(input_word2idx)
    num_target_tokens = len(target_word2idx)

    config = dict()
    config['input_word2idx'] = input_word2idx
    config['input_idx2word'] = input_idx2word
    config['target_word2idx'] = target_word2idx
    config['target_idx2word'] = target_idx2word
    config['num_input_tokens'] = num_input_tokens
    config['num_target_tokens'] = num_target_tokens
    config['max_input_seq_length'] = max_input_seq_length
    config['max_target_seq_length'] = max_target_seq_length

    return config

