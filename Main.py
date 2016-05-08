# coding=UTF8
__author__ = 'Paulina Początek'

from spacy import load
from spacy import attrs
from spacy import parts_of_speech as pos
from os import listdir
from os.path import isfile, join
import csv
from sklearn import svm
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn import preprocessing
import itertools
import time
from datetime import datetime
from pylab import pcolor, show, colorbar, xticks, yticks

#MODES
POS_FEATURES = 0
CHARACTER_BASED_FEATURES = 1
WORD_BASED_FEATURES = 2
SENTENCE_BASED_FEATURES = 3
DICTIONARY_BASED_FEATURES = 4
UNIGRAMS_FEATURES = 5
SYNTACTIC_FEATURES = 6

MULTI_LABELED = 0


punctuationArray = ['.', ',', '?', '-', '!', "\"", "("]
stopWords = [u'the', u'a', u'.', u',', u'?']

timestamp = int(time.time())
output_path = 'svm_all_combinations' + str(timestamp) + '.txt'

def main():
    modes = [POS_FEATURES, CHARACTER_BASED_FEATURES, UNIGRAMS_FEATURES, WORD_BASED_FEATURES, SENTENCE_BASED_FEATURES, DICTIONARY_BASED_FEATURES]
    if MULTI_LABELED:
        labels = read_labels_multi('D:\Zgr\Dropbox\Praca\PWr\mgr\dane\Youtube personality merge.csv', ['Xc', 'Ac', 'Cc', 'Ec', 'Oc'])
    else:
        labels = read_labels('D:\Zgr\Dropbox\Praca\PWr\mgr\dane\Youtube personality merge.csv', ['Xc', 'Ac', 'Cc', 'Ec', 'Oc'])
    print labels
    # correlation matrix
    # R = np.corrcoef(x=labels, rowvar=0)
    # pcolor(R)
    # colorbar()
    # yticks(np.arange(0.5, 10.5), range(0, 10))
    # xticks(np.arange(0.5, 10.5), range(0, 10))
    # show()

    data = read_data('D:\Zgr\Dropbox\Praca\PWr\mgr\dane\youtube-personality\\')
    positive_dictionary_raw = unicode(open('D:\Zgr\Dropbox\Praca\PWr\mgr\dane\opinion-lexicon-English\positive-words.txt').read(), "utf-8", errors='ignore')
    negative_dictionary_raw = unicode(open('D:\Zgr\Dropbox\Praca\PWr\mgr\dane\opinion-lexicon-English\\negative-words.txt').read(), "utf-8", errors='ignore')

    positive_dictionary = positive_dictionary_raw.split()
    negative_dictionary = negative_dictionary_raw.split()

    print "read " + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result = ''

    nlp = load('en')
    tokensList = []
    vectors_by_mode = []
    for doc in nlp.pipe(data.itervalues(), batch_size=100, n_threads=4):
        rowVectors = [0] * len(modes)
        if POS_FEATURES in modes:
            pos_features_vector = get_pos_features(doc)
            rowVectors[modes.index(POS_FEATURES)] = pos_features_vector
        if WORD_BASED_FEATURES in modes:
            word_features_vector = []
            # number of stop words
            stop_words_count = doc.count_by(attrs.IS_STOP)
            if stop_words_count:
                stop_words_count = stop_words_count[1L]
            else:
                stop_words_count = 0
            word_features_vector.append(stop_words_count)
            #number of words
            words = doc.count_by(attrs.ORTH, exclude=lambda x: x.is_punct).values()
            words_count = sum(words)
            word_features_vector.append(words_count)
            # number of different words
            word_features_vector.append(len(words))
            rowVectors[modes.index(WORD_BASED_FEATURES)] = word_features_vector
        if SENTENCE_BASED_FEATURES:
            sentence_features_vector = []
            # number of sentences
            sentences_number = len(list(doc.sents))
            sentence_features_vector.append(len(list(doc.sents)))
            # number of words per sentence
            words = doc.count_by(attrs.ORTH, exclude=lambda x: x.is_punct).values()
            words_count = sum(words)
            sentence_features_vector.append(words_count/sentences_number)
            rowVectors[modes.index(SENTENCE_BASED_FEATURES)] = sentence_features_vector
        if any(x in [UNIGRAMS_FEATURES, CHARACTER_BASED_FEATURES, DICTIONARY_BASED_FEATURES] for x in modes):
            if UNIGRAMS_FEATURES in modes:
                unigramsVector = [0] * len(tokensList)
            if CHARACTER_BASED_FEATURES in modes:
                character_features_vector = []
                punctuationVector = [0] * len(punctuationArray)
                uppercase_count = doc.count_by(attrs.IS_TITLE)
                if uppercase_count:
                    uppercase_count = uppercase_count[1L]
                else:
                    uppercase_count = 0
                character_features_vector.append(uppercase_count)
            if DICTIONARY_BASED_FEATURES in modes:
                positiveCount = 0
                negativeCount = 0
            for token in doc:
                if UNIGRAMS_FEATURES in modes:
                    if not token.is_stop:
                        tokenString = token.orth_.lower()
                        if tokenString not in tokensList:
                            tokensList.append(tokenString)
                        index = tokensList.index(tokenString)
                        if len(unigramsVector) > index:
                            unigramsVector[index] = unigramsVector[index]+1
                        else:
                            unigramsVector.append(1)
                if CHARACTER_BASED_FEATURES in modes:
                    if token.pos == pos.PUNCT and token.orth_ in punctuationArray:
                        punctuationVector[punctuationArray.index(token.orth_)] += 1
                if DICTIONARY_BASED_FEATURES in modes:
                    tokenString = token.orth_.lower()
                    if tokenString in positive_dictionary:
                        positiveCount += 1
                    if tokenString in negative_dictionary:
                        negativeCount += 1
            if CHARACTER_BASED_FEATURES in modes:
                character_features_vector += punctuationVector
                characters_count = len(doc.string) - doc.string.count(' ')
                character_features_vector.append(characters_count)
                rowVectors[modes.index(CHARACTER_BASED_FEATURES)] = character_features_vector
            if DICTIONARY_BASED_FEATURES in modes:
                words = doc.count_by(attrs.ORTH, exclude=lambda x: x.is_punct).values()
                words_count = sum(words)
                rowVectors[modes.index(DICTIONARY_BASED_FEATURES)] = [int(positiveCount*1.0/words_count*100), int(negativeCount*1.0/words_count*100)]
        # if SYNTACTIC_FEATURES in modes:
            # for sentence in doc.sents:
            #     print sentence
        if UNIGRAMS_FEATURES in modes:
            rowVectors[modes.index(UNIGRAMS_FEATURES)] = unigramsVector
        vectors_by_mode.append(rowVectors)

    if UNIGRAMS_FEATURES in modes:
        vector_size = len(vectors_by_mode[-1][modes.index(UNIGRAMS_FEATURES)])
        for vector in vectors_by_mode:
            vector[modes.index(UNIGRAMS_FEATURES)] += [0] * (vector_size - len(vector[modes.index(UNIGRAMS_FEATURES)]))
    for L in range(0, len(modes) + 1):
        for current_modes in [modes]: #itertools.combinations(modes, L):
            if current_modes:
                # join current modes in one new features vector
                current_vectors = []
                for row in vectors_by_mode:
                    current_row = []
                    for mode in current_modes:
                        mode_features = row[modes.index(mode)]
                        if type(mode_features) is list:
                            current_row += mode_features
                        else:
                            current_row.append(mode_features)
                    current_vectors.append(current_row)
                print "vector length: " + str(len(current_vectors[0]))
                current_vectors = np.array(current_vectors)
                print current_modes
                print 'learning ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if MULTI_LABELED:
                    clf = OneVsRestClassifier(svm.SVC(kernel='linear')).fit(current_vectors, labels)
                else:
                    clf = svm.SVC(kernel='rbf').fit(current_vectors, labels)
                print 'classification ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                predicted = cross_validation.cross_val_predict(clf, current_vectors, labels, cv=5)
                print 'scores ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                result = ','.join(str(v) for v in current_modes) + '\n'
                result += classification_report(labels, predicted)
                result += '\n'
                print result
                results_file = open(output_path, 'ab')
                results_file.write(result)
                results_file.close()


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def read_labels_multi(labels_path, labels_names):
    labels = []
    with open(labels_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row_labels = []
            for label_name in labels_names:
                row_labels.append(int(row[label_name]))
            labels.append(row_labels)
    return np.array(labels)

def read_labels(labels_path, labels_names):
    labels = []
    with open(labels_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row_labels = []
            for label_name in labels_names:
                row_labels.append(str(row[label_name]))
            labels.append("".join(row_labels))
    return np.array(labels)


def read_data(data_path):
    dataFilesNames = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    data = {}
    for fileName in dataFilesNames:
        data[fileName] = unicode(open(data_path+fileName).read(), "utf-8")
    return data


def get_pos_features(doc):
    pos_ids = pos.IDS.values()
    posCount = doc.count_by(attrs.POS)
    vector = []
    for id in pos_ids:
        if id in posCount:
            vector.append(posCount[id])
        else:
            vector.append(0)
    return vector


if __name__ == "__main__":
    main()
