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

#MODES
CHARACTER_BASED_FEATURES = 1
WORD_BASED_FEATURES = 2
SENTENCE_BASED_FEATURES = 3
DICTIONARY_BASED_FEATURES = 4
POS_FEATURES = 5
UNIGRAMS_FEATURES = 6


punctuationArray = ['.', ',', '?', '-', '!', "\""]
stopWords = [u'the', u'a', u'.', u',', u'?']

def main():
    mode = [POS_FEATURES, CHARACTER_BASED_FEATURES, UNIGRAMS_FEATURES, WORD_BASED_FEATURES, SENTENCE_BASED_FEATURES]
    labels = read_labels('D:\Zgr\Dropbox\Praca\PWr\mgr\dane\Youtube personality merge.csv', ['Xc', 'Ac', 'Cc', 'Ec', 'Oc'])
    data = read_data('D:\Zgr\Dropbox\Praca\PWr\mgr\dane\youtube-personality\\')

    print "read"

    tokensList = []
    vectors = []

    nlp = load('en')
    for doc in nlp.pipe(data.itervalues(), batch_size=100, n_threads=4):
        rowVectors = []
        if POS_FEATURES in mode:
            pos_vector = get_pos_features(doc)
            rowVectors += pos_vector
        if WORD_BASED_FEATURES in mode:
            # number of stop words
            stop_words_count = doc.count_by(attrs.IS_STOP)
            if stop_words_count:
                stop_words_count = stop_words_count[1L]
            else:
                stop_words_count = 0
            rowVectors.append(stop_words_count)
            #number of words
            words = doc.count_by(attrs.ORTH, exclude=lambda x: x.is_punct).values()
            words_count = sum(words)
            rowVectors.append(words_count)
            # number of different words
            rowVectors.append(len(words))
        if SENTENCE_BASED_FEATURES:
            # number of sentences
            sentences_number = len(list(doc.sents))
            rowVectors.append(len(list(doc.sents)))
            # number of words per sentence
            rowVectors.append(words_count/sentences_number)
        if any(x in [UNIGRAMS_FEATURES, CHARACTER_BASED_FEATURES] for x in mode):
            if UNIGRAMS_FEATURES in mode:
                unigramsVector = [0] * len(tokensList)
            if CHARACTER_BASED_FEATURES in mode:
                punctuationVector = [0] * len(punctuationArray)
                uppercase_count = doc.count_by(attrs.IS_TITLE)
                if uppercase_count:
                    uppercase_count = uppercase_count[1L]
                else:
                    uppercase_count = 0
                rowVectors.append(uppercase_count)
            for token in doc:
                if UNIGRAMS_FEATURES in mode:
                    if not token.is_stop:
                        tokenString = token.orth_.lower()
                        if tokenString not in tokensList:
                            tokensList.append(tokenString)
                        index = tokensList.index(tokenString)
                        if len(unigramsVector) > index:
                            unigramsVector[index] = unigramsVector[index]+1
                        else:
                            unigramsVector.append(1)
                if CHARACTER_BASED_FEATURES in mode:
                    if token.pos == pos.PUNCT and token.orth_ in punctuationArray:
                        punctuationVector[punctuationArray.index(token.orth_)] += 1
            if CHARACTER_BASED_FEATURES in mode:
                rowVectors += punctuationVector
                characters_count = len(doc.string) - doc.string.count(' ')
                rowVectors.append(characters_count)
        #has to be last because of unigrams
        if UNIGRAMS_FEATURES in mode:
            rowVectors += unigramsVector
        vectors.append(rowVectors)

    if UNIGRAMS_FEATURES in mode:
        vectorSize = len(vectors[-1])
        newVectors = []
        for vector in vectors:
            newVector = np.pad(vector, (0, vectorSize-len(vector)), 'constant', constant_values=0)
            newVectors.append(newVector)
        vectors = newVectors
        print vectors
    vectors = np.array(vectors)
    labels = np.array(labels)
    print 'classification'
    clf = OneVsRestClassifier(svm.SVC(kernel='linear')).fit(vectors, labels)

    predicted = cross_validation.cross_val_predict(clf, vectors, labels, cv=5)
    print 'scores'
    print(classification_report(labels, predicted, target_names=['Extraversion', 'Agreeableness', 'Conscientiousness', 'Emotional Stability', 'Openness to Experience']))


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def read_labels(labels_path, labels_names):
    labels = []
    with open(labels_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row_labels = []
            for label_name in labels_names:
                row_labels.append(int(row[label_name]))
            labels.append(row_labels)
    return labels


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
