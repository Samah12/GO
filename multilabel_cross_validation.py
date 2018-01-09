# -*- coding: utf-8 -*-
"""
Created on Thu May  5 02:37:02 2016

@author: adityat
"""
from __future__ import division

import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import timeit
from sklearn import decomposition
from random import shuffle

# Constants
METHODNAME = 'knn'  # str(sys.argv[2])
print ("METHODNAME : ", METHODNAME)
K_NEIGHBORS = 10
print ("K_NEIGHBORS: ", K_NEIGHBORS)


def shuffleLists(proteins, Abstract, classes):
    proteins_shuf = []
    Abstract_shuf = []
    classes_shuf = []
    index_shuf = range(len(Abstract))
    shuffle(index_shuf)
    for i in index_shuf:
        proteins_shuf.append(proteins[i])
        Abstract_shuf.append(Abstract[i])
        classes_shuf.append(classes[i])
    return [proteins_shuf, Abstract_shuf, classes_shuf]


class DenseTransformer():
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

def main():
    """
    Current the I have put the complete training and accuracy calculation in this method
    Need to organize it better
    """
    startTime = timeit.default_timer()
    Abstract_train = pickle.load(open('Abstract_train.pkl', 'rb'))
    Abstract_validation = pickle.load(open('Abstract_validation.pkl', 'rb'))
    Abstract_test = pickle.load(open('Abstract_test.pkl', 'rb'))
    Abstract_train = np.asarray(Abstract_train)
    Abstract_validation = np.asarray(Abstract_validation)
    Abstract_test = np.asarray(Abstract_test)

    classes_train = pickle.load(open('classes_train.pkl', 'rb'))
    classes_validation = pickle.load(open('classes_validation.pkl', 'rb'))
    classes_test = pickle.load(open('classes_test.pkl', 'rb'))
    features = [5, 10, 20, 50, 100, 200, 300]
    for NUMFEATURES in features:
        print('=' * 80)
        print ("NUMFEATURES : ", NUMFEATURES)
        if METHODNAME == 'NO_NMF_KNN':
            from sklearn.neighbors import KNeighborsClassifier

            classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', OneVsRestClassifier(KNeighborsClassifier(n_neighbors=K_NEIGHBORS)))])
        if METHODNAME == 'NO_NMF_SVM':
            classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', OneVsRestClassifier(LinearSVC()))])
        if METHODNAME == 'NO_NMF_rf':
            from sklearn.ensemble import RandomForestClassifier

            classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators=100, criterion='entropy')))])
        if METHODNAME == 'NO_NMF_lda':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('FunctionTransformer', DenseTransformer()),
                ('clf', OneVsRestClassifier(LinearDiscriminantAnalysis()))])
        if METHODNAME == 'svm':
            classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('reduce_dim', decomposition.NMF(n_components=NUMFEATURES, random_state=1)),
                ('clf', OneVsRestClassifier(LinearSVC()))])
        if METHODNAME == 'svm_rbf':
            from sklearn import svm

            classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('reduce_dim', decomposition.NMF(n_components=NUMFEATURES, random_state=1)),
                ('clf', OneVsRestClassifier(svm.SVC(kernel='linear')))])
        if METHODNAME == 'gnb':
            from sklearn.naive_bayes import GaussianNB

            classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('reduce_dim', decomposition.NMF(n_components=NUMFEATURES, random_state=1)),
                ('clf', OneVsRestClassifier(GaussianNB()))])
        if METHODNAME == 'dt':
            from sklearn.tree import DecisionTreeClassifier

            classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('reduce_dim', decomposition.NMF(n_components=NUMFEATURES, random_state=1)),
                ('clf', OneVsRestClassifier(DecisionTreeClassifier(max_depth=5)))])
        if METHODNAME == 'rf':
            from sklearn.ensemble import RandomForestClassifier

            classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('reduce_dim', decomposition.NMF(n_components=NUMFEATURES, random_state=1)),
                ('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators=100, criterion='entropy')))])
        if METHODNAME == 'lda':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('reduce_dim', decomposition.NMF(n_components=NUMFEATURES, random_state=1)),
                ('clf', OneVsRestClassifier(LinearDiscriminantAnalysis()))])
        if METHODNAME == 'knn':
            from sklearn.neighbors import KNeighborsClassifier

            classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('reduce_dim', decomposition.NMF(n_components=NUMFEATURES, random_state=1)),
                ('clf', OneVsRestClassifier(KNeighborsClassifier(n_neighbors=K_NEIGHBORS)))])
        mlb = MultiLabelBinarizer()
        classes_train_mlb = mlb.fit_transform(classes_train)
        print ("mlb.classes_: ", mlb.classes_)
        print ("type(mlb.classes_): ", type(mlb.classes_))
        all_classes = mlb.classes_
        # [ 3824,0,0,5488,0,0,0,60089]

        classifier.fit(Abstract_train, classes_train_mlb)
        predicted = classifier.predict(Abstract_validation)
        print ("Prediction Done")
        from sklearn import metrics

        classes_validation = mlb.fit_transform(classes_validation)

        # RESULTS
        precision_per_class = metrics.precision_score(classes_validation, predicted, average=None)
        recall_score_per_class = metrics.recall_score(classes_validation, predicted, average=None)
        f1_score_per_class = metrics.f1_score(classes_validation, predicted, average=None)
        for i in range(0, len(all_classes)):
            print ("class : ", all_classes[i], " Precision : ", precision_per_class[i], " Recall : ", \
            recall_score_per_class[i], " F-Measure : ", f1_score_per_class[i])
        print ("Exact Match : ", metrics.accuracy_score(classes_validation, predicted))
        print ("Average precision_score : ", metrics.precision_score(classes_validation, predicted, average='micro'))
        print ( "Average recall_score : ", metrics.recall_score(classes_validation, predicted, average='micro'))
        print ("Average f1_score : ", metrics.f1_score(classes_validation, predicted, average='micro'))

        classes_validation = (mlb.inverse_transform(classes_validation))
        for i in range(0, len(classes_validation)):
            classes_validation[i] = list(classes_validation[i])

        predicted = (mlb.inverse_transform(predicted))
        for i in range(0, len(predicted)):
            predicted[i] = list(predicted[i])

        Accuracy = 0
        Precision = 0
        Recall = 0
        Fmeasure = 0
        HammingLoss = 0
        for i in range(0, len(predicted)):
            whole_intersect = len(set(predicted[i]).intersection(classes_validation[i]))
            whole_union = len(set(predicted[i]).union(classes_validation[i]))
            len_predicted = len(predicted[i])
            len_gold = len(classes_validation[i])
            if len_predicted == 0:
                Precision += 0
            else:
                Precision += (1 / len(predicted)) * (whole_intersect / len(predicted[i]))
            if len_gold == 0:
                Recall += 0
            else:
                Recall += (1 / len(predicted)) * (whole_intersect / len(classes_validation[i]))
            if (len_gold + len_predicted) == 0:
                Fmeasure += 0
            else:
                Fmeasure += (1 / len(predicted)) * (
                2 * whole_intersect / (len(classes_validation[i]) + len(predicted[i])))
            if whole_union == 0:
                Accuracy += 0
            else:
                Accuracy += (1 / len(predicted)) * (whole_intersect / whole_union)
                HammingLoss += (1 / len(predicted)) * ((whole_union - whole_intersect) / len(mlb.classes_))
        print ("Accuracy, Precision, Recall, Fmeasure : ", Accuracy, Precision, Recall, Fmeasure)
        print ("Hamming Score : ", Accuracy)
        print ("Hamming Loss: ", HammingLoss)
        endTime = timeit.default_timer()
        print ("Time taken: ", (endTime - startTime))
        # Accuracy calculation

#### MAIN STARTS HERE ############
if __name__ == "__main__":
    main()



