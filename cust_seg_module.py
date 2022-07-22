# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:45:09 2022

@author: Lai Kar Wei
"""

import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.linear_model import LogisticRegression
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def cramers_corrected_stat(confusion_m):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_m)[0]
    n = confusion_m.sum()
    phi2 = chi2/n
    r,k = confusion_m.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

class EDA:
    def countplot_graph(self, cat_col, df): #displaying visualization for categorical data
        for i in cat_col:
            plt.figure()
            sns.countplot(df[i])
            plt.show()

    def distplot_graph(self, cont_col, df): #displaying visualization for continuous data
        for i in cont_col:
            plt.figure()
            sns.distplot(df[i])
            plt.show()


class FeatSel:
    def cont_col_selection(self, cont_col, tar_col, df): #cont features
        for i in cont_col:
            lr = LogisticRegression()
            lr.fit(np.expand_dims(df[i], axis=-1), df[tar_col]) #X cont y cat
            print(lr.score(np.expand_dims(df[i], axis=-1), df[tar_col]))

    def cat_col_selection(self, cat_col, tar_col, df): #cat: cat vs cat
        for i in cat_col:
            print(i)
            confusion_m = pd.crosstab(df[i], df[tar_col]).to_numpy()
            print(cramers_corrected_stat(confusion_m))


class ModDev:
    def dl_model(self, X_train, nb_class, nb_node=128,dropout_rate=0.3): 
        #values inside this is default unless stated
        
        model = Sequential()
        model.add(Input(shape=np.shape(X_train)[1:])) #9
        model.add(Dense(nb_node, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_node, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_class, activation='softmax'))
        model.summary()
        
        return model


class ModEval:
    def plot_hist_graph(self, hist):
        plt.figure()
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.xlabel('epoch')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.show()

        plt.figure()
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.legend(['Training Accuracy', 'Validation Accuracy'])
        plt.show()