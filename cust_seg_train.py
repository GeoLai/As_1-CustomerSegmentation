# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:44:03 2022

@author: Lai Kar Wei
"""

import os
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from cust_seg_module import EDA, FeatSel, ModDev, ModEval
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

#%% Constant saving paths
CSV_PATH = os.path.join(os.getcwd(), 'dataset', 'Train.csv')
LOGS_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
OHE_PATH = os.path.join(os.getcwd(), 'model', 'ohe.pkl')
BEST_MODEL_PATH = os.path.join(os.getcwd(), 'model', 'best_model.h5')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model', 'model.h5')

#%% Data Loading
df = pd.read_csv(CSV_PATH)

#%% Data Visualization/ Inspection
df.info()
df.describe().T

#4 columns drop due to where no significant result from such info for attracting subsscribers
# id not required features for training
df = df.drop(labels=['id', 'last_contact_duration', 'num_contacts_in_campaign', 
                     'days_since_prev_campaign_contact', 'num_contacts_prev_campaign'], axis=1) 
#print(df)

cat_col = list(df.columns[df.dtypes=='object']) #to list the cat data which is usually 'object'
cat_col.append('term_deposit_subscribed') #despite this is an int but columns showing result which is categorical

cont_col = list(df.columns[(df.dtypes=='float64') | (df.dtypes=='int64')])
cont_col.remove('term_deposit_subscribed') #term deposit subscription is not a cont values 

eda = EDA()
eda.countplot_graph(cat_col, df) #plot categorical features
eda.distplot_graph(cont_col, df) #plot continuous features

#%% Data Cleaning
df.isna().sum()

msno.matrix(df)
msno.bar(df)

for i in cat_col:
    if i == 'term_deposited_subscribed':
        continue
    else:
        le = LabelEncoder()
        temp = df[i]
        temp[temp.notnull()] = le.fit_transform(temp[df[i].notnull()]) #learn & transform not null values
        df[i] = pd.to_numeric(df[i], errors='coerce')
        PICKLE_SAVE_PATH = os.path.join(os.getcwd(), 'model', i+'_encoder.pkl')
        with open(PICKLE_SAVE_PATH, 'wb') as file:
            pickle.dump(le, file)

knn = KNNImputer()
df_imputed = knn.fit_transform(df)
df_imputed = pd.DataFrame(df_imputed)
df_imputed.columns = df.columns

df_imputed.info()
df_imputed.describe().T

df_imputed['term_deposit_subscribed'].isna().sum()

#%% Features selection
# cont = balance, day_of_month, last_contact_duration, num_contacts_in_campaign, days_since_prev_campaign_contact
# num_contacts_prev_campaign vs 'Segmentation' (categorical)
tar_col = 'term_deposit_subscribed'

fs = FeatSel()
fs.cont_col_selection(cont_col, tar_col, df_imputed)

#%% Data Preprocessing
X = df_imputed[['customer_age', 'balance', 'day_of_month']]
#.drop(labels='term_deposit_subscribed', axis=1)
y = df_imputed['term_deposit_subscribed']

ohe = OneHotEncoder(sparse=(False))
y = ohe.fit_transform(np.array(y).reshape(-1,1))

with open(OHE_PATH, 'wb') as file:
    pickle.dump(OHE_PATH, file)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

#%% Model Deployment
md = ModDev()
nb_class = len(np.unique(y, axis=0))

model = md.dl_model(X_train, nb_class, dropout_rate=0.2)

plot_model(model, show_shapes=(True))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#%% Callbacks
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)

early_callback = EarlyStopping(monitor='val_loss', patience=5)

mdc = ModelCheckpoint(BEST_MODEL_PATH, monitor='val_acc', 
                      save_best_only=(True), mode='max', verbose=1)

#training model
hist = model.fit(X_train, y_train, epochs=50, batch_size=100,
                 validation_data=(X_test, y_test),
                 callbacks=[tensorboard_callback, early_callback])

#%% Model Evaluation

print(hist.history.keys())

me = ModEval()
mod_eval = me.plot_hist_graph(hist)

print(model.evaluate(X_test, y_test))

#%% Model Analysis

pred_y = model.predict(X_test)
pred_y = np.argmax(pred_y, axis=1)
true_y = np.argmax(y_test, axis=1)

cm = confusion_matrix(true_y, pred_y)
cr = classification_report(true_y, pred_y)

labels = ['0', '1']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

print(cm)
print(cr)

#%% Model saving
model.save(MODEL_SAVE_PATH)
