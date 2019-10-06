from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction  import FeatureHasher

cols_to_remove = ['id', 'wpt_name', 'num_private', 'region_code', 'lga', 'ward', 'subvillage', 'recorded_by', 'scheme_name', 'extraction_type_group', 'extraction_type_class', 'payment', 'quality_group', 'quantity_group', 'source_type', 'source_class', 'waterpoint_type_group']
cols_to_normalize = ['amount_tsh', 'gps_height', 'longitude', 'latitude', 'population', 'construction_year']
cols_bool_to_int = ['public_meeting', 'permit']
cols_to_circularize = ['date_recorded']
cols_to_feature_hash = ['funder', 'installer']
cols_to_one_hot = ['basin', 'region', 'district_code', 'scheme_management', 'extraction_type', 'management', 'management_group', 'payment_type', 'water_quality', 'quantity', 'source', 'waterpoint_type']

FEATURE_HASHER_NUM_FEATURES = 16

x_url = "https://raw.githubusercontent.com/carldegs/EE-298Z/master/Pump%20It%20Up/x_train.csv"
y_url = "https://raw.githubusercontent.com/carldegs/EE-298Z/master/Pump%20It%20Up/y_train.csv"
x_problem_url = "https://raw.githubusercontent.com/carldegs/EE-298Z/master/Pump%20It%20Up/x_test.csv"

def drop_and_fill_na(df):
    "Remove columns that will not be used and fill null values with the corresponding alternative values"
    df = df.drop(cols_to_remove, axis=1)
    df[cols_to_one_hot + cols_to_feature_hash] = df[cols_to_one_hot + cols_to_feature_hash].fillna('Not Known')
    df[cols_bool_to_int] = df[cols_bool_to_int].fillna(False)
    return df

def transform_data(df):
    # For date_recorder, convert to three columns, 2 for cyclical day of year recorded and the year recorded
    # https://www.avanwyk.com/encoding-cyclical-features-for-deep-learning/
    df["date_recorded"] = pd.to_datetime(df["date_recorded"], format="%Y-%m-%d")
    df["day_of_year_recorded"] = df["date_recorded"].apply(lambda x: x.timetuple().tm_yday)

    df["year_recorded"] = df["date_recorded"].apply(lambda x: x.year)
    df["doy_recorded_sin"] = df["day_of_year_recorded"].apply(lambda x: np.sin(2 * np.pi * x/365.0)) # TODO: how about leap year?
    df["doy_recorded_cos"] = df["day_of_year_recorded"].apply(lambda x: np.cos(2 * np.pi * x/365.0)) # TODO: how about leap year?
    df = df.drop(["date_recorded", "day_of_year_recorded"], axis=1)

    # One Hot Encode data
    oh_enc_data = oh_enc.transform(df[cols_to_one_hot]).toarray()
    oh_enc_cols = oh_enc.get_feature_names()
    oh_enc_df = pd.DataFrame(oh_enc_data, columns=oh_enc_cols)
    df = df.drop(cols_to_one_hot, axis=1)
    df = pd.concat([df, oh_enc_df], axis=1)

    # Change boolean to int
    df[cols_bool_to_int] = df[cols_bool_to_int].astype(int)

    # Use feature hasher
    funder_feat_hasher_data = funder_feat_hasher.transform(df['funder']).toarray()
    funder_feat_hasher_cols = ['funder_feat_hasher_' + str(i+1) for i in range(FEATURE_HASHER_NUM_FEATURES)]
    funder_feat_hasher_df = pd.DataFrame(funder_feat_hasher_data, columns=funder_feat_hasher_cols)

    installer_feat_hasher_data = installer_feat_hasher.transform(df['installer']).toarray()
    installer_feat_hasher_cols = ['installer_feat_hasher_' + str(j+1) for j in range(FEATURE_HASHER_NUM_FEATURES)]
    installer_feat_hasher_df = pd.DataFrame(installer_feat_hasher_data, columns=installer_feat_hasher_cols)

    df = df.drop(cols_to_feature_hash, axis=1)
    df = pd.concat([df, funder_feat_hasher_df, installer_feat_hasher_df], axis=1)

    # normalize integers
    for col in df[cols_to_normalize + funder_feat_hasher_cols + installer_feat_hasher_cols + ['year_recorded']]:
        df[col] -= df[col].min()
        df[col] /= df[col].max()
        
    return df

# fetch data sets
x = pd.read_csv(x_url)
x_prob = pd.read_csv(x_problem_url)

y = pd.read_csv(y_url)
y = y.pop('status_group').values

# initialize data sets
x = drop_and_fill_na(x)
x_prob = drop_and_fill_na(x_prob)

# Fit encoders and transform x
oh_enc = OneHotEncoder(handle_unknown='ignore')
oh_enc.fit(x[cols_to_one_hot])

y_oh_enc = OneHotEncoder(handle_unknown='ignore')

funder_feat_hasher = FeatureHasher(n_features=FEATURE_HASHER_NUM_FEATURES, input_type='string', alternate_sign=False)
funder_feat_hasher.fit(x['funder'])

installer_feat_hasher = FeatureHasher(n_features=FEATURE_HASHER_NUM_FEATURES, input_type='string', alternate_sign=False)
installer_feat_hasher.fit(x['installer'])

x = transform_data(x)
x_prob = transform_data(x_prob)
y = y_oh_enc.fit_transform(y.reshape(-1,1)).toarray()

# split into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 27)
num_labels = 3
input_size = x_train.shape[1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# network parameters
batch_size = 128
hidden_units = 1024
dropout = 0.2

# Create a 3-layer MLP with ReLU and dropout regularization
model = Sequential()

# layer 1
model.add(Dense(hidden_units, input_dim = input_size))
model.add(Activation('relu'))
model.add(Dropout(dropout))

# layer 2
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))

# output layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()

# create the loss function for a one-hot vector
model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

# train network
model.fit(x_train, y_train, epochs=100, batch_size=batch_size)

# use test data to validate
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\n ACCURACY: %.1f%%" % (100.0 * score[1]))