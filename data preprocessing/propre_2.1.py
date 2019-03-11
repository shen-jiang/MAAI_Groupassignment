import numpy as np
import pandas as pd
import random
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek


# load data set
train = pd.read_csv('/.../train_3.csv')
valid = pd.read_csv('/.../valid_3.csv')
test = pd.read_csv('/.../test_3.csv')

# set a random seed
rand_seed = 100

# features for pCTR predictor
train_out_p = train.loc[train["click"] == 1]
train_in_p = train.loc[train["click"] == 0].sample(n = 20000, random_state = rand_seed).reset_index(drop = True)

train_p = pd.concat([train_in_p, train_out_p], axis = 0)

X_p = train_p.drop(columns = ['bidprice','payprice','click'])

y_p = train_p["click"].values

X_validation = valid.drop(columns = ['bidprice','payprice','click'])
y_validation = valid["click"]


# SMOTE oversampling
sm = SMOTE(ratio=dict({0: train_in_p.shape[0], 1: train_in_p.shape[0]}), random_state=rand_seed)
X_resampled_p, y_resampled_p = sm.fit_sample(X_p, y_p)


#
# features for base_bid predictor.
#

train_out_base = train.loc[train["click"] == 1]
train_in_base = train.loc[train["click"] == 0].sample(n = 40000, random_state = rand_seed).reset_index(drop = True)
train_base = pd.concat([train_in_base, train_out_base], axis = 0)

X_base = train_base.drop(columns = ['bidprice','payprice','click']).values
y_base = train_base["payprice"].values

X_validation_base = valid.drop(columns = ['bidprice','payprice','click'])
y_validation_base = valid["payprice"]

X_resampled_base, y_resample_base = X_base, y_base

#
#   save
#


# total_feature
features_total_names = X_validation.columns.values.tolist()

# save
joblib.dump(features_total_names,"total_features_basebid.pkl")


