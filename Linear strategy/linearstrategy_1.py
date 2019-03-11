import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import math
import random
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, OneClassSVM
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
import time
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, precision_score, make_scorer

# load files
validation = pd.read_csv('/Users/dgy/PycharmProjects/AI/or_validation.csv')
X_downsampled = joblib.load("pCTR_X_train_downsampled.pkl")
y_downsampled = joblib.load("pCTR_y_train_downsampled.pkl")
X_validation = joblib.load("pCTR_X_validation.pkl")
y_validation = joblib.load("pCTR_y_validation.pkl")
features_total_names = joblib.load("total_features_basebid.pkl")

# Set a random seed
rand_seed = 100

# read files 
X_train = pd.DataFrame(columns=features_total_names, data=X_downsampled)
y_train = y_downsampled

X_val = pd.DataFrame(columns=features_total_names, data=X_validation)
y_val = y_validation

# tune estimator
lr_grid = {
           "penalty": ["l1", "l2"],
           "C": [0.001, 0.01, 0.05, 0.1, 1, 10, 100]
          }


xgb_grid = {"n_estimators": [400, 500, 600],
            "max_depth": [5, 8, 10],
            "learning_rate": [0.05, 0.1, 0.15]}

estimator1 = LogisticRegression(C = 0.1, random_state = rand_seed)

estimator2 = XGBClassifier(n_estimators = 500, max_depth = 15, learning_rate = 0.1,
                                                          random_state = rand_seed)

estimator3 = LGBMClassifier(
learning_rate =0.1,
n_estimators=600,
max_depth=5,
min_child_weight=7,
subsample=0.8,
colsample_bytree=0.8,
reg_alpha=0.01,
#objective= 'binary:logistic',
)

grid = RandomizedSearchCV(estimator,
                        param_distributions=grid,
                        scoring="precision",
                        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=rand_seed)
                        )

    # Fit grid on train data
 grid.fit(X_train, y_train)

 preds = grid.predict(X_val)

    # Predicted probabilities
 pred_proba = grid.predict_proba(X_val)
 pred_proba_click = [p[1] for p in pred_proba]

 # Find the model with the highest precision_score
    precision = precision_score(y_val, preds) * 100


####### Evaluation_
xgb_preds = pd.read_csv("xgb_pCTR.csv", index_col  = "Unnamed: 0")
lr_preds = pd.read_csv('/Users/dgy/PycharmProjects/AI/.ipynb_checkpoints/lr_pCTR.csv', index_col  = "Unnamed: 0")
train = pd.read_csv('/Users/dgy/PycharmProjects/AI/or_train.csv')

base_bid = train.loc[ train["click"] == 1, "payprice"].mean()
clicks = sum(train.click)
avgCTR = clicks/train.shape[0]

# define linearstrategy
def LinearStrategy(pCTR, basebid):
    avgCTR = pCTR["click_proba"].mean()
    bidprice = (pCTR["click_proba"] * basebid) / avgCTR
    return bidprice

budget = 6250 * 1000

def Val_df(s_pCTR, s_basebid):
    validation_1 = validation[["bidid", "click", "bidprice", "payprice"]]
    validation_1["click_proba"] = s_pCTR["click_proba"]
    validation_1["bidprice_predicted"] = LinearStrategy(s_pCTR, s_basebid)

    return validation_1

def ValidateStrategy(df):
    impressions = 0
    clicks = 0
    cost = 0
    num = 0
    balance = budget
    for row in df.iterrows():
        if cost < budget:
            num += 1
            if (row[1]["p_bidprice"] >= row[1]["payprice"]):
                if (balance > row[1]["p_bidprice"]):
                    impressions += 1
                    clicks += row[1]["click"]
                    cost += row[1]["payprice"]
                    balance -= row[1]["payprice"]
                else:
                    pass
        else:
            break
    ctr = clicks * 100 / impressions
    cpm = cost / impressions
    cpc = cost / clicks / 1000
    print("num: {} | Impressions: {} | Clicks: {} | Cost: {} | CTR: {} | CPM: {} | CPC: {}".format(
        num, \
        impressions, clicks, \
        cost, \
        np.round(ctr, 5), \
        np.round(cpm, 5), \
        np.round(cpc, 2)))
    print("\n")

    return impressions, clicks, cost, auctions_participated, ctr, cpm, cpc

pCTR_preds = [lr_preds, xgb_preds]
pCTR_model_names = ["Logistic Regression", "XGBoost"]
results = pd.DataFrame(
    columns=["pCTR_model", "basebid", "Coefficient", "Impressions", "Clicks", "Cost", "CTR", "CPM", "CPC"])
i = 0
k = 0

for pCTR_pred in pCTR_preds:
    for basebid_pred in basebids:
        # Strategy 1
        validation_check = ValidationDataFrame(pCTR_pred, basebid_pred)
        impressions, clicks, cost, auctions_participated, ctr, cpm, cpc = ValidateStrategy(validation_check)
        k += 1
        i += 1
        results.loc[k] = [pCTR_model_names[i], basebid_pred, basebid_pred / base_bid, impressions, clicks, cost, ctr,
                          cpm, cpc]
       





