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
X_resampled = joblib.load("pCTR_X_train_resampled.pkl")
y_resampled = joblib.load("pCTR_y_train_resampled.pkl")
X_validation = joblib.load("pCTR_X_validation.pkl")
y_validation = joblib.load("pCTR_y_validation.pkl")
features_total_names = joblib.load("total_features_basebid.pkl")

# Set a random seed for repeatability
rand_seed = 100
random.seed(rand_seed)
np.random.seed(rand_seed)

#   define
def FitAndEvaluateClassifier(estimator, features):
    # Data preparation
    X_train = pd.DataFrame(columns=features_total_names, data=X_resampled)[features]
    y_train = y_resampled

    X_val = pd.DataFrame(columns=features_total_names, data=X_validation)[features]
    y_val = y_validation

    # Fit classifier to training data
    estimator.fit(X_train, y_train)

    # Predict on validation set
    preds = estimator.predict(X_val)

    # Predicted probabilities
    pred_proba = estimator.predict_proba(X_val)
    pred_proba_click = [p[1] for p in pred_proba]

    # Evaluate performance
    print("\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, preds))
    print("\n")

    # The most important measure is TP/FP rate for the clicked class
    kpi = precision_score(y_val, preds) * 100
    print("Click Precision: {}".format(kpi))

    # Store probabilities
    submission = pd.DataFrame(data=list(zip(validation["bidid"], pred_proba_click)),
                              columns=["bidid", "click_proba"])

    # submission.to_csv(lr.__doc__.split("(")[0][:-1] + ".csv")

    return kpi, submission


# log reg

lr_res, lr_sub = FitAndEvaluateClassifier(LogisticRegression(C = 0.1, random_state = rand_seed), features_total_names)

# ## Confusion Matrix:
# [[296515   7208]
#  [   127     75]]
# Click Precision: 1.0297954139777565

# xgboost
xgb_res, xgb_sub = FitAndEvaluateClassifier(XGBClassifier(n_estimators = 500, max_depth = 15, learning_rate = 0.1,
                                                          random_state = rand_seed), features_total_names)
# Confusion Matrix:
# [[298484   5239]
#  [   105     97]]
# Click Precision: 1.81784107946027


## visualise

plt.figure(figsize = (15,4))

plt.subplot(1,3,1)
sns.distplot(lr_sub["click_proba"], kde = False)
plt.xlabel("pCTR")
plt.title("Logistic Regression pCTR districution")


plt.subplot(1,3,3)
sns.distplot(xgb_sub["click_proba"], kde = False)
plt.xlabel("pCTR")
plt.title("XGBoost pCTR districution")


#
## tune
#  tune lr
lr_grid = {
           "penalty": ["l1", "l2"],
           "C": [0.001, 0.01, 0.05, 0.1, 1, 10, 100]
          }

# tune xgb first-tone
# xgb_grid = {"n_estimators": [75, 500, 800],
#             "max_depth": [10, 15],
#             "learning_rate": [0.01, 0.1, 0.3, 0.5]}

# tune xgb second-tone
xgb_grid = {"n_estimators": [400, 500, 600],
            "max_depth": [5, 8, 10],
            "learning_rate": [0.05, 0.1, 0.15]}


def Tune(estimator, grid, features, name_store):
    # define
    X_train = pd.DataFrame(columns=features_total_names, data=X_resampled)[features]
    y_train = y_resampled

    X_val = pd.DataFrame(columns=features_total_names, data=X_validation)[features]
    y_val = y_validation

    # Define grid
    grid = RandomizedSearchCV(estimator,
                        param_distributions=grid,
                        scoring="precision",
                        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=rand_seed)
                        )

    # Fit grid on train data
    grid.fit(X_train, y_train)

    # Store best model
    best_model = grid.best_estimator_
    joblib.dump(best_model, name_store)
    print("Best parameters: {}".format(grid.best_params_))

    # Predict on validation set
    preds = grid.predict(X_val)

    # Predicted probabilities
    pred_proba = grid.predict_proba(X_val)
    pred_proba_click = [p[1] for p in pred_proba]

    # Evaluate performance
    print("\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, preds))
    print("\n")

    # The most important measure is TP/FP rate for the clicked class
    kpi = precision_score(y_val, preds) * 100
    print("Click Precision: {}".format(kpi))

    # Store probabilities
    submission = pd.DataFrame(data=list(zip(validation["bidid"], pred_proba_click)),
                              columns=["bidid", "click_proba"])

    return kpi, submission


lr_res_tuned, lr_sub_tuned = Tune(LogisticRegression(random_state = rand_seed),
                                                 lr_grid,
                                                 features_total_names,
                                                 "LR-BestModel.pkl")

###
# Best parameters: {'C': 1, 'penalty': 'l1'}
# Confusion Matrix:
# [[298443   5280]
#  [   125     77]]
# Click Precision: 1.4373716632443532

xgb_res_tuned, xgb_sub_tuned = Tune(XGBClassifier(random_state = rand_seed),
                                                   xgb_grid,
                                                   features_total_names,
                                                   "XGBoost-BestModel2.pkl")



## tuned model

# lr_res, lr_sub = FitAndEvaluateClassifier(LogisticRegression(C = 1, random_state = rand_seed), features_total_names)


### xgb

#
# Best parameters: {'n_estimators': 500, 'max_depth': 10, 'learning_rate': 0.1}
# Confusion Matrix:
# [[298903   4820]
#  [   108     94]]
# Click Precision: 1.912901912901913
#
def StoreXGBModel(features):
    # Data preparation
    X_train = pd.DataFrame(columns=features_total_names, data=X_resampled)[features]
    y_train = y_resampled

    # Model
    xgb_model = XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.1, random_state=rand_seed)

    # Fit grid on train data

    xgb_model.fit(X_train, y_train)

    # Store model
    joblib.dump(xgb_model, "XGBoost-BestModel.pkl")

    return xgb_model

# save

xgb_sub_tuned.to_csv("xgb_pCTR.csv")



####### Evaluation_
xgb_preds = pd.read_csv("xgb_pCTR.csv", index_col  = "Unnamed: 0")
lr_preds = pd.read_csv('/Users/dgy/PycharmProjects/AI/.ipynb_checkpoints/lr_pCTR.csv', index_col  = "Unnamed: 0")
train = pd.read_csv('/Users/dgy/PycharmProjects/AI/or_train.csv')

base_bid = train.loc[ train["click"] == 1, "payprice"].mean()
basebid_max = train.loc[ train["click"] == 1, "payprice"].max()
basebid_min = train.loc[ train["click"] == 1, "payprice"].min()
clicks = sum(train.click)
avgCTR = clicks/train.shape[0]

# define linearstrategy
def LinearStrategy(sub_pCTR, sub_basebid):
    avgCTR = sub_pCTR["click_proba"].mean()
    bidprice = (sub_pCTR["click_proba"] * sub_basebid) / avgCTR
    return bidprice

##
budget = 6250 * 1000


def ValidationDataFrame(submission_pCTR, submission_basebid):
    validation_check = validation[["bidid", "click", "bidprice", "payprice"]]
    validation_check["click_proba"] = submission_pCTR["click_proba"]
    validation_check["bidprice_predicted"] = LinearStrategy(submission_pCTR, submission_basebid)

    return validation_check


def ValidateStrategy(df):
    impressions = 0
    clicks = 0
    cost = 0
    auctions_participated = 0
    balance = budget

    for row in df.iterrows():

        if cost < budget:

            auctions_participated += 1

            if (row[1]["bidprice_predicted"] >= row[1]["payprice"]):

                if (balance > row[1]["bidprice_predicted"]):

                    impressions += 1
                    clicks += row[1]["click"]
                    cost += row[1]["payprice"]
                    balance -= row[1]["payprice"]
                else:
                    pass

        else:
            break

    # Metrics
    ctr = clicks * 100 / impressions
    cpm = cost / impressions
    cpc = cost / clicks / 1000

    print("Strategy statistics:")
    print("Auctions participated: {} | Impressions: {} | Clicks: {} | Cost: {} | CTR: {} | CPM: {} | CPC: {}".format(
        auctions_participated, \
        impressions, clicks, \
        cost, \
        np.round(ctr, 5), \
        np.round(cpm, 5), \
        np.round(cpc, 2)))
    print("\n")

    return impressions, clicks, cost, auctions_participated, ctr, cpm, cpc




#####
pCTR_preds = [lr_preds, xgb_preds]
pCTR_model_names = ["Logistic Regression", "XGBoost"]
results = pd.DataFrame(
    columns=["pCTR_model", "basebid", "Coefficient", "Impressions", "Clicks", "Cost", "CTR", "CPM", "CPC"])
i = 0
k = 0

start = time.time()

for pCTR_pred in pCTR_preds:
    for basebid_pred in basebids:
        # Strategy 1
        validation_check = ValidationDataFrame(pCTR_pred, basebid_pred)
        print(
            "\033[1m pCTR model: {} \033[0m, \033[1m basebid price: {} \033[0m, \033[1m Coefficient: {} \033[0m ".format(
                pCTR_model_names[i], basebid_pred, basebid_pred / base_bid))
        impressions, clicks, cost, auctions_participated, ctr, cpm, cpc = ValidateStrategy(validation_check)
        results.loc[k] = [pCTR_model_names[i], basebid_pred, basebid_pred / base_bid, impressions, clicks, cost, ctr,
                          cpm, cpc]
        k += 1
    i += 1

end = time.time()
print("Total time: {} mins".format((end - start) / 60))


# pd show full needed columns

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

#
#               pCTR_model     basebid  Coefficient Impressions Clicks     Cost       CTR        CPM        CPC
# 9    Logistic Regression   76.701110     0.727273      110586    158  5888957  0.142875  53.252283  37.271880
# 116              XGBoost  119.952014     1.137374      106128    155  6249997  0.146050  58.891122  40.322561


plt.figure(figsize = (9,7))

plt.subplot(3,1,1)
plt.plot(basebids, results.loc[results["pCTR_model"] == "Logistic Regression", "Clicks"].values, "b")
plt.plot(basebids, results.loc[results["pCTR_model"] == "XGBoost", "Clicks"].values, "g")

plt.xlabel("Base Bid")
plt.ylabel("Clicks")
plt.legend(pCTR_model_names)

plt.subplot(3,1,2)
plt.plot(basebids, results.loc[results["pCTR_model"] == "Logistic Regression", "CTR"].values, "b")
plt.plot(basebids, results.loc[results["pCTR_model"] == "XGBoost", "CTR"].values, "g")

plt.xlabel("Base Bid")
plt.ylabel("CTR")
plt.legend(pCTR_model_names)

plt.subplot(3,1,3)
plt.plot(basebids, results.loc[results["pCTR_model"] == "Logistic Regression", "Cost"].values, "b")
plt.plot(basebids, results.loc[results["pCTR_model"] == "XGBoost", "Cost"].values, "g")

plt.xlabel("Base Bid")
plt.ylabel("Cost")
plt.legend(pCTR_model_names)

plt.tight_layout()
# plt.savefig("Linear_Strategy_Results.png")