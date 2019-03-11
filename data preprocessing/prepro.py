import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict


# Load data set
train = pd.read_csv('/Users/dgy/Desktop/we_data/train.csv')
valid = pd.read_csv('/Users/dgy/Desktop/we_data/validation.csv')
test = pd.read_csv('/Users/dgy/Desktop/we_data/test.csv')

# Drop unique/almost unique features
train_1 = train.drop(["bidid", "userid", "IP", "url", "urlid"], axis=1)
valid_1 = valid.drop(["bidid", "userid", "IP", "url", "urlid"], axis=1)
test_1 = test.drop(["bidid", "userid", "IP", "url", "urlid"], axis=1)

train_1.fillna('NULL', inplace=True)
valid_1.fillna('NULL', inplace=True)
test_1.fillna('NULL', inplace=True)

train_1.domain = train_1.domain.apply(lambda x: 'NULL' if x == 'null' else x)
valid_1.domain = valid_1.domain.apply(lambda x: 'NULL' if x == 'null' else x)
test_1.domain = test_1.domain.apply(lambda x: 'NULL' if x == 'null' else x)

#
#   one hot encode - usertag
#
mlb = MultiLabelBinarizer()

train_1.usertag = train_1.usertag.apply(lambda x: x.split(","))
valid_1.usertag = valid_1.usertag.apply(lambda x: x.split(","))
test_1.usertag = test_1.usertag.apply(lambda x: x.split(","))

# test

enc_tag_test = pd.DataFrame(mlb.fit_transform(test_1.usertag),
                            columns=mlb.classes_,
                            index=test_1.index)
test_2 = pd.concat([test_1, enc_tag_test], axis=1)

# train

enc_tag_train = pd.DataFrame(mlb.fit_transform(train_1.usertag),
                             columns=mlb.classes_,
                             index=train_1.index)
train_2 = pd.concat([train_1, enc_tag_train], axis=1)

# valid

enc_tag_valid = pd.DataFrame(mlb.fit_transform(valid_1.usertag),
                             columns=mlb.classes_,
                             index=valid_1.index)
valid_2 = pd.concat([valid_1, enc_tag_valid], axis=1)

# drop original usertag column
test_2 = test_2.drop('usertag', axis=1)
train_2 = train_2.drop('usertag', axis=1)
valid_2 = valid_2.drop('usertag', axis=1)

# slotid and domain
# keep those frequency larger than 5000 in training set

slotid = defaultdict(int)
domain = defaultdict(int)
n = 5000

for k in train_2.slotid:
    slotid[k] += 1

for k in train_2.domain:
    domain[k] += 1

slotid_in = set()
domain_in = set()

for a, b in slotid.items():
    if b > 5000:
        slotid_in |= {a}

for a, b in domain.items():
    if b > 5000:
        domain_in |= {a}

def update(k, y):
    if k in y:
        return (k)
    else:
        return ('null')

train_2.slotid = train_2.slotid.apply(lambda x: update(x, slotid_in))
valid_2.slotid = valid_2.slotid.apply(lambda x: update(x, slotid_in))
test_2.slotid = test_2.slotid.apply(lambda x: update(x, slotid_in))

train_2.domain = train_2.domain.apply(lambda x: update(x, domain_in))
valid_2.domain = valid_2.domain.apply(lambda x: update(x, domain_in))
test_2.domain = test_2.domain.apply(lambda x: update(x, domain_in))

#
# get dummies for following features
#
train_3 = pd.get_dummies(train_2,
                         columns=['advertiser', 'useragent', 'region', 'city', 'adexchange', 'domain', 'slotid',
                                  'slotvisibility', 'slotformat',
                                  'creative', 'keypage'])

valid_3 = pd.get_dummies(valid_2,
                         columns=['advertiser', 'useragent', 'region', 'city', 'adexchange', 'domain', 'slotid',
                                  'slotvisibility', 'slotformat',
                                  'creative', 'keypage'])

test_3 = pd.get_dummies(test_2, columns=['advertiser', 'useragent', 'region', 'city', 'adexchange', 'domain', 'slotid',
                                         'slotvisibility', 'slotformat',
                                         'creative', 'keypage'])

#
#   clean up, validation need to be aligned with train
#

# remove additional columns
drop_cols = [x for x in valid_3.columns if x not in train_3.columns]
valid_3 = valid_3.drop(drop_cols, axis=1)

# fill in zeros for missing columns
missing = [x for x in train_3.columns if x not in valid_3.columns]
for it in missing:
    valid_3[it] = 0

# update order to same
valid_3 = valid_3[list(train_3.columns)]

#
#   clean up, test need be align with train set
#

# remove additional columns
drop_cols = [x for x in test_3.columns if x not in train_3.columns]
test_3 = test_3.drop(drop_cols, axis=1)

# fill in zeros for missing columns
missing = [x for x in train_3.columns if x not in test_3.columns]
for it in missing:
    test_3[it] = 0

# update order to same
test_3 = test_3[list(train_3.columns)]


#
#   save data set
#

test_3.to_csv('/...test_3.csv')
valid_3.to_csv('/...valid_3.csv')
train_3.to_csv('/...train_3.csv')