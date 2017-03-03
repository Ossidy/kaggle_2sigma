# author: Xiao Wang

import numpy as np 
from baselines import *
import cPickle as pickle
from sklearn.preprocessing import StandardScaler

## Read data
print("Loading data...")
Xd = pickle.load(open("./2016.04C.multisnr.pkl",'rb'))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
classes = mods

X = []  
lbl = []
for mod in mods:
    for snr in snrs:
		X.append(Xd[(mod,snr)])
		for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))

X = np.vstack(X)

np.random.seed(2017)
n_examples = X.shape[0]
n_train = int(n_examples * 0.8)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1




# one hot vector representation
Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

# label representation
Y_train_labels = np.array(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test_labels = np.array(map(lambda x: mods.index(lbl[x][0]), test_idx))

print("Number of all Training samples: {}".format(X.shape[0]))
print("Number of classes: {}".format(Y_train.shape[1]))
print("Shape of original features: {}".format(X[0].shape))

# data transformation 
X_train = X_train.reshape(X_train.shape[0],-1)
print("Transformed training data shape: {}".format(X_train.shape))
X_test = X_test.reshape(X_test.shape[0],-1)
print("Transformed testing data shape: {}".format(X_test.shape))

# standarization
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# specify params
rf_params = {
    'n_estimators': 100,
    'max_features': 0.6,
    'max_depth': 16,
    'min_samples_leaf': 2,
    'verbose': 1,
    'n_jobs': -1,
}

lr_params = {
'penalty':'l2', 
'C':5.0, 
'random_state':None, 
'solver':'sag', 
'max_iter':120, 
'multi_class':'ovr', 
'verbose':0, 
'n_jobs':1
}

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mae',
}

svc_params = {
    'loss': 'hinge', 
    'penalty':'l2' ,
    'alpha': 0.0001,
    'l1_ratio': 0.15,
    'n_iter': 20,
    'shuffle': True,
    'learning_rate': 'optimal',
}


base = base_line()
idx = np.random.choice(len(X_train), 2000)
# random forest
# rf_model = base.random_forest(X_train, Y_train_labels, rf_params)
# accs, acc = base.test_model_labels(model,lbl, snrs, test_idx, X_test, Y_test_labels, classes)

# logistic regression
# lr_model = base.logistic_regression(X_train[idx], Y_train_labels[idx], lr_params)
# accs, acc = base.test_model_label(lr_model,lbl, snrs, test_idx, X_test, Y_test_labels, classes)

# xgboosting
# xgb_model = base.xgb_tree(X_train[idx], Y_train_labels[idx], xgb_params)
# accs, acc = base.test_model_label(xgb_model,lbl, snrs, test_idx, X_test, Y_test, classes)

# svm
svm_model = base.sgd_svc(X_train, Y_train_labels, svc_params)
accs, acc = base.test_model_label(svm_model,lbl, snrs, test_idx, X_test, Y_test_labels, classes)
mp = np.mean(accs[-10:])
print("Mean accuracy for snr >= 0 is  {}".format(mp))
model_file = "model.p"
print("model saved in "+model_file+".")
pickle.dump(svm_model, open(model_file, 'wb'))