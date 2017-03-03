# author: Xiao Wang

import numpy as np
import os,random
from sklearn.metrics import log_loss
import time
from sklearn import model_selection, preprocessing, ensemble
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, SGDClassifier
# import xgboost  # this one need to install xgboost for python

# class xgb_model(object):
# 	""" xgblearner wrapper
# 	"""
#     def __init__(self, seed=0, params=None):
#         self.param = params
#         self.param['seed'] = seed
#         self.nrounds = params.pop('nrounds', 250)

#     def train(self, x_train, y_train):
#         dtrain = xgboost.DMatrix(x_train, label=y_train)
#         self.xgb = xgboost.train(self.param, dtrain, self.nrounds)

#     def predict(self, x):
#         return self.xgb.predict(xgboost.DMatrix(x))


class sklearn_model(object):
	"""sklearner wrapper
	"""
	def __init__(self, model,params=None,seed=2017):
		params['random_state'] = seed
		self.model = model(**params)

	def train(self, x_train, y_train):
		self.model.fit(x_train, y_train)

	def predict(self, x_test):
		return self.model.predict(x_test)

class base_line(object):
	""" bench mark class, contains random forest, svm, xgboost etc
	"""
	def __init__(self, cv_mode=False):
		self.cv_mode = cv_mode

	# def xgb_tree(self, x_train, y_train, params, cv_mode=False):
	# 	""" random forest training
	# 		@param:
	# 			x_train, y_train: training data and labels
	# 			params: parameters for rando forest
	# 		@output:
	# 			trained model(in full mode or cv mode)
	# 	"""
	# 	xg = xgb_model(seed=SEED, params=xgb_params)
	# 	if self.cv_mode:
	# 		cv_scores, model = self.run_cv_model(xg, x_train, y_train),
	# 		print("cv_scores is {}".format(np.mean(cv_scores)))
	# 	else:
	# 		model = self.run_full_model(xg, x_train, y_train)
	# 	return model

	def random_forest(self, x_train, y_train, params):
		""" random forest training
			@param:
				x_train, y_train: training data and labels
				params: parameters for rando forest
			@output:
				trained model(in full mode or cv mode)
		"""
		# create a sklearn wrapper
		rf = sklearn_model(model=RandomForestClassifier, params=params)
		if self.cv_mode:
			cv_scores, model = self.run_cv_model(rf, x_train, y_train),
			print("cv_scores is {}".format(np.mean(cv_scores)))
		else:
			model = self.run_full_model(rf, x_train, y_train)
		return model

	def logistic_regression(self, x_train, y_train_labels, params):
		""" logistic_regression training
		"""
		lr = sklearn_model(model=LogisticRegression, params=params)
		model = self.run_full_model(lr, x_train, y_train_labels)
		return model

	def sgd_svc(self, x_train, y_train_labels, params):
		""" logistic_regression training
		"""
		svc = sklearn_model(model=SGDClassifier, params=params)
		model = self.run_full_model(svc, x_train, y_train_labels)
		return model


	def run_cv_model(self, model, x_train, y_train, NFOLDS=5):
		""" train a model
			@param:
				model: model wrapper
				x_train, y_train: training data and labels
				NFOLDS: cross validation number
			@output:
				trained model and cvscore list
		"""
		kfolds = KFold(x_train.shape[0], n_folds=NFOLDS, shuffle=True, random_state=2017)
		cv_scores = []
		for i, (tr_index, te_index) in enumerate(kfolds):
			x_tr = x_train[tr_index]
			y_tr = y_train[tr_index]
			x_te = x_train[te_index]
			y_te = y_train[te_index]
			print("{} fold trianing".format(i+1))
			start_time = time.time()
			model.train(x_tr, y_tr)
			elapsed = time.time()-start_time
			print("{} seconds elapsed".format(elapsed))
			print(" ")
			print("{} fold predicting".format(i+1))
			start_time = time.time()
			pred = model.predict(x_te)
			elapsed = time.time()-start_time
			print("{} seconds elapsed".format(elapsed))
			cv_scores.append(log_loss(y_te, pred))
			break
		return cv_scores, model

	def run_full_model(self, model, x_train, y_train):
		""" train a model
			@param:
				model: model wrapper
				x_train, y_train: training data and labels
			@output:
				trained model
		"""
		print("Starting Training...")
		start_time = time.time()
		model.train(x_train, y_train)
		elapsed = time.time()-start_time
		print("{} seconds elapsed".format(elapsed))
		return model

	def test_model(self, model,lbl, snrs, test_idx, X_test, Y_test, classes, show=False):
		""" test model on testing data, for model with probability output
		"""
		acc = {}
		accs = []
		for snr in snrs:
			test_SNRs = map(lambda x: lbl[x][1], test_idx)
			test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
			test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    
			test_Y_i_hat = model.predict(test_X_i)
			conf = np.zeros([len(classes),len(classes)])
			confnorm = np.zeros([len(classes),len(classes)])
			for i in range(0,test_X_i.shape[0]):
				j = list(test_Y_i[i,:]).index(1)
				k = int(np.argmax(test_Y_i_hat[i,:]))
				conf[j,k] = conf[j,k] + 1

			for i in range(0,len(classes)):
				confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
			if show:
				plt.figure()
				plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
				plt.savefig('confusion_matrix_' + str(snr) + '.pdf')
			cor = np.sum(np.diag(conf))
			ncor = np.sum(conf) - cor
			print("SNR: " + str(snr) + "; Overall Accuracy: " + str(cor / (cor+ncor)))
			acc[snr] = 1.0*cor/(cor+ncor)
			accs.append(acc[snr])
		return accs, acc

	def test_model_label(self, model, lbl, snrs, test_idx, X_test, Y_test_labels, classes, show=False):
		""" test model on testing data, for model with label output
		"""
		acc = {}
		accs = []
		for snr in snrs:
			test_SNRs = map(lambda x: lbl[x][1], test_idx)
			test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
			test_Y_i = Y_test_labels[np.where(np.array(test_SNRs)==snr)]    
			test_Y_i_hat = model.predict(test_X_i)
			conf = np.zeros([len(classes),len(classes)])
			confnorm = np.zeros([len(classes),len(classes)])
			for i in range(0,test_X_i.shape[0]):
				# j = list(test_Y_i[i,:]).index(1)
				j = test_Y_i[i]
				# k = int(np.argmax(test_Y_i_hat[i,:]))
				k = test_Y_i_hat[i]
				conf[j,k] = conf[j,k] + 1

			for i in range(0,len(classes)):
				confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
			if show:
				plt.figure()
				plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
				plt.savefig('confusion_matrix_' + str(snr) + '.pdf')
			cor = np.sum(np.diag(conf))
			ncor = np.sum(conf) - cor
			print("SNR: " + str(snr) + "; Overall Accuracy: " + str(cor / (cor+ncor)))
			acc[snr] = 1.0*cor/(cor+ncor)
			accs.append(acc[snr])
		return accs, acc