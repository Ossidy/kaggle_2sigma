from keras.models import Model 
from keras.layers import Input, Dense, Activation, Dropout
from keras.optimizers import *

class NN_model():
	def __init__(self, 
				 train_x, train_y, 
				 learning_rate = 0.001, 
				 optimizer = 'RMSprop', 
				 weight_decay = 0.0, 
				 dropout = 0.5,
				 momentum = 0.9,
				 num_class = 3,
				 loss = 'categorical_crossentropy',
				 batch_size = 128,
				 num_epoch = 100)

		self.train_x = train_x
		self.train_y = train_y
		self.num_class = num_class
		self.dim = train_x.shape


		self.learning_rate = learning_rate
		self.optimizer = optimizer
		self.weight_decay = weight_decay
		self.dropout = dropout
		self.momentum = momentum

		self.loss = loss
		self.batch_size = batch_size
		self.num_epoch = num_epoch


	def model_1(self, dim, num_class, drop_prob = 0.5):
		inputs = Input(shape = dim)
		
		x = Dense(128, activation = 'relu')(inputs)
		x = Dense(128, activation = 'relu')(x)
		x = Dropout(drop_prob)(x)
		x = Dense(64, activation = 'relu')(x)
		x = Dense(10, activation = 'relu')(x)
		predictions = Dense(3, activation = 'softmax')(x)
		model = Model(input = inputs, output = predictions)
		
		return model

	def train_model(self, model):
		if self.optimizer == "RMSprop":
			optimizer = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-08, decay=self.weight_decay)
		elif self.optimizer == "Adam":
			optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=self.weight_decay)
		else:
			raise("not implemented")

		model.compile(optimizer = optimizer, loss = self.loss, metrics = None)
		model.fit(self.train_x, self.train_y, batch_size = self.batch_size, nb_epoch = self.num_epoch)
		scores = model.evaluate(self.train_x, self.train_y)

		return model 

	def predict_model(self, predict_x):
		predictions = model.predict(predict_x)
		return predictions

	

