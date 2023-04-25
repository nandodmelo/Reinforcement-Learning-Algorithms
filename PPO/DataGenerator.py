import pandas as pd
import os
os.environ['PYTHONHASHSEED']=str(1000)

from binance.client import Client
import datetime
import pandas as pd
import numpy as np
from indicators import AddIndicators
import time
#from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
#import tensorflow as tf
#tf.random.set_seed(42)
#from tensorflow.keras.layers import Input, Dense,Conv1D, MaxPooling1D,  UpSampling1D, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
#from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.losses import BinaryCrossentropy
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.layers.experimental.preprocessing import Normalization

np.random.seed(32)

api_key = "zSO4fSqoc2GigjvOnDqy0GHbRqX10DLeDKer5kDG8g73qx2eqpuVpnQUYV5Ns8UY"
api_secret = "5yc8vMJa1lewkj6l2y3Q20TqKbxzs46ZWCChZ1UB4XL7eMIf8yULkDLylVvgDiC9"

client=Client(api_key,api_secret)


timestamp = client._get_earliest_valid_timestamp('BTCUSDT', '1m')
#print(df.head())



class Data_Generator:
	def __init__(self, list_of_stocks, frequency, client, start_date):
		self.m =  len(list_of_stocks)
		self.stocks = list_of_stocks
		self.freq = frequency
		self.start = start_date
		self.end = '1654048800'
		print('self.start',self.start)

	def GetStocks(self, indicators=False):
		df_final = pd.DataFrame()
		Data = []
		df_final = df = pd.DataFrame()
		for i in self.stocks:
			print('Download Data for:', i)
			bars = client.get_historical_klines(i, self.freq, self.start, self.end, limit=1000)
			for line in bars:
				del line[6:]
			
			


			#print('CARAOLHOS', type(bars))
			if indicators == True:
				df = pd.DataFrame(bars,columns=['Date','Open','Close','Low','High','Volume'], dtype='float')

				print('indicators', df.tail(1))	
				for i in df.columns:
					#print(i,df[i].mean())
					df[i] = df[i].replace(0, df[i].mean())
				#df['tick'] = i
				df['Date']= pd.to_datetime(df['Date'], unit='ms')
				print('tailHead',df.head(1), df.tail(1))
				df = df.sort_values('Date')
				#print('info', df.info())
				df = AddIndicators(df)
				
				#print('isnan', df.isnull().values.any())
				#print('isnan22', df[99:].isnull().values.any())
				#print('CABECA',df[99:].head(),df.tail())
				bars = df[99:].values
			
			Data.append(bars)

		return  Data



	def AutoEncode(self, X, X_val, X_final_norm, type='decoder',model = '', **kwargs):
		
		X = np.concatenate([X[:,:,i] for i in range(0,X.shape[2])], axis=0)
		
		X_val = np.concatenate([X_val[:,:,i] for i in range(0,X_val.shape[2])], axis=0)
		X = np.expand_dims(X,axis=2)
		X_val = np.expand_dims(X_val,axis=2)
		print('Shape Encoder',X.shape, X_val.shape)
		window_length = X.shape[1]
		if model == 'CNN':
			input_window = Input(shape=(window_length,1))
			x = Conv1D(16, 3, activation="relu", padding="same")(input_window) # 10 dims
			#x = BatchNormalization()(x)
			x = MaxPooling1D(2, padding="same")(x) # 5 dims
			x = Conv1D(1, 3, activation="relu", padding="same")(x) # 5 dims
			#x = BatchNormalization()(x)
			encoded = MaxPooling1D(2, padding="same")(x) # 3 dims

			encoder = Model(input_window, encoded)

			# 3 dimensions in the encoded layer

			x = Conv1D(1, 3, activation="relu", padding="same")(encoded) # 3 dims
			#x = BatchNormalization()(x)
			x = UpSampling1D(2)(x) # 6 dims
			x = Conv1D(16, 1, activation='relu')(x) # 5 dims
			#x = BatchNormalization()(x)
			x = UpSampling1D(2)(x) # 10 dims
			decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x) # 10 dims
			autoencoder = Model(input_window, decoded)
			autoencoder.summary()

			autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
			history = autoencoder.fit(X, X,
							**kwargs,
							validation_data=(X_val, X_val))

			#print('a')
		else:
			
			
			input_window = Input(shape=(window_length,))
				# "encoded" is the encoded representation of the input


			encoded = BatchNormalization()(input_window)
			encoded = GaussianNoise(0.05)(encoded)
			encoded = Dense(6,activation='relu')(encoded)
			encoded = BatchNormalization()(encoded)
			encoded = Activation('swish')(encoded)


				# "decoded" is the lossy reconstruction of the inpu

			decoded = Dropout(0.2)(encoded)
			decoded = Dense(window_length,name='decoded')(decoded)




				# this model maps an input to its reconstruction
			autoencoder = Model(input_window,decoded)


				# this model maps an input to its encoded representation
			encoder = Model(input_window, encoded)

			
			autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
			print('Training...')
			history = autoencoder.fit(X,X,
							**kwargs,
							validation_data=(X_val,X_val))



		if type == 'decoder':
			X_final_norm = 	[np.stack(np.append(X_final_norm[:,:,i], autoencoder.predict(X_final_norm[:,:,i]),axis=1 )) for i in range(0, X_final_norm.shape[2])]
			X_final_norm = np.moveaxis(X_final_norm, 0, -1)


		elif type == 'encoder':
			
			X_final_norm = 	[np.stack(np.append(X_final_norm[:,:,i], encoder.predict(X_final_norm[:,:,i]),axis=1 )) for i in range(0, X_final_norm.shape[2])]
			X_final_norm = np.moveaxis(X_final_norm, 0, -1)





		return X_final_norm

	def normalize(self, x):
		
		for i in range(0,x.shape[2]):
			start = time.time()

			for j in range(0, x.shape[1]):
				
				# Logging and Differencing


				x[:,j,i] = np.where(x[:,j,i] == 0, np.mean(x[:,j,i]), x[:,j,i])
				#x[:,j,i] = np.diff(np.log(x[:,j,i]),prepend=np.log(x[:,j,i])[0])

				if np.isnan(np.sum(np.diff(np.log(x[:,j,i]),prepend=np.log(x[:,j,i])[0]))):
					
					x[:,j,i] = np.diff(x[:,j,i],prepend=x[:,j,i][0])
				else:
					
					x[:,j,i] = np.diff(np.log(x[:,j,i]),prepend=np.log(x[:,j,i])[0])
				# Min Max Scaler implemented
				Min = np.amin(x[:,j,i])


				Max = np.amax(x[:,j,i])

				#print('MinMax',i,j,Min, Max,type(x[:,j,i]))
				x[:,j,i] = (x[:,j,i] - Min) / (Max - Min)
			end = time.time()
			print('Time',end-start)
		return x[1:,:,:]


		



	def formatData(self,data, addBtc):
		# (symbols x timesteps x features) --> (timesteps x features x symbols)
		
		features = list(range(1, np.array(data).shape[2]))
		
		fData = []
		print('pre-formatted data shape: ' + str(np.array(data).shape))
		for i, _ in enumerate(data[0]):
			
			stepData = []
			for idx in features:
				#print('IDXXXX', idx)
				try:
					featVec = [data[j][i][idx] for j, _ in enumerate(data)]

					#print('AQUI',i,idx,featVec)
					
					#print('featVec',featVec)
					stepData.append(np.insert(featVec, 0, 1.) if addBtc else featVec)
					#print('featVec2',stepData)
				except Exception:
					print('Exception occured with (i, idx) = (' + str(i) + ', ' + str(idx) + ')')
					raise
			fData.append(stepData)
		fData= np.squeeze(np.array([fData]), axis=0)
		#print(fData)


		#print('NAN',[ num for num in list(np.isnan(fData))])
		#print(np.isnan(fData))
		return fData
		


	

		
	def formatDataForInput(self, data, window):
		x = []
		y = []
		rates = []
		for i in range(window, len(data)):
			#print('++++++++++++++++TTTTTTTTTTTT+++',type(data), len(data))
			stepData = []
			for j, _ in enumerate(data[i]):
				print('data I',j, data[i], i )				
				#print('que porra é essaJ',j,data[i - 1][2],(i - window, i ))
	#			print('que porra é essa2',np.divide(data[k][j], data[i - 1][2]))
	# j= 0 - i = step = 50 , window = 50 - k = 0 - RANGE de (0 a 50)
	# j= 0 - i = step = 50 , window = 50 - k = 0 - RANGE de (1 a 50)
	#nesse passo do capiroto vc está normalizando pelo último valor da Janela
				stepData.append([data[k][j] for k in range(i - window, i)])
			x.append(stepData)
			#print('yyyyyyyyyyy',data[i - 1][2], data[i - 2][2], np.divide(data[i - 1][2], data[i - 2][2]))

			y.append(np.divide(data[i - 1][2], data[i - 2][2]))
			rates.append(data[i][2])
		
		return x, y, rates        


#print(b[b['stock'] == 'BNBBTC'].head(), b[b['stock'] == 'BNBBTC'].tail())

#print(b[b['stock'] == 'BTCUSDT'].head(), b[b['stock'] == 'BTCUSDT'].tail())

