import numpy as np
import hickle as hkl
import keras
from keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D, ConvLSTM2D, BatchNormalization, concatenate, Dropout
from keras.layers.advanced_activations import PReLU
from keras.models import Model
from keras import backend as K
from keras.models import load_model

from PyCA.Core import *
import PyCA.Core as ca
import PyCA.Common as common
import PyCA.Display as display

import matplotlib.pyplot as plt
import os, errno
from scipy.misc import imread, imsave



X1=hkl.load('X1.hkl') #images
X2=hkl.load('X2.hkl') #momentum sequences
y=hkl.load('y.hkl')   #target_momentums
test=hkl.load('test.hkl') #test images

def encoder(mode): #mode=1 for images, mode=3 for momentums
  #cnn network
	layer0=Input(shape=(64,64,mode))
	layer=Conv2D(32,(3,3), padding='same')(layer0)
	layer=PReLU()(layer)
	layer=Conv2D(32,(3,3), padding='same')(layer)
	layer=PReLU()(layer)
	layer=Dropout(0.5)(layer)
	layer=MaxPooling2D((2,2), padding='same')(layer)
	layer=Conv2D(64,(3,3), padding='same')(layer)
	layer=PReLU()(layer)
	layer=Conv2D(64,(3,3), padding='same')(layer)
	layer=PReLU()(layer)
	layer=Dropout(0.5)(layer)
	layer=MaxPooling2D((2,2), padding='same')(layer)
	encoder=Conv2D(128,(3,3), padding='same', name="encoder")(layer)
	layer=PReLU()(layer)
	layer=UpSampling2D((2,2), name="dec1")(encoder)
	layer=Conv2D(64,(3,3), padding='same', name="dec2")(layer)
	layer=PReLU()(layer)
	layer=Conv2D(64,(3,3), padding='same', name="dec3")(layer)
	layer=PReLU()(layer)
	layer=Dropout(0.5)(layer)
	layer=UpSampling2D((2,2), name="dec4")(layer)
	layer=Conv2D(32,(3,3), padding='same', name="dec5")(layer)
	layer=PReLU()(layer)
	layer=Conv2D(32,(3,3), padding='same', name="dec6")(layer)
	layer=PReLU()(layer)
	layer=Dropout(0.5)(layer)
	layer=Conv2D(mode,(3,3), padding='same', name="dec7")(layer)
	output0=PReLU()(layer)

	model = keras.models.Model(input=layer0, output=output0)
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
	if (mode==3):
		model.fit(X2[:,1,:,:,:],X2[:,1,:,:,:], shuffle=True, validation_split=0.10, epochs=250, batch_size=100)
		model.save('main.h5')
		return 
    
	model.fit(X1,X1,shuffle=True,validation_split=0.10,epochs=250, batch_size=100)
	im=np.asarray(model.predict(X1[:1]))
	imsave('inter.png',np.squeeze(im))

	intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('encoder').output)
	o1=np.asarray(intermediate_layer_model.predict(X1))
	intermediate_layer_model.save('enc.h5')
	
  
  #lstm network
	img_inp=Input(shape=(16, 16, 128))
	lstm0 = Input(shape=(None, 64, 64, 3))
	lstm = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', activation=PReLU(), return_sequences=True) (lstm0)
	lstm = BatchNormalization() (lstm)
	lstm = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', activation=PReLU(), return_sequences=True) (lstm)
	lstm = BatchNormalization() (lstm)
	lstm = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', activation=PReLU(), return_sequences=True) (lstm)
	lstm = BatchNormalization() (lstm)
	lstm = ConvLSTM2D(filters=64, kernel_size=(2, 2), activation=PReLU(), padding='same') (lstm)
	lstm = BatchNormalization() (lstm)
	lstm = MaxPooling2D((2,2), padding='same')(lstm)
	lstm = Conv2D(128,(3,3), padding='same')(lstm)
	lstm=PReLU()(lstm)
	lstm_out = MaxPooling2D((2,2), padding='same')(lstm)


	dec=concatenate([img_inp,lstm_out])
	dec=UpSampling2D((2,2))(dec)
	dec=Conv2D(64,(3,3), padding='same')(dec)
	dec=PReLU()(dec)
	dec=Conv2D(64,(3,3), padding='same', name="dec3")(dec)
	dec=PReLU()(dec)
 	dec=Dropout(0.5)(dec)
	dec=UpSampling2D((2,2), name="dec4")(dec)
	dec=Conv2D(32,(3,3), padding='same', name="dec5")(dec)
	dec=PReLU()(dec)
	dec=Conv2D(32,(3,3), padding='same', name="dec6")(dec)
	dec=PReLU()(dec)
	dec=Dropout(0.5)(dec)
	dec=Conv2D(3,(3,3), padding='same', name="dec7")(dec)
	dec_out=PReLU()(dec)

	#dec_model=load_model('dec.h5')
	dec_model = keras.models.Model(input=[img_inp,lstm0], output=dec_out)
	dec_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
	dec_model.load_weights('main.h5', by_name=True)
	dec_model.fit([o1,X2],y,shuffle=True,validation_split=0.10,epochs=1500, batch_size=30)
	#dec_model.save('dec.h5')
	
	#dec_model=load_model('dec.h5')
	#intermediate_layer_model=load_model('enc.h5')

  #preparing test set momentums - numpy arrays of zeros
  a=np.zeros((64,64,3)).tolist()
  out2=[]
	out1=[]
	out=[]
	for i in range(0,4):
		out.append(a)
  out1.append(out)

	test_images=np.asarray(intermediate_layer_model.predict(test))

  #making predictions
  for i in range(1,5):
  	features=np.asarray(dec_model.predict([test[:1],np.asarray(out1)]))
		if(i!=4):
      out1[0][i]=np.squeeze(features)
		  out2.append(np.squeeze(features))
		else:
			out2.append(np.squeeze(features))
	
	return np.asarray(out2)



if __name__ == '__main__':
	if GetNumberOfCUDADevices() > 0:
        	mType = MEM_DEVICE
    	else:
        	print "No CUDA devices found, running on CPU"
        	mType = MEM_HOST

	momentum_cnn = encoder(3)
	outputs=encoder(1)
	hkl.dump(outputs,'outputs.hkl')
	
