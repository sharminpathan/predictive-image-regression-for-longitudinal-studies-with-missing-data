import hickle as hkl
import numpy as np
import skimage.io as io


mom=hkl.load('mom.hkl') #ground truths from CAvmMatching & CAvmGeodesic Shooting
img=hkl.load('img.hkl') #baseline images

X1=[]
X2=[]
y=[]


for i in range(0,len(mom)):

	for j in range(1,5):
	    if(np.amax(mom[i][j])!=0.0):
	    	inseq=np.asarray(mom[i][:j])
		X2.append(inseq)
		y.append(mom[i][j])
		X1.append(img[i])

	 	
a=np.zeros((64,64,3)).tolist()
# a=np.zeros((128,128,3)).tolist()  #for oasis dataset
for i in range(0,len(X2)):
	X2[i]=X2[i].tolist()
	for j in range(len(X2[i]),4):
		X2[i].append(a)

print np.asarray(X2).shape
print np.asarray(X1).shape
print np.asarray(y).shape

hkl.dump(np.asarray(X1),'X1.hkl') #image data
hkl.dump(np.asarray(X2),'X2.hkl') #vector-momentum sequences
hkl.dump(np.asarray(y),'y.hkl') #target vector-momentum
