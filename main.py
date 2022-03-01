import scipy.io as scio
#data = scio.loadmat('./data/V1.mat')

import h5py
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


a = [2,2,2,3,3,3,3,4,4,5,6]
plt.hist(a,bins = 3)
plt.show()


# 输出的结果
# test_info_1
# test_info_2

#(12, 6, 3, 4, 735)

'''

X1 = V1_data[:, 1, 0, :, :].reshape((12*4,735))
#X2 = V1_data[:, 5, 0, :, :].reshape((12*4,735))
#print(X1[4,:]-V1_data[1, 1, 0, 0, :])
X2 = V1_data[:,1,1,:,:].reshape((12*4,735))
X = np.append(X1,X2,axis=0)
Y = np.array([1]*48+[0]*48)

clf = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-5))
clf.fit(X,Y)
w = clf.named_steps['linearsvc'].coef_
b = clf.named_steps['linearsvc'].intercept_
print(b)

print('six_loc')
a = []
for i in range(6):
    tmp = clf.predict(V1_data[:,i,0,0,:])
    a.append(tmp)
print(np.array(a))


print('all_none_figure')
c = []
for i in range(4):
    tmp = clf.predict(V1_data[:,1,1,i,:])
    c.append(tmp)
print(np.array(c))

plt.scatter(np.linspace(0,735,735),w)
#plt.show()
'''