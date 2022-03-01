import h5py
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


V4_data = h5py.File('./data/V4.mat')
V4_data = V4_data['G4_RspAvgOFFListTotal_Exp1BT_IsoR_RFfixed'][:]
print(V4_data.shape)
#(12, 4, 3, 4, 316)


#X1 = V4_data[:, 1, 0, 0, :]
#X2 = V4_data[:, 1, 1, 0, :]
b = V4_data[:, 0, 0, :, :].reshape((12*4,316))
a = V4_data[:, 1, 0, :, :].reshape((12*4,316))
c = V4_data[:, 2:4, 0, :, :].reshape((12*8,316))
e = V4_data[:, 1, 1, :, :].reshape((12*4,316))

ba = np.append(b,a,axis = 0)
ce = np.append(c,e,axis = 0)
X = np.append(ba,ce,axis = 0)
Y = np.array([0]*48+[0]*48+[-1]*(96+48))
'''
clf = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-5))
clf.fit(X,Y)
w = clf.named_steps['linearsvc'].coef_
b = clf.named_steps['linearsvc'].intercept_
acc = clf.score(X,Y)
print('acc=',acc)
'''
clf = LinearSVC(C=10**9)
clf.fit(X,Y)
w = clf.coef_[0]
b = clf.intercept_[0]
acc = clf.score(X,Y)
print('acc=',acc)

print('for_ori')
a = []

for i in range(4):
    tmp = clf.predict(V4_data[:,1,0,i,:])
    a.append(tmp)
print(np.array(a))

a = []
for i in range(4):
    tmp = clf.predict(V4_data[:,1,1,i,:])
    a.append(tmp)
print(np.array(a))

print(w.shape)
print(b.shape)
print(np.matmul(w,V4_data[:, 1, 0, 0, :].T)+b)
print(np.matmul(w,V4_data[:, 1, 1, 0, :].T)+b)

#plt.scatter(np.linspace(0,316,316),w)
#plt.show()
'''
V4_ori = h5py.File('./data/ori_pre.mat')
ori_pre = V4_ori['ori_pre'][:]

attention = []
for i in range(316):
    if w[0,i]> 0.05:
        attention.append(ori_pre[i][0])
count = np.linspace(22.5,180,8)
print(attention)
plt.hist(attention, bins = 8)
plt.savefig('./figure/V4_attribution_dis.jpg')
plt.show()
'''