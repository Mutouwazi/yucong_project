import h5py
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from label import labeling
from sklearn.manifold import TSNE
import random

V1_data = h5py.File('./data/V1.mat')
V1_data = V1_data['G4_RspAvgOFFListTotal_Exp1BT_IsoR_RFfixed'][:]
#(12, 6, 3, 4, 735)

V1_matrix = V1_data.reshape(12*6*3*4,735).T
'''
for i in [288,302,312,315,320,351,359,650,683,686,706]:
    V1_matrix[:,i] = 0
'''
V4_data = h5py.File('./data/V4.mat')
V4_data = V4_data['G4_RspAvgOFFListTotal_Exp1BT_IsoR_RFfixed'][:]
#(12, 4, 3, 4, 316)
V4_matrix = V4_data.reshape(12*4*3*4,316).T


data = 'V4'
if data == 'V1':
    '''
    U,sigma,V = np.linalg.svd(V1_matrix)
    V_1 = V[0,:]*sigma[0]
    V_2 = V[1,:]*sigma[1]
    
    data_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(V1_matrix.T)
    V_1 = data_embedded[:,0]
    V_2 = data_embedded[:,1]

    colordict = {'a': 'blue', 'b': 'orange', 'c': 'green', 'd': 'black', 'e': 'red' , 'z':'white'}
    alphadict = {'a': 1, 'b': 1, 'c': 1, 'd': 0, 'e': 1 , 'z': 0}
    oridict = {0:'.', 1:'*', 2:'x', 3:'^'}
    for i in range(12*6*3*4):
        l,orientation = labeling(i,data)
        plt.scatter(V_1[i],V_2[i], c = colordict[l], label=l, alpha = alphadict[l], marker = oridict[orientation])
    #plt.legend()
    plt.savefig('./figure/V1_svd.jpg')
    plt.show()
    '''
    neuron = range(735)
    neuron_sample = random.sample(neuron, 316)
    b = V1_data[:, 1, 0, :, neuron_sample].reshape((12 * 4, 316))
    a = V1_data[:, 3, 0, :, neuron_sample].reshape((12 * 4, 316))
    e = V1_data[:, 4:6, 0, :, neuron_sample].reshape((12 * 8, 316))
    c = V1_data[:, 1, 1, :, neuron_sample].reshape((12 * 4, 316))

    ba = np.append(b, a, axis=0)
    ce = np.append(c, e, axis=0)
    X = np.append(ba, ce, axis=0)
    #(240,316)
    Y = np.array([0] * 48 + [0] * 48 + [0] * 96 + [-1] *48)

    U, sigma, V = np.linalg.svd(X.T)
    print(V.shape)
    for i in range(240):
        V[i, :] = V[i, :] * sigma[i]

    acc = np.ones((240,1))
    tmp = 0
    for pcnum in range(240):
        pcnum = pcnum + 1
        feature = V[0:pcnum,:].T
        clf = LinearSVC(C=10 ** 9,max_iter=100000)
        clf.fit(feature, Y)
        tmp = max(tmp,clf.score(feature, Y))
        acc[pcnum-1] = tmp
        if tmp == 1:
            w = clf.coef_[0]
            acc[pcnum - 1] = 2/np.sum(w*w)
        print(pcnum)
    plt.plot(np.linspace(0,240,240),acc)
    plt.savefig('./figure/V1_svd_acc_c_abe.jpg')
    plt.show()
else:
    '''   
    U,sigma,V = np.linalg.svd(V4_matrix)
    V_1 = V[0,:]*sigma[0]
    V_2 = V[1,:]*sigma[1]
    V_3 = V[2,:]*sigma[2]
    
    #data_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(V1_matrix.T)
    #V_1 = data_embedded[:,0]
    #V_2 = data_embedded[:,1]
    
    colordict = {'a': 'blue', 'b': 'orange', 'c': 'green', 'd': 'black', 'e': 'red' , 'z':'white'}
    alphadict = {'a': 1, 'b': 1, 'c': 1, 'd': 0, 'e': 1 , 'z': 0}
    markerdict = {0:'.', 1:'*', 2:'x', 3:'^'}

    plt.figure()
    #ax3d = plt.gca(projection="3d")
    for i in range(12*4*3*4):
        l,orientation = labeling(i,data)
        plt.scatter(V_1[i],V_2[i], c = colordict[l], alpha = alphadict[l], marker = markerdict[orientation])
        #ax3d.scatter(V_1[i],V_2[i],V_3[i], c = colordict[l], alpha = alphadict[l], marker = markerdict[orientation])
    #plt.legend()
    plt.savefig('./figure/V4_svd.jpg')
    plt.show()
    '''
    b = V4_data[:, 0, 0, :, :].reshape((12 * 4, 316))
    a = V4_data[:, 1, 0, :, :].reshape((12 * 4, 316))
    e = V4_data[:, 2:4, 0, :, :].reshape((12 * 8, 316))
    c = V4_data[:, 1, 1, :, :].reshape((12 * 4, 316))

    ba = np.append(b, a, axis=0)
    ce = np.append(c, e, axis=0)
    X = np.append(ba, ce, axis=0)
    #(240,316)
    Y = np.array([0] * 48 + [0] * 48 + [0] * 96 + [-1] * 48)

    U, sigma, V = np.linalg.svd(X.T)
    print(V.shape)
    for i in range(240):
        V[i, :] = V[i, :] * sigma[i]

    acc = np.ones((240,1))
    tmp = 0
    for pcnum in range(240):
        pcnum = pcnum + 1
        feature = V[0:pcnum,:].T
        clf = LinearSVC(C=10 ** 9,max_iter=1000000)
        clf.fit(feature, Y)
        tmp = max(tmp,clf.score(feature, Y))
        acc[pcnum-1] = tmp
        if tmp == 1:
            w = clf.coef_[0]
            acc[pcnum - 1] = 2/np.sum(w*w)
        print(pcnum)
    plt.plot(np.linspace(0,240,240),acc)
    plt.savefig('./figure/V4_svd_acc_c_abe.jpg')
    plt.show()


