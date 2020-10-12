#!/usr/bin/python
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pickle as pk
import numpy as np
mmscaler = preprocessing.MinMaxScaler()

pca_reload = pk.load(open("../../../../pca.pkl",'rb'))
mmscaler_reload = pk.load(open("../../../../mmscaler.pkl", 'rb'))
np.set_printoptions(suppress=True)
try:
    while True:
        frame = np.asarray(input().split(), dtype=float)
        pca_frame = pca_reload.transform(np.expand_dims(frame, 0))
        mm_frame = mmscaler_reload.transform(pca_frame)
        print(str(mm_frame).replace("[[","").replace("]]","").replace("\n",""))
except EOFError as e:
    pass




#To run on multiple files

#for f in ./*
#do
#cat $f | python3 apply_pca.py >> ../phrase_data_pca/${f:1}
#done
