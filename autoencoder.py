import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from kneed import KneeLocator
import numpy as np
from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
import matplotlib.pyplot as plt

dataset = read_csv('water-treatment.data', header=None)
Statistics=read_csv('data.csv', header=None)

wtd = pd.DataFrame(dataset)
ss = pd.DataFrame(Statistics)  #data constrains

print(wtd)
for i in range(1,39):
    minv = float(ss.get_value(i-1 , 1, takeable=True))
    maxv = float(ss.get_value(i-1 , 2, takeable=True))

    for j in range(0, 527):
        val = (wtd.get_value(j, i, takeable=True))
        if(val=="?"):
            mean = float(ss.get_value(i-1, 3, takeable=True))
            wtd.set_value(j, i, mean)                    # replace the  empty vlaue by mean
            val = mean
        val = float(val)

        normalized = (val - minv) / (maxv - minv)     #normalize function : normalized(ai) = (ai-min)/(max-min)
        wtd.set_value(j, i, normalized)

wtd = wtd.iloc[:, 1:]        #do not use the date column
##start the work for autoencoder

Y = wtd.iloc[:, 0:19]
X = wtd
sX = minmax_scale(X, axis=0)  # SCALE EACH FEATURE INTO [0, 1] RANGE
ncol = sX.shape[1]
X_train, X_test, Y_train, Y_test = train_test_split(sX, Y, train_size=0.5, random_state=seed(2019))

input_dim = Input(shape=(ncol, ))
encoding_dim = 23           # DEFINE THE DIMENSION OF ENCODER ASSUMED 23

encoded = Dense(encoding_dim, activation='relu')(input_dim)   #encoder layer
decoded = Dense(ncol, activation='sigmoid')(encoded)          #decoder layer
autoencoder = Model(input=input_dim, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, nb_epoch = 50, batch_size = 100, shuffle = True, validation_data = (X_test, X_test))
encoder = Model(input = input_dim, output = encoded)
encoded_input = Input(shape = (encoding_dim, ))
encoded_out = encoder.predict(X_test)
#print(encoded_out[0:23])
data = pd.DataFrame(encoded_out[:,0:23])

##end the work for autoencoder
distortions = []
K = range(1,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(data)
    kmeanModel.fit(data)
    distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The optimal k with autoencoder')
plt.show()

kn = KneeLocator(K, distortions, curve='convex', direction='decreasing')
kvalue = kn.knee
print("the k value is= ", kvalue, "with Distortion: ", distortions[kvalue])


cluster_map = pd.DataFrame()
cluster_map['data_index'] = data.index.values
kmeanModel = KMeans(n_clusters=kvalue).fit(data)
cluster_map['cluster'] = kmeanModel.labels_+1




with open('outputofautoencoder', 'w') as f:
    print(cluster_map.iloc[1:,1:].to_string(), file=f)
