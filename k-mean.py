import pandas as pd
from kneed import KneeLocator
from pandas import read_csv
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

dataset = read_csv('water-treatment.data', header = None)
Statistics = read_csv('data.csv', header=None)

wtd = pd.DataFrame(dataset)
Xs = pd.DataFrame(dataset)
ss = pd.DataFrame(Statistics)  #data constrains

print(wtd)
for i in range(1, 39):
    minv = float(ss.get_value(i-1, 1, takeable=True))
    maxv = float(ss.get_value(i-1, 2, takeable=True))

    for j in range(0, 527):
        val = (wtd.get_value(j, i, takeable = True))
        if(val == "?"):
            mean = float(ss.get_value(i-1, 3, takeable=True))
            wtd.set_value(j, i, mean)                    # replace the  empty vlaue by mean
            val = mean
        val = float(val)

        normalized = (val - minv) / (maxv - minv)     #normalize function : normalized(ai) = (ai-min)/(max-min)
        wtd.set_value(j, i, normalized)



wtd = wtd.iloc[:,1:]        #do not use the date column
distortions = []
K = range(1, 20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(wtd)
    kmeanModel.fit(wtd)
    distortions.append(sum(np.min(cdist(wtd, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / wtd.shape[0])

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

print("the k value is= 7", "with Distortion: ", distortions[7])

cluster_map = pd.DataFrame()
cluster_map['data_index'] = wtd.index.values
kmeanModel = KMeans(n_clusters=7).fit(wtd)
cluster_map['cluster'] = kmeanModel.labels_+1

with open('output', 'w') as f:
    print(cluster_map.iloc[1:, 1:].to_string(), file=f)

dfp=wtd.iloc[:, 1:]        #do not use the date column
pca = PCA(n_components=0.95)
dfp = pca.fit_transform(dfp)
dfp = pd.DataFrame(dfp)
print("the shape of data after pca is: ", dfp.shape)
print(dfp)

distortions = []
K = range(1, 20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(dfp)
    kmeanModel.fit(dfp)
    distortions.append(sum(np.min(cdist(dfp, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / dfp.shape[0])

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k with PCA')
plt.show()

kn = KneeLocator(K, distortions, curve='convex', direction='decreasing')          # use that to find the elbow point
kvalue=kn.knee
print("the k value is= ", kvalue, "with Distortion: ", distortions[kvalue])


cluster_map = pd.DataFrame()
cluster_map['data_index'] = dfp.index.values
kmeanModel = KMeans(n_clusters=kvalue).fit(dfp)
cluster_map['cluster'] = kmeanModel.labels_+1




with open('outputwithPCA', 'w') as f:
    print(cluster_map.iloc[1:, 1:].to_string(), file=f)