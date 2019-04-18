import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

from sklearn.datasets.samples_generator import make_blobs


x,y = make_blobs(n_samples=100,centers=3,n_features=2,cluster_std=0.5)

# =============================================================================
# x = np.array([[0.5,2],
#               [0.7,3],
#               [0.2,4],
#               [1,3],
#               [1.3,2.4],
#               [2.4,4],
#               [1,9],
#               [2.4,8],
#               [2.9,11],
#               [3.4,9.7],
#               [11,2],
#               [12,5],
#               [11.4,3],
#               [11.9,5],
#               [13,3]])
# =============================================================================

colors = 10*['g','r','c','b','k']


class MeanShift():
    def __init__(self,radius=None, radius_norm_step=20):
        self.radius = radius
        self.radius_norm_step = radius_norm_step
        
        
    def fit(self,data):
        
        if self.radius == None:
            all_data_centroid = np.average(abs(data),axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm/self.radius_norm_step
            
        
        
        
        centroids = {}
        
        for i in range(len(data)):
            centroids[i] = data[i]
        
        weights = [i for i in range(self.radius_norm_step)[::-1]]
                
        
        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                
                
                for featureset in data:
                    distance = np.linalg.norm(featureset-centroid)
                    if distance == 0:
                        distance = 0.000000001
                    weight_index = int(distance/self.radius)
                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step-1
                    
                    to_add = (weights[weight_index]**2)*[featureset]
                    
                    in_bandwidth += to_add
                
                new_centroid = np.average(in_bandwidth,axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))
            
            to_pop = []
            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius:
                        if [ii,i] in to_pop:
                            pass
                        else:
                            to_pop.append([i,ii])
                        break
                    
            for i in to_pop: 
                try:
                    uniques.remove(i[0])
                except:
                    pass
                
            prev_centroids = dict(centroids)
            
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])
                
            optimized = True
            
            for i in centroids:
                if not np.array_equal(centroids[i],prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
            if optimized:
                break
            
        self.centroids = centroids
        
        self.classifications = {}
        
        for i in range(len(self.centroids)):
            self.classifications[i]=[]
            
        for featureset in data:
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in centroids]
            classification = distances.index(min(distances))
            
            self.classifications[classification].append(featureset)
            
        

    def predict():
        distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in centroids]
        classification = distances.index(min(distances))
        return classification    



clf = MeanShift()

clf.fit(x)


centroids = clf.centroids


for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1],marker='x',color=color,s=150,linewidths=8)
        
    

for c in centroids:
    plt.scatter(centroids[c][0],centroids[c][1],color='k',marker='o',s=150)

plt.show()



























