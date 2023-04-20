#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn import metrics


# In[ ]:


#REMEMBER TO REPLACE THESE FILE PATHS
zoo_path = "/Users/mattiamosconi/Documents/GitHub/zoo_clustering/zoo.csv"
class_path = "/Users/mattiamosconi/Documents/GitHub/zoo_clustering/class.csv"


# In[ ]:


#defining a Manager class we'll use to handle the dataframe
class Manager():
    def __init__(self): #we don't need attributes for Manager objects
        pass
    def new(self, path): #creates a new df by reading a csv file
        new_dataframe = pd.read_csv(path)
        return new_dataframe
    def truth(self, df, target_column): #extracts the target column as a numpy array
        target = df[target_column].to_numpy()
        return target
    def drop(self, df, columns): #drops columns from a pandas df
        for i in columns:
            df.drop(i, axis=1, inplace=True)
    def standardize(self, df): #standardizes a pandas df
        for i in df.columns.values:
            df[i] = (df[i] - df[i].mean() ) / df[i].std() 
    def np(self, df): #converts a pandas df into a numpy array
        df = df.to_numpy()
        return df
    def analyze(self, df, clusters): #adds clusters array to df as the "clusters" column, then groups by cluster and gives a statistical description of the samples in each cluster
        df["cluster"] = clusters
        analysis = df.groupby(["cluster"])
        for v in range(len(np.unique(clusters))): #not all our algorithms give 7 clusters as output
            print("CLUSTER "+str(v+1)+":\n")
            print(analysis.get_group(v+1).describe(percentiles=[]))
            print("\n")
#Manager could have more methods to modify the starting dataframe. We just implemented drop and standardize due to the nature of our dataset


# In[ ]:


#defining an Algorithm class we'll use to implement different clustering algorithms
class Algorithm():
    def __init__(self):
        pass
    def kmodes(self, data):
        kmodes = KModes(n_clusters=7, init = "Cao", n_init = 100, verbose=0).fit(data)
        output = kmodes.predict(data).astype('int64')+1 #+1 because the classes are 1 to 7, while the 7 kmodes clusters will be 0 to 6, so adding 1 provides us with better visualization
        return output
    def kmeans(self, data, rand = None):
        kmeans = KMeans(n_clusters=7, random_state=rand, n_init=10,).fit(data)
        output = kmeans.predict(data).astype('int64')+1
        return output
    def affinity(self, data, rand = None):
        affinity = AffinityPropagation(random_state=rand).fit(data)
        output = affinity.predict(data).astype('int64')+1 
        return output  
    def meanshift(self, data):
        meanshift = MeanShift().fit(data)
        output = meanshift.predict(data).astype('int64')+1
        return output
    def spectral(self, data, rand = None):
        output = SpectralClustering(n_clusters=7, assign_labels='discretize', random_state=rand).fit_predict(data).astype('int64')+1
        return output
    def hierarchical(self, data):
        output = AgglomerativeClustering(n_clusters=7,linkage='complete').fit_predict(data).astype('int64')+1
        return output
    def dbscan(self, data):
        output = DBSCAN().fit_predict(data).astype('int64')+1  
        return output
    def optics(self, data):
        output = OPTICS().fit_predict(data).astype('int64')+2
        return output
    def birch(self, data):
        birch = Birch(n_clusters=7).fit(data)
        output = birch.predict(data).astype('int64')+1 
        return output     


# In[ ]:


#defining a Visualizer class we'll use to visually evaluate models
class Visualizer():
    def __init__(self):
        pass
    def stair(self, target, model_output):
        ordered_target = np.zeros(len(data)).astype('int64') #ordered ground truth. It will store, in increasing order, all the class values in target array
        tindex = np.zeros(len(data)).astype('int64') #truth index. It will store the ordered indexes of the values we moved to ordered_target
        ordered_model_output = np.zeros(len(data)).astype('int64') #it will store the values in clusters array, ordered by the indexes in tindex
        c1 = 0 #counter 1 and 2 will allow us to replace the zeros with our values in the right order
        c2 = 0 
        for n in range(1,8): #update ordered_target and tindex
            for i, v in enumerate(target):
                if v == n:
                    ordered_target[c1]=v
                    tindex[c1]=i
                    c1 += 1
        for j in tindex: #update ordered_model_output
            ordered_model_output[c2]=model_output[j]
            c2 += 1
        return (ordered_target, ordered_model_output)
    def lineplot(self, target, model_output):
        plt.plot(np.c_[target,model_output])
    def tsne(self, data, target, model_output, class_number, class_type, rand=None): #class_number and class_type are df columns
        #TSNE dimensionality reduction will help us plot and visually understand how well our clustering algorithm performed
        data_tsne= TSNE(n_components=2, learning_rate='auto', init='random', random_state=rand).fit_transform(data)
        mapper = dict(zip(class_number, class_type))
        target2 = pd.Series(target)
        plot_df = pd.DataFrame(data_tsne, columns=["tsne1", "tsne2"])
        plot_df["label"] = target2.map(mapper)
        plot_df["cluster"] = model_output.astype(str)
        fig = px.scatter(plot_df, x="tsne1", y="tsne2", color="cluster", hover_name="label")
        fig.show()
        return plot_df
    def update_tsne(self, plot_df, model_output): #this method allows us to use the same tsne plot for all algorithms by just changing the model_output array. An alternative would be to run tsne everytime with a fixed seed, but it would cost us more from a computational point of view
        plot_df["cluster"] = model_output.astype(str)
        fig = px.scatter(plot_df, x="tsne1", y="tsne2", color="cluster", hover_name="label")
        fig.show()


# In[ ]:


m = Manager()
v = Visualizer()
a = Algorithm()


# In[ ]:


#importing the csv files as pandas DataFrames
df = m.new(zoo_path)
classes = m.new(class_path)
#before dropping df.class_type (our ground truth), let's store it in a numpy array called "target"
target = m.truth(df, "class_type")
#dropping the ground truth and the string column we don't need
m.drop(df, ["animal_name", "class_type"])
#transforming the df into a np array
data = m.np(df)
df2 = df.copy(deep=True) #this is a copy of our dataframe we'll need later


# In[ ]:


df


# In[ ]:


target


# In[ ]:


plot = v.tsne(data, target, target, classes["Class_Number"], classes["Class_Type"], rand=0) #we'll use plot variable to update the tsne plot with update_tsne for a better comparison between algorithms


# In[ ]:


m.analyze(df, target)


# In[ ]:


kmodes_output = a.kmodes(data)
kmodes_output


# In[ ]:


x, y = v.stair(target, kmodes_output)
v.lineplot(x, y)


# In[ ]:


v.update_tsne(plot, kmodes_output)


# In[ ]:


scores = np.zeros(9)
s1 = metrics.adjusted_rand_score(target, kmodes_output)
scores[0] = s1
s1


# In[ ]:


m.analyze(df, kmodes_output)


# In[ ]:


#the next algorithms don't work with categorical data, so we'll standardize df2 (keeping the original df for our m.analyze method)
m.standardize(df2)
df2


# In[ ]:


data = m.np(df2)
kmeans_output = a.kmeans(data)
kmeans_output


# In[ ]:


x, y = v.stair(target, kmeans_output)
v.lineplot(x, y)


# In[ ]:


v.update_tsne(plot, kmeans_output)


# In[ ]:


s2 = metrics.adjusted_rand_score(target, kmeans_output)
scores[1] = s2
s2


# In[ ]:


m.analyze(df, kmeans_output)


# In[ ]:


affinity_output = a.affinity(data)
affinity_output


# In[ ]:


x, y = v.stair(target, affinity_output)
v.lineplot(x, y)


# In[ ]:


v.update_tsne(plot, affinity_output)


# In[ ]:


s3 = metrics.adjusted_rand_score(target, affinity_output)
scores[2] = s3
s3


# In[ ]:


if len(np.unique(affinity_output))>1: #this is to prevent an error, in case AffinityPropagation doesn't converge
    m.analyze(df, affinity_output)


# In[ ]:


meanshift_output = a.meanshift(data)
meanshift_output


# In[ ]:


x, y = v.stair(target, meanshift_output)
v.lineplot(x, y)


# In[ ]:


v.update_tsne(plot, meanshift_output)


# In[ ]:


s4 = metrics.adjusted_rand_score(target, meanshift_output)
scores[3] = s4
s4


# In[ ]:


m.analyze(df, meanshift_output)


# In[ ]:


spectral_output = a.spectral(data)
spectral_output


# In[ ]:


x, y = v.stair(target, spectral_output)
v.lineplot(x, y)


# In[ ]:


v.update_tsne(plot, spectral_output)


# In[ ]:


s5 = metrics.adjusted_rand_score(target, spectral_output)
scores[4] = s5
s5


# In[ ]:


m.analyze(df, spectral_output)


# In[ ]:


hierarchical_output = a.hierarchical(data)
hierarchical_output


# In[ ]:


x, y = v.stair(target, hierarchical_output)
v.lineplot(x, y)


# In[ ]:


v.update_tsne(plot, hierarchical_output)


# In[ ]:


s6 = metrics.adjusted_rand_score(target, hierarchical_output)
scores[5] = s6
s6


# In[ ]:


m.analyze(df, hierarchical_output)


# In[ ]:


dbscan_output = a.hierarchical(data)
dbscan_output


# In[ ]:


x, y = v.stair(target, dbscan_output)
v.lineplot(x, y)


# In[ ]:


v.update_tsne(plot, dbscan_output)


# In[ ]:


s7 = metrics.adjusted_rand_score(target, dbscan_output)
scores[6] = s7
s7


# In[ ]:


m.analyze(df, dbscan_output)


# In[ ]:


optics_output = a.optics(data)
optics_output


# In[ ]:


x, y = v.stair(target, optics_output)
v.lineplot(x, y)


# In[ ]:


v.update_tsne(plot, optics_output)


# In[ ]:


s8 = metrics.adjusted_rand_score(target, optics_output)
scores[7] = s8
s8


# In[ ]:


m.analyze(df, optics_output)


# In[ ]:


birch_output = a.birch(data)
birch_output


# In[ ]:


x, y = v.stair(target, birch_output)
v.lineplot(x, y)


# In[ ]:


v.update_tsne(plot, birch_output)


# In[ ]:


s9 = metrics.adjusted_rand_score(target, birch_output)
scores[8] = s9
s9


# In[ ]:


m.analyze(df, birch_output)


# In[ ]:


algorithms = ["KModes", "KMeans", "AffinityPropagation", "MeanShift", "SpectralClustering", 
              "HierarchicalClustering", "DBSCAN", "OPTICS", "Birch"]
score_frame = pd.concat([pd.Series(algorithms), pd.Series(scores)], axis=1)
score_frame.columns = ["algorithm", "adjusted Rand score"]
score_frame

