import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans

matplotlib.style.use('ggplot') # Look Pretty

#
# INFO: This dataset has call records for 10 users tracked over the course of 3 years.
# Your job is to find out where the users likely live at!


def showandtell(title=None):
  if title != None: plt.savefig(title + ".png", bbox_inches='tight', dpi=300)
  plt.show()
  # exit()

def clusterInfo(model):
  print ("Cluster Analysis Inertia: ", model.inertia_)
  print ('------------------------------------------')
  for i in range(len(model.cluster_centers_)):
    print ("\n  Cluster ", i)
    print ("    Centroid ", model.cluster_centers_[i])
    print ("    #Samples ", (model.labels_==i).sum()) # NumPy Power

# Find the cluster with the least # attached nodes
def clusterWithFewestSamples(model):
  # Ensure there's at least on cluster...
  minSamples = len(model.labels_)
  minCluster = 0
  for i in range(len(model.cluster_centers_)):
    if minSamples > (model.labels_==i).sum():
      minCluster = i
      minSamples = (model.labels_==i).sum()
  print ("\n  Cluster With Fewest Samples: ", minCluster)
  return (model.labels_==minCluster)


def doKMeans(data, clusters=0):
  #
  # TODO: Be sure to only feed in Lat and Lon coordinates to the KMeans algo, since none of the other
  # data is suitable for your purposes. Since both Lat and Lon are (approximately) on the same scale,
  # no feature scaling is required. Print out the centroid locations and add them onto your scatter
  # plot. Use a distinguishable marker and color.
  #
  # Hint: Make sure you fit ONLY the coordinates, and in the CORRECT order (lat first). This is part
  # of your domain expertise. Also, *YOU* need to instantiate (and return) the variable named `model`
  # here, which will be a SKLearn K-Means model for this to work.
  #
  # .. your code here ..
  data = data.loc[:,['TowerLat','TowerLon']]
  model = KMeans(n_clusters = clusters)
  model.fit(data)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(model.cluster_centers_[:,1], model.cluster_centers_[:,0], s=169, c='r', marker='x', alpha=0.8, linewidths=2)
  return model




df = pd.read_csv('C:/Users/anshangu/Documents/GitHub/DAT210x/Module5/Datasets/CDR.csv')
df.CallDate = pd.to_datetime(df.CallDate)
df.CallTime = pd.to_timedelta(df.CallTime)
unqlst = df.In.unique()

def forall(i):
    print ("\n\nExamining person: ", 0)
    user1 = df[df.In == unqlst[i]]
    user1 = user1[~user1.DOW.isin(['Sat','Sun'])]
    user1 = user1[user1.CallTime < '17:00:00']
    return user1
  #  fig = plt.figure()
   # ax = fig.add_subplot(111)
    #ax.scatter(user1.TowerLon,user1.TowerLat, c='g', marker='o', alpha=0.2)
    #ax.set_title('Weekday Calls (<5pm)')


for x in range(10):
    model = doKMeans(forall(x), 3)


#
# INFO: Print out the mean CallTime value for the samples belonging to the cluster with the LEAST
# samples attached to it. If our logic is correct, the cluster with the MOST samples will be work.
# The cluster with the 2nd most samples will be home. And the K=3 cluster with the least samples
# should be somewhere in between the two. What time, on average, is the user in between home and
# work, between the midnight and 5pm?
midWayClusterIndices = clusterWithFewestSamples(model)
midWaySamples = user1[midWayClusterIndices]
print ("    Its Waypoint Time: ", midWaySamples.CallTime.mean())


#
# Let's visualize the results!
# First draw the X's for the clusters:
ax.scatter(model.cluster_centers_[:,1], model.cluster_centers_[:,0], s=169, c='r', marker='x', alpha=0.8, linewidths=2)
#
# Then save the results:
showandtell('Weekday Calls Centroids')  # Comment this line out when you're ready to proceed
