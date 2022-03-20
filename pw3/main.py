from kmeans import KMeans
from kmeansplusplus import KMeanPlusPlus
import pandas as pd

df = pd.read_csv ('Iris.csv')
df = df.drop(columns = ['Species'])

feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
data = pd.DataFrame(df, columns = feature_columns).to_numpy()

km = KMeans(data, 3)
# km = KMeans(data, 4)
# km = KMeans(data, 5)
km.update()
km.plot_graphs()

kmplus = KMeanPlusPlus(data, 3)
# kmplus = KMeanPlusPlus(data, 4)
# kmplus = KMeanPlusPlus(data, 3)

kmplus.first_centroid()
kmplus.set_next_centroids()
kmplus.centroids
kmplus.set_clusters()

km.plot_graphs()