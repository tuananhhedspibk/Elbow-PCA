from kmeans import *
from pca import *
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name):
  dataset = []
  with open(file_name) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      row[0] = row[0].strip()
      data = [float(x) for x in row[0].split(" ")]
      dataset.append(data)

  dataset = np.array(dataset)
  dataset_size = len(dataset[0])
  dimension = len(dataset)
  
  dataset = dataset.reshape(dataset_size, dimension)
  return dataset, dataset_size, dimension

if __name__ == "__main__":
  pca = PCA("./input/data.csv")
  mean_vector = pca.calculate_mean_vector()
  scatter_matrix = pca.calculate_scatter_matrix(mean_vector)

  eig_val_sc, eig_vec_sc = pca.calculate_eig(scatter_matrix)
  eig_pair = pca.sort_eig(eig_val_sc, eig_vec_sc)
  pca_k = pca.calculate_pca_k(eig_val_sc)
  transformed_data = pca.transform_data(pca_k, eig_pair)
  pca.save_data("./input/data.csv", transformed_data)

  dataset, maxK, dimension = load_data("./input/data.csv")
  sse_results = []

  K = range(1, maxK + 1)

  for k in K:
    km = K_Means(dataset, maxK, dimension, 0.0001, k)
    km.clustering()
    sse_results.append(km.calculate_sse())

  plt.plot(K, sse_results, "bx-")
  plt.xlabel("k")
  plt.ylabel("SSE")
  plt.title("The Elbow method to show optimal of k")
  plt.show()
