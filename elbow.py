from kmeans import *
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name):
  dataset = []
  with open(file_name) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      data = [float(x) for x in row[0].split(" ")]
      dataset.append(data)

  dataset = np.array(dataset)
  dataset_size = len(dataset[0])
  dimension = len(dataset)
  
  dataset = dataset.reshape(dataset_size, dimension)
  return dataset, dataset_size, dimension

if __name__ == "__main__":
  
  dataset, maxK, dimension = load_data("./data.csv")
  sse_results = []

  K = range(1, maxK + 1)

  for k in range(1, maxK + 1):
    km = K_Means(dataset, maxK, dimension, 0.0001, k)
    km.clustering()
    sse_results.append(km.calculate_sse())

  plt.plot(K, sse_results, "bx-")
  plt.xlabel("k")
  plt.ylabel("SSE")
  plt.title("The Elbow method to show optimal of k")
  plt.show()
