import numpy as np
import csv
import math

class K_Means:
  def __init__(self, dataset, dataset_size, dimension, tolerance, k=3, max_iterations=5000):
    self.tolerance = tolerance
    self.k = k
    self.max_iterations = max_iterations
    self.dataset = dataset
    self.dataset_size = dataset_size
    self.dimension = dimension
    self.centroids = {}
    self.classes = {}

  def euclidean_distance(self, feat_one, feat_two):
    squared_distance = 0

    for i in range(len(feat_one)):
      squared_distance += (feat_one[i] - feat_two[i])**2
    
    return squared_distance
  
  def calculate_sse(self):
    sse = 0
    for cluster_idx, data_points in self.classes.items():
      means = np.mean(data_points, axis=0)
      for point in data_points:
        sse += self.euclidean_distance(means, point)

    return sse

  def clustering(self):
    for i in range(self.k):
      self.centroids[i] = self.dataset[i]

    for i in range(self.max_iterations):
      self.classes = {}
      for j in range(self.k):
        self.classes[j] = []

      for features in self.dataset:
        distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        self.classes[classification].append(features)

      previous = dict(self.centroids)

      for classification in self.classes:
        self.centroids[classification] = np.average(self.classes[classification], axis=0)

      isOptimal = True

      for centroid in self.centroids:
        original_centroid = previous[centroid]
        curr = self.centroids[centroid]

        if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
          isOptimal = False
      if isOptimal:
        break
