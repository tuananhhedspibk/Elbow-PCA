import numpy as np
import csv

class PCA:
  def __init__(self, ipt_data):
    self.dataset = None
    self.dimension = 0

    self.dataset, self.dimension = self.load_data(ipt_data)

  def load_data(self, file_name):
    dataset = []

    f = open(file_name, "r")
    for line in f:
      line = line.strip()
      data = [float(x) for x in line.split(" ")]
      dataset.append(data)

    dataset = np.array(dataset).reshape(len(dataset[0]), len(dataset))
    dimension = len(dataset)

    return dataset, dimension

  def accumulate(self, array, index):
    total = 0.0
    idx_ct = 0
    for item in array:
      idx_ct += 1
      if idx_ct <= index - 1:
        total += item
      else:
        return total
    return total

  def calculate_mean_vector(self):
    vector_elements = []
    for i in range(self.dimension):
      element = np.ndarray(shape=(1, ), buffer=np.array(np.mean(self.dataset[i, :])))
      vector_elements.append(element)

    mean_vector = np.array(vector_elements)
    return mean_vector

  def calculate_scatter_matrix(self, mean_vector):
    scatter_matrix = np.zeros((self.dimension, self.dimension))
    for i in range(self.dataset.shape[1]):
      scatter_matrix += (self.dataset[:, i].reshape(self.dimension, 1) - mean_vector).dot((self.dataset[:, i].reshape(self.dimension, 1) - mean_vector).T)

    return scatter_matrix

  def calculate_eig(self, scatter_matrix):
    return np.linalg.eig(scatter_matrix)

  def sort_eig(self, eig_val_sc, eig_vec_sc):
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:, i]) for i in range(len(eig_val_sc))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    return eig_pairs

  def save_data(self, opt_file, data):
    np.savetxt(opt_file, data, "%.6f")

  def calculate_pca_k(self, eig_val_sc):
    accumulate_n = self.accumulate(eig_val_sc, self.dimension)
    eig_val_sc = sorted(eig_val_sc, key=float, reverse=True)

    for k in range(1, self.dimension + 1):
      if float(self.accumulate(eig_val_sc, k)) / float(accumulate_n) > 0.9:
        return k

  def transform_data(self, pca_k, eig_pairs):
    stack_arr_list = list()

    for i in range(pca_k):
      stack_arr_list.append(eig_pairs[i][1].reshape(self.dimension, 1))

    matrix_w = np.hstack(stack_arr_list)
    transformed = matrix_w.T.dot(self.dataset)
    return transformed
