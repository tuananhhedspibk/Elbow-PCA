import numpy as np
import csv

class PCA:
  def __init__(self, ipt_data):
    self.dataset = None
    self.dimension = 0

    self.dataset, self.dimension = load_data(ipt_data)

  def load_data(file_name):
    dataset = []

    with open(file_name) as csvfile:
      reader = csv.reader(csvfile)
      for row in reader:
        data = [float(x) for x in row[0].split(" ")]
        dataset.append(data)

    dataset = np.array(dataset)
    dimension = len(dataset)

    return dataset, dimension

  def accumulate(array, index):
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
      element = np.array(np.mean(self.dataset[i, :]))
      vector_elements.append(vector_elements)

    mean_vector = np.array(vector_elements)

    return mean_vector

  def calculate_scatter_matrix(self):
    scatter_matrix = np.zeros((5, 5))
    for i in range(self.dataset.shape[1]):
      scatter_matrix += (self.dataset[:, i].reshape(5, 1) - self.mean_vector).dot((all_samples[:, i].reshape(5, 1)).T)

    return scatter_matrix

  def calculate_eig(self, scatter_matrix):
    return np.linalg.eig(scatter_matrix)

  def sort_eig(self, eig_val_sc, eig_vec_sc):
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:, i]) for i in range(len(eig_val_sc))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    return eig_pairs

  def save_data(opt_file, data, fmt="%.5f"):
    np.savetxt(opt_file, data, fmt=fmt)

  def calculate_pca_k(self, eig_val_sc):
    accumulate_n = accumulate(eig_val_sc, self.dimension)
    eig_val_sc = sorted(eig_val_sc, key=float, reverse=True)

    for k in range(1, self.dimension + 1):
      if float(accumulate(eig_val_sc, k)) / float(accumulate_n) > 0.9:
        return k

  def transform_data(self, pca_k, eig_pairs):
    stack_arr_list = list()

    for i in range(pca_k):
      stack_arr_list.append(eig_pairs[i][1].reshape(self.dimension, 1))

    matrix_w = np.hstack(stack_arr_list)
    transformed = matrix_w.T.dot(self.dataset)
    return transformed
