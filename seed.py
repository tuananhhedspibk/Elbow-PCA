import numpy as np

def create_data(file_name, dimension, number_of_data, fmt):
  np.random.seed(11)

  mu_vec1 = np.zeros(shape(1, dimension))
  mu_vec2 = np.ones(shape(1, dimension))
  cov_mat = np.identity(dimension)

  class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat, number_of_data).T
  class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat, number_of_data).T

  all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
  np.savetxt(file_name, all_samples, fmt=fmt)

if __name__ == "__main__":
  create_data("./data.csv", 5, 30, "%.5f")
