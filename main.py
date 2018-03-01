import numpy as np

if __name__ == "__main__":
  np.random.seed(1)

  mu_vec1 = np.array([0, 0, 0, 0, 0])
  cov_mat1 = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

  class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 30).T

  mu_vec2 = np.array([0, 0, 1, 0, 0])
  cov_mat2 = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

  class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 30).T

  mu_vec3 = np.array([1, 1, 1, 1, 1])
  cov_mat3 = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

  class3_sample = np.random.multivariate_normal(mu_vec3, cov_mat3, 30).T

  all_samples = np.concatenate((class1_sample, class2_sample, class3_sample), axis=1)

  np.savetxt('./data.csv', all_samples, fmt='%.5f')
