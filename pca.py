import numpy as np
import csv

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

all_samples = []

with open("data.csv") as csvfile:
  reader = csv.reader(csvfile)
  for row in reader:
    data = [float(x) for x in row[0].split(" ")]
    all_samples.append(data)

all_samples = np.array(all_samples)

mean_x = np.mean(all_samples[0, :])
mean_y = np.mean(all_samples[1, :])
mean_z = np.mean(all_samples[2, :])
mean_t = np.mean(all_samples[3, :])
mean_w = np.mean(all_samples[4, :])

mean_vector = np.array([[mean_x], [mean_y], [mean_z], [mean_t], [mean_w]])

# Computing the Scatter Matrix

scatter_matrix = np.zeros((5, 5))
for i in range(all_samples.shape[1]):
  scatter_matrix += (all_samples[:, i].reshape(5, 1) - mean_vector).dot((all_samples[:, i].reshape(5, 1)).T)

# Computing eigenvectors and corresponding eigenvalues

eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:, i]) for i in range(len(eig_val_sc))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

accumulate_n = accumulate(eig_val_sc, 5)
pca_k = 0
eig_val_sc = sorted(eig_val_sc, key=float, reverse=True)

for k in range(1, 6):
  if float(accumulate(eig_val_sc, k)) / float(accumulate_n) > 0.9:
    pca_k = k
    break

print eig_val_sc
print pca_k

stack_arr_list = list()

for i in range(pca_k):
  stack_arr_list.append(eig_pairs[i][1].reshape(5, 1))

matrix_w = np.hstack(stack_arr_list)
transformed = matrix_w.T.dot(all_samples)

np.savetxt('./data.csv', transformed, fmt='%.5f')
