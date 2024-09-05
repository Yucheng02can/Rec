import numpy as np
from scipy.sparse import csr_matrix

# Creating a sample CSR matrix
data = np.array([1, -1, 1, -1, 1])
rows = np.array([0, 0, 1, 2, 2])
cols = np.array([0, 2, 1, 0, 2])
csr = csr_matrix((data, (rows, cols)), shape=(3, 3))

print("Original CSR Matrix:\n", csr.toarray())

# Convert to COO format
coo = csr.tocoo()

# Filter out elements with a value of -1
mask = coo.data != -1
filtered_coo = csr_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=csr.shape)

# Convert back to CSR format
filtered_csr = filtered_coo.tocsr()

print("Filtered CSR matrix:\n", filtered_csr.toarray())
