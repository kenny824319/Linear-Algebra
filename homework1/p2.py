from scipy.sparse import *
import numpy as np

def p2_has_cycle(sets):
    matrix = csr_matrix(sets)
    orig_mat = matrix
    for i in range(len(sets)):
      diag = matrix.diagonal().copy()
      above_zero = np.where(matrix.diagonal() > 0)[0]
      if len(above_zero) > 0:
        return True
      matrix = matrix.dot(orig_mat)



    return False
