from scipy.sparse import *
import numpy as np

def p1_has_cycle(sets):
    matrix = csr_matrix(sets)
    row = len(sets)
    col = len(sets[0])
    while matrix.shape[0] != 0:
      col_index = np.where(matrix[0].toarray()==1)[1][0]
      negative_row = np.where(matrix[1:, col_index].toarray()==-1)[0] + 1
      for row in negative_row:
        new_row = matrix[0] + matrix[row]
        all_zero_row = np.where(new_row.toarray() == 0)[1]
        if len(all_zero_row) == col:
          return True
        
        matrix = vstack((matrix, new_row))

      matrix = matrix[1:]

    return False
