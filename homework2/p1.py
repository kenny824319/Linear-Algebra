import numpy as np
from util import mod_inv

def decode(cipher, key):
  letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_.,?!'
  key = [int(k) for k in key.split()]
  key = np.array(key).reshape(3, 3)
  inv = mod_inv(key)
  cipher = [letters.index(i) for i in cipher]
  cipher = np.array(cipher).reshape(-1, 3).T
  plain = np.mod(inv @ cipher, 31)
  plain2text = [letters[p] for p in plain.T.flatten()]
  plain = ''.join(plain2text)
  return plain
