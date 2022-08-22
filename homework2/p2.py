import numpy as np
from util import mod_inv


def get_key(cipher, plain):
  letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_.,?!'
  cipher = [letters.index(i) for i in cipher]
  cipher = np.array(cipher).reshape(-1, 3).T
  plain = [letters.index(i) for i in plain]
  plain = np.array(plain).reshape(-1, 3).T
  inv = mod_inv(plain)
  public_key = np.mod(cipher @ inv, 31)
  key = [str(p.item()) for p in public_key.flatten()]
  key = ' '.join(key)
  return key

