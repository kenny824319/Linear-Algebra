import sys
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_wave(x, path = './wave.png'):
    plt.gcf().clear()
    plt.plot(x)
    plt.xlabel('n')
    plt.ylabel('xn')
    plt.savefig(path)

def plot_ak(a, path = './freq.png'):
    plt.gcf().clear()

    # Only plot the mag of a
    a = np.abs(a)
    plt.plot(a)
    plt.xlabel('k')
    plt.ylabel('ak')
    plt.savefig(path)

def CosineTrans(x, B):
    # implement cosine transform
    a = np.dot(np.linalg.inv(B), x)
    return a

def InvCosineTrans(a, B):
    # implement inverse cosine transform
    x = np.dot(B, a)
    return x

def gen_basis(N):
    K = N
    B = np.zeros(N*K).reshape(N, K)
    for k in range(K):
      for n in range(N):
        if k == 0:
          B[n][k] = 1 / np.sqrt(N)
        else:
          B[n][k] = np.sqrt(2 / N) * np.cos((n + 0.5) * k * np.pi / N)
    return B

if __name__ == '__main__':
    # Do not modify these 2 lines
    signal_path = sys.argv[1]
    out_directory_path = sys.argv[2]
    x = np.loadtxt(signal_path)
    N = len(x)
    B = gen_basis(N)
    a = CosineTrans(x, B)
    tmp_a = np.absolute(a)
    tmp_a = np.sort(tmp_a, axis=0)[::-1]
    min_val = int(tmp_a[4])
    mask = np.where(a > min_val)[0]
    mask_f1 = np.zeros(N)
    mask_f1[mask[0]] = a[mask[0]]
    mask_f3 = np.zeros(N)    
    mask_f3[mask[2]] = a[mask[2]]
    f1 = InvCosineTrans(mask_f1, B)
    f3 = InvCosineTrans(mask_f3, B)
    
    # Do not modify these 3 lines
    plot_ak(a, path=os.path.join(out_directory_path, 'freq.png'))
    plot_wave(f1, path=os.path.join(out_directory_path, 'f1.png'))
    plot_wave(f3, path=os.path.join(out_directory_path, 'f3.png'))

