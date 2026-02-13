import numpy as np
f = open('../data/trainset/train_para_input.txt')
X = f.read().split("\n")# 按行分割
X = X[:-1]
X_npy = np.ones([len(X),10])
for i, x in enumerate(X):
    for j, h in enumerate(x.split()[1:]):
        X_npy[i][j] = float(h)
