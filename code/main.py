from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# 获得特征
def get_data(filename):
    with open(filename, 'r') as f:
        X = f.read().split('\n')
        X = X[:-1]
        X_npy = np.ones([len(X), 10])
        for i, x in enumerate(X):
            for j, h in enumerate(x.split()[1:]):
                X_npy[i][j] = float(h)
    return X_npy


# 获得标签
def get_label(filename):
    with open(filename, 'r') as f:
        y = f.read().split('\n')
        y = y[:-1]
        y_npy = np.ones([len(y), ])
        for i, y_i in enumerate(y):
            y_npy[i] = int(y_i.split()[1])
    return y_npy


# KNN 缺失值填充
def fill_data(X_train, X_test):
    print(X_test)

    imputer = KNNImputer()
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    print(X_train)
    return X_train, X_test


# 挑选特征
def choose_features(X_train, X_test):
    X_train = X_train[:, (0, 1, 2, 4, 6)]
    X_test = X_test[:, (0, 1, 2, 4, 6)]
    return X_train, X_test


# 标准化
def normalize(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


# 保存结果
def to_result(test_file, res_file, result):
    with open(test_file, 'r') as f:
        test_data = f.read().split("\n")
        test_data = test_data[:-1]
    res = ''
    for i in range(len(test_data)):
        res += test_data[i].split()[0] + " " + str(int(result[i])) + "\n"
    res = res.strip()
    with open(res_file, 'w') as f:
        f.write(res)


if __name__ == '__main__':
    # 获取数据
    train_file = '../data/trainset/train_para_input.txt'
    train_label_file = '../data/trainset/train_output.txt'
    test_file = '../data/testset/test_para_input.txt'
    X_train = get_data(train_file)
    y_train = get_label(train_label_file)
    X_test = get_data(test_file)

    # 缺失值填充
    X_train, X_test = fill_data(X_train, X_test)

    # 挑选特征
    X_train, X_test = choose_features(X_train, X_test)

    # 标准化
    X_train, X_test = normalize(X_train, X_test)

    # 加载模型
    model = joblib.load('../user_data/model_data/model.model')

    # 获得测试结果
    result = model.predict(X_test)
    # 将结果保存进 prediction_result 目录
    result_file = '../prediction_result/result.txt'
    to_result(test_file, result_file, result)