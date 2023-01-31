import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt


class NMF():
    def init_param(self, df, k, rms_flag, rms_t):
        """
        :df: 対象データ
        :param k: 特徴因子数
        """
        # self._row = len(df[0])
        # self._col = len(df)
        self.X = np.array(df)  # i*j
        if (rms_flag):
            self.X = []
            for n in range(len(df)):
                self.X.append(np.convolve(df[n], np.ones(5) / 5, mode='valid'))
            self.X = np.array(self.X)
        self.W = np.abs(np.random.uniform(low=0, high=1, size=(k, len(self.X))))  # k*i

        self.H = np.abs(np.random.uniform(low=0, high=1, size=(k, len(self.X[0]))))  # k*j

    def EUC(self):
        """
        ユークリッド距離
        """
        tmp_W = self.W * (np.dot(self.X, self.H.T)).T / np.dot(np.dot(self.W.T, self.H), self.H.T).T
        tmp_H = self.H * (np.dot(self.W, self.X)) / np.dot(self.W, np.dot(self.W.T, self.H))

        self.H = tmp_H
        self.W = tmp_W

    def KL(self):
        """
        カルバックライブラーダイバージェンス
        """
        tmp_W_numerator = np.dot((self.X / np.dot(self.W.T, self.H)), self.H.T).T
        tmp_W_denominator = np.array([np.sum(self.H, axis=1) for i in range(len(self.W[0]))])
        tmp_W = self.W * (tmp_W_numerator / tmp_W_denominator.T)

        tmp_H_numerator = np.dot((self.X / np.dot(self.W.T, self.H)).T, self.W.T).T
        tmp_H_denominator = np.array([np.sum(self.W, axis=1) for i in range(len(self.H[0]))]).T
        tmp_H = self.H * tmp_H_numerator / tmp_H_denominator

        self.H = tmp_H
        self.W = tmp_W

    def IS(self):
        # ISの更新式のWの分子について. 分子はij*jkをして、これを転置することでkiの行列を作成
        tmp_W_numerator = np.dot((self.X / (np.dot(self.W.T, self.H) ** 2)), self.H.T).T
        tmp_W_denominator = np.dot(1 / np.dot(self.W.T, self.H), self.H.T).T
        tmp_W = self.W * np.sqrt(tmp_W_numerator / tmp_W_denominator)

        tmp_H_numerator = np.dot((self.X / np.dot(self.W.T, self.H) ** 2).T, self.W.T).T
        tmp_H_denominator = np.dot(1 / np.dot(self.W.T, self.H).T, self.W.T).T
        tmp_H = self.H * np.sqrt(tmp_H_numerator / tmp_H_denominator)

        self.H = tmp_H
        self.W = tmp_W

    def NMF_solve(self, iter, method):
        if method == 1:
            for n in range(iter):
                self.EUC()
        if method == 2:
            for n in range(iter):
                self.KL()
        if method == 3:
            for n in range(iter):
                self.IS()
        print("誤差のノルム: ", np.linalg.norm(self.X - np.dot(self.W.T, self.H)))
        print("W: ", self.W)
        print("H: ", self.H)
        print("X: ", self.X)
        print("WH: ", np.dot(self.W.T, self.H))


nmf = NMF()

import pandas as pd

df=pd.read_csv("NMF_sample.csv")
df_=df.values
print(df_)

nmf.init_param(df=df_, k=3, rms_flag=False,rms_t=30)
nmf.NMF_solve(2000, 3)
# inport EMG File
# with open("NMF_sample.csv", 'r', encoding='utf-8') as f:
#     reader = csv.reader(f)
#     df = []
#     for row in reader:
#         df.append(row)
#     df = np.array(pd.DataFrame(df[4:]).values)
#     df = df[:, 3:]
# --------------------------------------------------
