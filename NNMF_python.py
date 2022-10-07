    import numpy as np
    import csv
    import pandas as pd
    import matplotlib.pyplot as plt

    class NMF():
        def init_param(self,df,k):
            """
            :df: 対象データ
            :param k: 特徴因子数
            """
            self._k=k
            self._row=len(df[0])
            self._col=len(df)
            self.X=df
            self.W=np.abs(np.random.uniform(low=0, high=np.max(self.X), ))
            self.H=np.abs(np.random.uniform(low=0, high=np.max(self.X), ))

        emg_max = np.max(df)

        W = np.abs(np.random.uniform(low=0, high=emg_max, ))

        def update(self):   #更新

        def EUC(self):
            self.H = self.H * np.dot(self.W.T, self.X).T / np.dot(self.W.T, np.dot(self.W, self.H.T)).T
            tmp_W = self.W * np.dot(self.X, self.H) / np.dot(np.dot(self.W, self.H.T), self.H)
            # self.H=tmp_H
            self.W=tmp_W

        def KL(self):
            """
            カルバックライブラーダイバージェンス
            """
            self.H=self.H*np.dot(self.W.T,self.X/np.dot(self.W,self.H))/self.W.T
            tmp_W=self.W*np.dot(self.X/(np.dot(self.W,self.H),self.H.T))/self.H.T

        def IS(self):
            tmp_H=self.H * np.sqrt(np.linalg.norm(self.H * self.X/)^2)

        def NMF_solve(self):
            """
            :param eva_flag:EUC,KL,IS
            :return:
            """
            iter=200    #反復回数

            nmf=NMF()
            nmf.init([[1, 2, 3, 4], [2, 3, 4, 5]], k=3)




    #inport EMG File
    with open("NMF_sample.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        df = []
        for row in reader:
            df.append(row)
        df = np.array(pd.DataFrame(df[4:]).values)
        df=df[:,3:]
    #--------------------------------------------------
    #パラメータ
    k=4
    m,n=1000,2

