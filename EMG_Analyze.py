from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial import distance
class_n = 4  # 指
class_m = 4  # 手首

lda_finger=LinearDiscriminantAnalysis(n_components=2)
lda_wrist=LinearDiscriminantAnalysis(n_components=2)

base_df = pd.read_csv('EMG_features.csv', header=0, index_col=0)
features_finger = np.loadtxt("finger_label.txt")
features_wrist = np.loadtxt("wrist_label.txt")
base2d_finger = lda_finger.fit(base_df.values, features_finger).transform(base_df.values)
base2d_wrist = lda_wrist.fit(base_df.values, features_wrist).transform(base_df.values)

basedata_finger=[] #0番目から各クラスの重心
basedata_wrist=[]

for k in range(class_m):
    tmp = []
    tmp.append([base2d_finger[j][0] for j in range(len(base2d_finger)) if features_finger[j] == (k+1)])
    tmp.append([base2d_finger[j][1] for j in range(len(base2d_finger)) if features_finger[j] == (k+1)])
    basedata_finger.append(tmp)

for k in range(class_m):
    tmp = []
    tmp.append([base2d_wrist[j][0] for j in range(len(base2d_wrist)) if features_finger[j] == (k+1)])
    tmp.append([base2d_wrist[j][1] for j in range(len(base2d_wrist)) if features_finger[j] == (k+1)])
    basedata_wrist.append(tmp)


basedata_f_centers=[]
basedata_w_centers=[]

for i in range(class_m):
    tmp=[]
    tmp.append(np.mean(basedata_finger[i][0]))
    tmp.append(np.mean(basedata_finger[i][1]))
    basedata_f_centers.append(tmp)
#使ってたデータが5クラス有ったので
basedata_f_centers.append([1,1])


for i in range(class_n):
    tmp=[]
    tmp.append(np.mean(basedata_wrist[i][0]))
    tmp.append(np.mean(basedata_wrist[i][1]))
    basedata_w_centers.append(tmp)

#--ここまでが初回計測データからの重心計算と、使用ファイル読み込み
#ここからが計測データの処理
basedata_f_cov=[]
basedata_w_cov=[]

for i in range(class_n):
    tmp=np.cov(basedata_finger[i][0],basedata_finger[i][1])
    basedata_f_cov.append(np.linalg.pinv(tmp))
basedata_f_cov.append([[1,1],[1,1]])
for i in range(class_m):
    tmp = np.cov(basedata_wrist[i][0], basedata_wrist[i][1])
    basedata_w_cov.append(np.linalg.pinv(tmp))

input_f_maharanobis=[]
input_w_maharanobis=[]

input_df = pd.read_csv('samle.csv', header=0, index_col=0)
input_finger2d=np.array(input_df.values[:,0:3])
input_wrist2d=np.array(input_df.values[:,3:6])

input_f_labels=np.array(input_df.values[:,0])
for i in range(len(input_finger2d)):
    input_f_maharanobis.append(distance.mahalanobis(list(input_finger2d[i][1:3]),basedata_f_centers[int(input_finger2d[i][0])-1],basedata_f_cov[int(input_finger2d[i][0])-1]))

fig, ax = plt.subplots()

ax.plot(np.arange(len(input_f_maharanobis[0:100])),input_f_maharanobis[0:100])
ax.grid()
ax.axvspan(50, 100, facecolor='orange', alpha=0.5)
plt.show()










