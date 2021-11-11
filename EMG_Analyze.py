from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial import distance
class_f = 7  # 指

lda_finger=LinearDiscriminantAnalysis(n_components=2)

base_df = pd.read_csv('kmr_l.csv', header=0, index_col=0)
features_finger = base_df.values[:,-1]
base_df=base_df.values[:,0:-1]
base2d_finger = lda_finger.fit(base_df, features_finger).transform(base_df)

basedata_finger=[] #各クラスの重心

for k in range(class_f):
    tmp = []
    tmp.append([base2d_finger[j][0] for j in range(len(base2d_finger)) if features_finger[j] == (k)])
    tmp.append([base2d_finger[j][1] for j in range(len(base2d_finger)) if features_finger[j] == (k)])
    basedata_finger.append(tmp)

basedata_f_centers=[]

for i in range(class_f):
    tmp=[]
    tmp.append(np.mean(basedata_finger[i][0]))
    tmp.append(np.mean(basedata_finger[i][1]))
    basedata_f_centers.append(tmp)
#使ってたデータが5クラス有ったので
#--ここまでが初回計測データからの重心計算と、使用ファイル読み込み
#ここからが計測データの処理
basedata_f_cov=[]

for i in range(class_f):
    tmp=np.cov(basedata_finger[i][0],basedata_finger[i][1])
    basedata_f_cov.append(np.linalg.pinv(tmp))

input_f_maharanobis=[]

input_df = pd.read_csv('1108rec_l2.csv', header=0, index_col=0)
input_finger2d=np.array(input_df.values[:,0:3])
input_wrist2d=np.array(input_df.values[:,3:6])

input_f_labels=np.array(input_df.values[:,0])
for j in range(class_f):
    center=basedata_f_centers[j]
    tmp=[]
    for i in range(len(input_finger2d)):
        tmp.append(distance.mahalanobis(list(input_finger2d[i][1:3]),center,basedata_f_cov[int(input_finger2d[i][0])-1]))
    input_f_maharanobis.append(tmp)


fig, axlist = plt.subplots(2,1,figsize=(18.0, 12.0))
id=421
title=["fist", "point","wave in","wave out","spread","nomotion","fox"]
i=0
for ax in axlist:
    ax.set_title(title[i])
    ax.plot(np.arange(len(input_f_maharanobis[0])),input_f_maharanobis[i]) #int(input_finger2d[i][0])-1
    ax.grid()
    ax.set_xlabel('time(1 sample/0.16s)')
    ax.set_ylabel('maharanobis distance')
    i+=1
plt.subplots_adjust(wspace=0.4, hspace=1)

target_class=0
pre=9
target_flag=False
part_ave_mahalanobis=[]
part_var_mahalanobis=[]
tmp=[]
tmp1=[]
j=0
for ax in axlist:

    for i in range(len(input_finger2d)):    #len(input_finger2d)-1
        if target_flag:
            tmp.append(input_f_maharanobis[j][i])
            tmp1.append(input_f_maharanobis[j][i])
        if input_finger2d[i][0] != pre:
            if target_flag:
                ax.axvspan(start, i, facecolor='orange', alpha=0.5)
                target_flag=False
                part_ave_mahalanobis.append(round(np.mean(tmp[2:-3]),4))
                part_var_mahalanobis.append(round(np.var(tmp[2:-2]),4))
            if input_finger2d[i][0]==target_class:
                start=i
                target_flag=True
                tmp = []
            pre=input_finger2d[i][0]
    print(part_ave_mahalanobis)
    print(part_var_mahalanobis)
    j+=1
    plt.xlabel('time(1 sample/0.16s)')
    plt.ylabel('maharanobis distance')
    plt.show()
    if j==1 :
        break


fig1, ax1 = plt.subplots()
ax1.set_title('Mahalanobis distance')
ax1.set_xticklabels(['average'])
ax1.boxplot(tmp1)
plt.show()
plt.show()








