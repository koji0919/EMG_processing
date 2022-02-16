from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread
import myo
import time
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from socket import socket, AF_INET, SOCK_DGRAM
from statistics import mode
from scipy.spatial import distance

emg_train=np.array([[] for i in range(8)])

def queue_init(dim,q_length):   #キューを作成する
    tmp=deque(maxlen=q_length)
    a=0
    if dim==2:
        a=[0,0]
    for i in range(q_length):
        tmp.append(a)
    return tmp

finger_2d = queue_init(2,10)
finger_predict = queue_init(1,10)

testdata_2d_rec=[]  #評価タスク時のデータを格納

finger_label=[]
emg_features=[]
lda_finger=LinearDiscriminantAnalysis(n_components=2)
count=0
ovr_l = 20
win_l = 40
fps=0.001

queue_len=6
do_record=True
class_f = 10  # 指

ADDR = '127.0.0.1'
PORT_TO = 50007 #送信ポート
M_SIZE = 1024

ax1=0    #pltのリアルタイムプロット用のグローバル変数(0は仮で代入)
scat_finger=0

def queue_init(dim,q_length):
    tmp=deque(maxlen=q_length)
    a=0
    if dim==2:
        a=[0,0]
    for i in range(q_length):
        tmp.append(a)
    return tmp

def update_plot(scat1,ax1):
    tmp = np.array([x for x in list(finger_2d)])
    scat1.set_offsets(tmp)
    label_ = ["グー", "人差し指","中薬指","小指","パー","ピース","中指立","内屈","外屈","無動作"]
    ax1.set_title("finger motion : " + label_[finger_predict[-1]],fontname="MS Gothic")
    plt.pause(fps)


class EmgCollector(myo.DeviceListener):
  """
  Collects EMG data in a queue with *n* maximum number of elements.
  """
  def __init__(self, n):
    self.n = n
    self.idle=False  #Trueの間計測を行う
    self.lock = Lock()
    self.time_pre=0
    self.emg_data_queue = deque(maxlen=win_l)  #テスト時の取得筋電位データの数はここで決定
    self.start_time=time.time()
    self.sampleamount = 0

  def get_emg_queue(self):
    with self.lock:
      return list(self.emg_data_queue)

  def on_connected(self, event):
    event.device.stream_emg(True)

  def start(self):
      for i in range(ovr_l):
        self.emg_data_queue.append((0, [0,0,0,0,0,0,0,0]))
      self.idle=True

  def predict(self):
      #global scat_finger, ax1
      tmp=self.get_emg_queue()
      tmp=np.array([x[1] for x in tmp]).T
      test_features=[feature_calc(tmp, win_l)]
      df = pd.DataFrame(test_features)
      global finger_predict,wrist_predict,testdata_2d_rec
      lda_fingerresult=int(lda_finger.predict(df.values))
      finger_predict.append(lda_fingerresult)
      a= lda_finger.transform(df.values)
      global finger_2d,wrist_2d
      finger_2d.append(a[0])

  def end(self):
      self.idle=False

  def on_emg(self, event):
    with self.lock:
        if self.idle:
            tmp=event.emg
            self.sampleamount+=1
            self.emg_data_queue.append((event.timestamp, event.emg))

            if self.sampleamount==ovr_l:
                self.sampleamount=0
                thread = Thread(target=self.predict)
                thread.start()
                thread.join(0.0001)

class Train(object):
    def __init__(self,listener):
        self.n = listener.n
        self.listener = listener
    def Show_emg_fb(self):
        global ax1,finger_2d,scat_finger,finger_predict
        snd = socket(AF_INET, SOCK_DGRAM)
        self.listener.start()
        while True:
            update_plot(scat_finger,ax1)
            msg = str(finger_predict[-1]) + "0"
            snd.sendto(msg.encode(),(ADDR,PORT_TO))

    def Show_emg_nfb(self,df,ff,fm):
        global finger_2d,finger_predict
        fig = plt.figure(figsize=(4, 12))
        ax1 = plt.subplot(111)
        snd = socket(AF_INET, SOCK_DGRAM)
        base_df = df
        features_finger = ff
        base2d_finger = fm

        basedata_finger = []  # 0番目から各クラスの重心

        for k in range(class_f):
            tmp = []
            tmp.append([base2d_finger[j][0] for j in range(len(base2d_finger)) if features_finger[j] == (k)])
            tmp.append([base2d_finger[j][1] for j in range(len(base2d_finger)) if features_finger[j] == (k)])
            basedata_finger.append(tmp)

        basedata_f_centers = []

        for i in range(class_f):
            tmp = []
            tmp.append(np.mean(basedata_finger[i][0]))
            tmp.append(np.mean(basedata_finger[i][1]))
            basedata_f_centers.append(tmp)

        # --ここまでが初回計測データからの重心計算と、使用ファイル読み込み
        # ここからが計測データの処理
        basedata_f_cov = []

        for i in range(class_f):
            tmp = np.cov(basedata_finger[i][0], basedata_finger[i][1])
            basedata_f_cov.append(np.linalg.pinv(tmp))

        label_ =["グー", "人差し指","中薬指","小指","パー","ピース","中指立","内屈","外屈","無動作"]
        self.listener.start()
        while True:
            msg=str(finger_predict[-1])+"0"
            snd.sendto(msg.encode(),(ADDR,PORT_TO))
            ax1.cla()

            ax1.set_title("finger motion : " + label_[finger_predict[-1]], fontname="MS Gothic")
            f_maharanobis=distance.mahalanobis(list(finger_2d[-1]),basedata_f_centers[finger_predict[-1]],basedata_f_cov[finger_predict[-1]])
            ax1.bar([1], f_maharanobis)
            ax1.set_ylim(0,8)
            plt.pause(0.00001)

def feature_calc(emg,win_l):    #改変時は横のEMGRecordも同じにすること
    FEATURES = []
    for i in range(8):
        tmp = emg[i]
        FEATURES.append(np.mean(np.abs(tmp)))  # MAV
        FEATURES.append(np.var(tmp))  # VAR
        zero = 0
        for j in range(0, win_l - 1):
            if tmp[j] * tmp[j + 1] < 0:
                zero += 1
        FEATURES.append(zero)   #ZC
        diff = np.diff(tmp, n=1)
        FEATURES.append(np.sum(np.abs(diff)))  # WL
        # freq = np.abs(np.fft.fft(tmp))  # 周波数領域
        # FEATURES.append(np.max(freq))  # PKF
        # FEATURES.append(np.mean(freq))  # MKF

    return FEATURES

def main():
    finger_2d = queue_init(2,queue_len)
    finger_predict = queue_init(1,queue_len)
#-----------------------------------------
    myo.init()
    hub = myo.Hub()
    listener = EmgCollector(512)

    features_finger=[]
    features_wrist=[]
    with hub.run_in_background(listener.on_event):
        global lda_finger, lda_wrist
        print("training")
        df = pd.read_csv('nkn_r_base1.csv', header=0, index_col=0)
        features_finger = df.values[:,-1]
        df=df.values[:,0:-1]
        finger_motion = lda_finger.fit(df, features_finger).transform(df)

        fb = input("start Train with FeedBack? y/n:")
        if fb == "y":
            fig = plt.figure(figsize=(18, 12))
            global ax1, ax2, scat_finger, scat_wrist
            ax1 = plt.subplot(111)
            label_ = ["グー", "人差し指","中薬指","小指","パー","ピース","中指立","内屈","外屈","無動作"]
            for k in range(class_f):  # 手描画
                tmp = []
                tmp.append([finger_motion[j][0] for j in range(len(finger_motion)) if features_finger[j] == (k)])
                tmp.append([finger_motion[j][1] for j in range(len(finger_motion)) if features_finger[j] == (k)])
                tmp = np.array(tmp)
                scat_finger = ax1.scatter(tmp[0], tmp[1], label=label_[k], cmap='viridis', edgecolor='blacK')
            ax1.set_title("finger_pattern")
            ax1.legend(labels=label_, fontsize=12,prop={"family":"MS Gothic"})
            scat_finger = ax1.scatter(0, 0, label="current", c="crimson", s=100,
                                      marker="X")  # 空撃ちすることでリアルタイム分類時のset_datasに備える

            plt.pause(0.05)
            Train(listener).Show_emg_fb()
        if fb =="n":
            Train(listener).Show_emg_nfb(df,features_finger,finger_motion)

if __name__ == '__main__':
  main()