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
emg_test=np.array([[] for i in range(8)])

def queue_init(dim,q_length):   #キューを作成する
    tmp=deque(maxlen=q_length)
    a=0
    if dim==2:
        a=[0,0]
    for i in range(q_length):
        tmp.append(a)
    return tmp

finger_2d = queue_init(2,10)
wrist_2d = queue_init(2,10)
finger_predict = queue_init(1,10)
wrist_predict = queue_init(1,10)

testdata_2d_rec=[]  #評価タスク時のデータを格納

finger_label=[]
wrist_label=[]
emg_features=[]
lda_finger=LinearDiscriminantAnalysis(n_components=2)
lda_wrist=LinearDiscriminantAnalysis(n_components=2)
count=0
ovr_l = 32
win_l = 64
fps=0.05


queue_len=6
do_record=True
class_f = 7  # 指
class_w = 4  # 手首

ADDR = '127.0.0.1'
PORT_TO = 50007 #送信ポート
M_SIZE = 1024

ax1=0    #pltのリアルタイムプロット用のグローバル変数(0は仮で代入)
ax2=0
scat_finger=0
scat_wrist=0

def queue_init(dim,q_length):
    tmp=deque(maxlen=q_length)
    a=0
    if dim==2:
        a=[0,0]
    for i in range(q_length):
        tmp.append(a)
    return tmp

def hist_data(inputs,n):
    results=[]
    inputlist=list(inputs)
    for i in range(n):
        results.append(inputlist.count((i+1)))
    return results

def update_plot(scat1,scat2):
    print
    tmp = np.array([x for x in list(finger_2d)])
    #tmp2 = np.array([x for x in list(wrist_2d)])
    scat1.set_offsets(tmp)
    #scat2.set_offsets(tmp2)
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
      self.idle=True

  def predict(self):
      #global scat_finger, ax1
      tmp=self.get_emg_queue()
      tmp=np.array([x[1] for x in tmp]).T
      test_features=[feature_calc(tmp, win_l)]
      df = pd.DataFrame(test_features)
      global finger_predict,wrist_predict,testdata_2d_rec
      lda_fingerresult=int(lda_finger.predict(df.values))
      #lda_wristresult=int(lda_wrist.predict(df.values))
      finger_predict.append(lda_fingerresult)
      #wrist_predict.append(lda_wristresult)
      a= lda_finger.transform(df.values)
      #b= lda_wrist.transform(df.values)
      global finger_2d,wrist_2d
      finger_2d.append(a[0])
      #wrist_2d.append(b[0])
      #testdata_2d_rec.append([lda_fingerresult,a[0][0],a[0][1],lda_wristresult,b[0][0],b[0][1]])

  def end(self):
      self.idle=False

  def on_emg(self, event):
    with self.lock:
        if self.idle:
            tmp=event.emg
            self.sampleamount+=1
            self.emg_data_queue.append((event.timestamp, event.emg))

            if self.sampleamount==32:
                self.sampleamount=0
                thread = Thread(target=self.predict)
                thread.start()
                thread.join(0.0001)



class Train(object):
    def __init__(self,listener):
        self.n = listener.n
        self.listener = listener
    def Show_emg_fb(self):
        global ax1,finger_2d,scat_finger,ax2,wrist_2d,scat_wrist,finger_predict,wrist_predict
        snd = socket(AF_INET, SOCK_DGRAM)
        self.listener.start()
        while True:
            update_plot(scat_finger,scat_wrist)
            msg = str(mode(finger_predict)) + str(mode(wrist_predict))
            snd.sendto(msg.encode(),(ADDR,PORT_TO))

    def Show_emg_nfb(self,df,ff,fm):
        global finger_2d, wrist_2d, finger_predict, wrist_predict
        fig = plt.figure(figsize=(18, 12))
        ax1 = plt.subplot(111)
        #ax2 = plt.subplot(122)
        snd = socket(AF_INET, SOCK_DGRAM)

        base_df = df
        features_finger = ff
        #features_wrist = fw
        base2d_finger = fm
        #base2d_wrist = wm

        basedata_finger = []  # 0番目から各クラスの重心
        basedata_wrist = []

        for k in range(class_f):
            tmp = []
            tmp.append([base2d_finger[j][0] for j in range(len(base2d_finger)) if features_finger[j] == (k)])
            tmp.append([base2d_finger[j][1] for j in range(len(base2d_finger)) if features_finger[j] == (k)])
            basedata_finger.append(tmp)

        # for k in range(class_w):
        #     tmp = []
        #     tmp.append([base2d_wrist[j][0] for j in range(len(base2d_wrist)) if features_finger[j] == (k)])
        #     tmp.append([base2d_wrist[j][1] for j in range(len(base2d_wrist)) if features_finger[j] == (k)])
        #     basedata_wrist.append(tmp)

        basedata_f_centers = []
        basedata_w_centers = []

        for i in range(class_f):
            tmp = []
            tmp.append(np.mean(basedata_finger[i][0]))
            tmp.append(np.mean(basedata_finger[i][1]))
            basedata_f_centers.append(tmp)

        # for i in range(class_w):
        #     tmp = []
        #     tmp.append(np.mean(basedata_wrist[i][0]))
        #     tmp.append(np.mean(basedata_wrist[i][1]))
        #     basedata_w_centers.append(tmp)

        # --ここまでが初回計測データからの重心計算と、使用ファイル読み込み
        # ここからが計測データの処理
        basedata_f_cov = []
        basedata_w_cov = []

        for i in range(class_f):
            tmp = np.cov(basedata_finger[i][0], basedata_finger[i][1])
            basedata_f_cov.append(np.linalg.pinv(tmp))
        # for i in range(class_w):
        #     tmp = np.cov(basedata_wrist[i][0], basedata_wrist[i][1])
        #     basedata_w_cov.append(np.linalg.pinv(tmp*tmp))
        self.listener.start()
        while True:
            msg=str(mode(finger_predict))+str(mode(wrist_predict))
            snd.sendto(msg.encode(),(ADDR,PORT_TO))
            ax1.cla()
            #ax2.cla()
            ax1.set_title("finger motion")
            #ax2.set_title("wrist motion")
            f_maharanobis=distance.mahalanobis(list(finger_2d[-1]),basedata_f_centers[finger_predict[-1]],basedata_f_cov[finger_predict[-1]])
            # = distance.mahalanobis(list(wrist_2d[-1]), basedata_w_centers[wrist_predict[-1]],basedata_w_cov[wrist_predict[-1]])
            #print(finger_predict[-1],f_maharanobis, wrist_predict[-1],w_maharanobis)
            ax1.bar([1], f_maharanobis)
            #ax2.bar([1], w_maharanobis)
            ax1.set_ylim(0,7.2)
            ax1.set_xlim(0, class_f)
            #ax2.set_ylim(0,7.2)
            #ax2.set_xlim(0, class_w)
            print()
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
    wrist_2d = queue_init(2,queue_len)
    finger_predict = queue_init(1,queue_len)
    wrist_predict = queue_init(1,queue_len)
#-----------------------------------------
    myo.init()
    hub = myo.Hub()
    listener = EmgCollector(512)

    features_finger=[]
    features_wrist=[]
    with hub.run_in_background(listener.on_event):
        global lda_finger, lda_wrist
        print("training")
        df = pd.read_csv('test.csv', header=0, index_col=0)

        features_finger = df.values[:,-1]
        df=df.values[:,0:-1]
        print(df)
        #features_wrist = np.loadtxt("test2.txt")

        finger_motion = lda_finger.fit(df, features_finger).transform(df)
        #wrist_motion = lda_wrist.fit(df, features_wrist).transform(df)

        fb = input("start Train with FeedBack? y/n:")
        if fb == "y":
            fig = plt.figure(figsize=(18, 12))
            global ax1, ax2, scat_finger, scat_wrist
            ax1 = plt.subplot(111)
            # ax2 = plt.subplot(122)
            label_ = ["fist", "point","wave in","wave out","spread","nomotion","fox"]
            label__ = ["nomotion", "flexion", "pronation", "supination"]
            for k in range(class_f):  # 手描画
                tmp = []
                tmp.append([finger_motion[j][0] for j in range(len(finger_motion)) if features_finger[j] == (k)])
                tmp.append([finger_motion[j][1] for j in range(len(finger_motion)) if features_finger[j] == (k)])
                tmp = np.array(tmp)
                scat_finger = ax1.scatter(tmp[0], tmp[1], label=label_[k], cmap='viridis', edgecolor='blacK')
            ax1.set_title("finger_pattern")
            ax1.legend(labels=label_, fontsize=12)
            scat_finger = ax1.scatter(0, 0, label="current", c="crimson", s=100,
                                      marker="X")  # 空撃ちすることでリアルタイム分類時のset_datasに備える

            # for k in range(class_w):
            #     tmp = []
            #     tmp.append([wrist_motion[j][0] for j in range(len(wrist_motion)) if features_wrist[j] == (k)])
            #     tmp.append([wrist_motion[j][1] for j in range(len(wrist_motion)) if features_wrist[j] == (k)])
            #     tmp = np.array(tmp)
            #     scat_wrist = ax2.scatter(tmp[0], tmp[1], label=label__[k], cmap='viridis', edgecolor='blacK')
            # ax2.legend(labels=label__, fontsize=12)
            # ax2.set_title("wrist_pattern")
            # scat_wrist = ax2.scatter(0, 0, label="current", c="crimson", s=100,
            #                          marker="X")  # 空撃ちすることでリアルタイム分類時のset_datasに備える

            plt.pause(0.05)
            Train(listener).Show_emg_fb()
        if fb =="n":
            Train(listener).Show_emg_nfb(df,features_finger,finger_motion)

if __name__ == '__main__':
  main()