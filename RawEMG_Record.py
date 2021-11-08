from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread
import myo
import time
import numpy as np
import keyboard
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

print(emg_train.shape )
finger_label=[]
emg_features=[]
lda_finger=LinearDiscriminantAnalysis(n_components=2)
count=0
ovr_l = 32
win_l = 64
queue_len=6
do_record=True

class_f = 7

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

class EmgCollector(myo.DeviceListener):
  """
  Collects EMG data in a queue with *n* maximum number of elements.
  """
  def __init__(self, n):
    self.n = n
    self.idle=False  #Trueの間計測を行う
    self.testflg=False
    self.lock = Lock()
    self.emg_data_list = [[] for i in range(8)]
    self.emg_data_queue = deque(maxlen=win_l)  #テスト時の取得筋電位データの数はここで決定
    self.num=0
  def get_emg_data(self):
    with self.lock:
      return list(self.emg_data_list)

  def get_emg_queue(self):
    with self.lock:
      return list(self.emg_data_queue)

  def on_connected(self, event):
    event.device.stream_emg(True)

  def start(self):
      self.emg_data_list = [[] for i in range(8)]
      self.idle=True
      self.num=0

  def end(self):
      self.idle=False
      print(self.num)
  def on_emg(self, event):
    with self.lock:
        if self.idle:
            self.num+=1
            tmp=event.emg
            for i in range(8):
                self.emg_data_list[i].append(tmp[i])

class Record(object):
    def __init__(self,listener):
        self.n = listener.n
        self.listener = listener

    def main(self):
        flag_pressed=False
        flag_pre=False
        flag_start=False
        tmp_w=0
        tmp_f=0
        global finger_label
        fingermotion = ["fist", "point","wave in","wave out","spread","nomotion","fox"]
        print(fingermotion[tmp_f])
        starttime=0
        while True:
            if keyboard.is_pressed("space"):
                flag_pressed = True
                if flag_pressed and not flag_pre and not flag_start:
                    print("start...")
                    flag_start=True
                    starttime=time.perf_counter()
                    self.listener.start()

            if time.perf_counter()-starttime>8 and flag_start:
                print("...end")
                self.listener.end()
                global emg_train
                emg_data=self.listener.get_emg_data()
                emg_train=np.concatenate([emg_train,np.array(emg_data) ],1)
                print(len(emg_data))
                print(len(emg_data[0]))
                finger_label.extend([tmp_f for i in range(len(emg_data[0]))])
                flag_start = False
                tmp_f+=1
                if tmp_f==class_f:
                    break
                time.sleep(1.0)
                print("next ",fingermotion[tmp_f])
            flag_pre=flag_pressed
            flag_pressed=False

def feature_calc(emg,win_l):    #特徴量計算
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
    ovr_l = 32
    win_l = 64
    queue_len = 64

    finger_2d = queue_init(2,queue_len)
    finger_predict = queue_init(1,queue_len)
#-----------------------------------------
    myo.init()
    hub = myo.Hub()
    listener = EmgCollector(512)

    features_finger=[]
    with hub.run_in_background(listener.on_event):
        global lda_finger, lda_wrist
        print("record")
        Record(listener).main()
        print("LDA training start")
        global finger_label
        for i in range(int(len(emg_train[0]) / ovr_l - 1)):
            tmp = i * ovr_l
            tmp1=feature_calc(emg_train[:,tmp:tmp+win_l],win_l)
            tmp1.append(finger_label[tmp])
            emg_features.append(tmp1     )
            features_finger.append(finger_label[tmp])
        df = pd.DataFrame(emg_features)
        tmp=np.array(features_finger)
        df.to_csv("test.csv")   #実験用のデータを保存する

        finger_motion = lda_finger.  fit(df.values, features_finger).transform(df.values)
        global ax1,scat_finger
        ax1 = plt.subplot(111)
        label_ = ["fist", "point","wave in","wave out","spread","nomotion","fox"]
        for k in range(class_f):    #手描画
            tmp = []
            tmp.append([finger_motion[j][0] for j in range(len(finger_motion)) if features_finger[j] == (k)])
            tmp.append([finger_motion[j][1] for j in range(len(finger_motion)) if features_finger[j] == (k)])
            tmp = np.array(tmp)
            scat_finger=ax1.scatter(tmp[0], tmp[1], label=label_[k], cmap='viridis', edgecolor='blacK')
        ax1.set_title("finger_pattern")
        ax1.legend(labels=label_, fontsize=12)
        scat_finger = ax1.scatter(0, 0, label="current", c="crimson", s=250, marker="X") #空撃ちすることでリアルタイム分類時のset_datasに備える
        plt.show()

if __name__ == '__main__':
  main()