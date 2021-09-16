from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread
import myo
import time
import numpy as np
import keyboard
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from socket import socket, AF_INET, SOCK_DGRAM
import tkinter

emg_train=np.array([[] for i in range(8)])
emg_test=np.array([[] for i in range(8)])
features_2d = deque(maxlen=6)
features_2d.append([0,0])
features_2d.append([0,0])
features_2d.append([0,0])
features_2d.append([0,0])
features_2d.append([0,0])

print(emg_train.shape )
finger_label=[]
wrist_label=[]
emg_features=[]
lda_finger=LinearDiscriminantAnalysis(n_components=2)
lda_wrist=LinearDiscriminantAnalysis(n_components=2)
count=0
ovr_l = 32
win_l = 64
finger_predict=1 #初回はnomotion
wrist_predict=1
ADDR = '127.0.0.1'
PORT_TO = 50007 #送信ポート
M_SIZE = 1024

ax=0    #pltのリアルタイムプロット用のグローバル変数
scat=0

def update_plot(scat):
    tmp = np.array([x for x in list(features_2d)])
    scat.set_offsets(tmp)
    plt.pause(0.008)

class EmgCollector(myo.DeviceListener):
  """
  Collects EMG data in a queue with *n* maximum number of elements.
  """
  def __init__(self, n):
    self.n = n
    self.idle=False  #Trueの間計測を行う
    self.testflg=False
    self.lock = Lock()
    self.time_pre=0
    self.emg_data_list = [[] for i in range(8)]
    self.emg_data_queue = deque(maxlen=64)
    self.start_time=time.time()

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

  def start_test(self):
      self.testflg=True

  def predict(self):
      global scat, ax
      tmp=self.get_emg_queue()
      tmp=np.array([x[1] for x in tmp]).T
      test_features=[feature_calc(tmp, win_l)]
      df = pd.DataFrame(test_features)
      global finger_predict,wrist_predict
      finger_predict=lda_finger.predict(df.values)
      wrist_predict=lda_wrist.predict(df.values)
      #print(motion_predict)
      a= lda_finger.transform(df.values)
      #print(a)
      global features_2d
      features_2d.append(a[0])

  def end(self):
      self.idle=False

  def on_emg(self, event):
    with self.lock:
        if self.idle:
            for i in range(8):
                self.emg_data_list[i].append(event.emg[i])

        if self.testflg:
            self.emg_data_queue.append((event.timestamp, event.emg))
            time_cu=time.time()
            if time_cu-self.time_pre>0.3:
                self.time_pre = time_cu
                thread = Thread(target=self.predict)
                thread.start()



class Train(object):
    def __init__(self,listener):
        self.n = listener.n
        self.listener = listener
    # def get_emg(self):
    #     emg_data=self.listener.get_emg_data()
    #     emg_data=np.array([x[1] for x in emg_data]).T



    def main(self):
        flag_pressed=False
        flag_pre=False
        flag_start=False
        tmp_w=1
        tmp=0
        global finger_label,wrist_label
        label_ = ["nomotion","flexion", "pronation", "supination"]
        print(label_[tmp_w-1],"motion",tmp+1)
        while True:
            if keyboard.is_pressed("space"):
                flag_pressed = True
                if flag_pressed and not flag_pre and not flag_start:
                    print("start...")
                    flag_start=True
                    self.listener.start()

                elif flag_pressed and not flag_pre and flag_start:
                    print("...end")
                    self.listener.end()
                    global emg_train
                    emg_data=self.listener.get_emg_data()
                    emg_train=np.concatenate([emg_train,np.array(emg_data) ],1)
                    tmp=int(input("class? : "))
                    finger_label.extend([tmp for i in range(len(emg_data[0]))])
                    wrist_label.extend([tmp_w for i in range(len(emg_data[0]))])
                    flag_start = False
                    if tmp==5:
                        tmp_w+=1
                        tmp=0
                        if(tmp_w==5):
                            break
                    print("next", label_[tmp_w - 1], "motion", tmp + 1)
            flag_pre=flag_pressed
            flag_pressed=False
            plt.pause(1.0 / 30)

    def Test_emg(self):
        global ax,features_2d,scat
        snd=socket(AF_INET,SOCK_DGRAM)

        self.listener.start_test()
        while True:
            update_plot(scat)
            msg=str(finger_predict)+str(wrist_predict)
            snd.sendto(msg.encode(),(ADDR,PORT_TO))
            if keyboard.is_pressed("space"):  # スペースでテスト終了
                self.listener.end()
                break


def feature_calc(emg,win_l):
    FEATURES = []
    for i in range(8):
        tmp = emg[i]
        FEATURES.append(np.mean(np.abs(tmp)))  # MAV
        FEATURES.append(np.var(tmp))  # VAR
        zero = 0
        for j in range(0, win_l - 1):
            if tmp[j] * tmp[j + 1] < 0:
                zero += 1
        FEATURES.append(zero)
        diff = np.diff(tmp, n=1)
        FEATURES.append(np.sum(np.abs(diff)))  # WL
        # freq = np.abs(np.fft.fft(tmp))  # 周波数領域
        # FEATURES.append(np.max(freq))  # PKF
        # FEATURES.append(np.mean(freq))  # MKF

    return FEATURES

scatter=0

def main():
  # ovr_l=32
  # win_l=64
  class_n=5
  myo.init()
  hub = myo.Hub()
  listener = EmgCollector(512)
  #label_ = ["flexion", "pronation", "supination"]
  features_finger=[]
  features_wrist=[]
  with hub.run_in_background(listener.on_event):

    while True:
        tmp = input("record finger EMG? y/n: ")
        if tmp == "y":
            Train(listener).main()
        print("LDA training start")
        global finger_label
        for i in range(int(len(emg_train[0]) / ovr_l - 1)):
            tmp = i * ovr_l
            emg_features.append(feature_calc(emg_train[:,tmp:tmp+win_l],win_l))
            features_finger.append(finger_label[tmp])
            features_wrist.append(wrist_label[tmp])
        df = pd.DataFrame(emg_features)
        tmp=np.array(features_finger)
        np.savetxt("features_finger.txt", tmp, fmt='%s', delimiter=',')
        tmp = np.array(features_wrist)
        np.savetxt("features_wrist.txt", tmp, fmt='%s', delimiter=',')
        df.to_csv("EMG_features.csv")   #データ一回作っておこう8/22

        #lda_finger = LinearDiscriminantAnalysis(n_components=2)

        global lda_finger,lda_wrist
        # df = pd.read_csv('EMG_features.csv', header=0, index_col=0)
        # features_finger = np.loadtxt("features_finger.txt")
        finger_motion = lda_finger.fit(df.values, features_finger).transform(df.values)
        lda_wrist.fit(df.values, features_wrist)

        fig = plt.figure(figsize=(18, 12))
        global ax,scat
        ax = plt.subplot(121)
        label_ = ["nomotion","fist", "point", "pinch", "internal"]
        for k in range(class_n):
            tmp = []
            tmp.append([finger_motion[j][0] for j in range(len(finger_motion)) if features_finger[j] == (k + 1)])
            tmp.append([finger_motion[j][1] for j in range(len(finger_motion)) if features_finger[j] == (k + 1)])
            tmp = np.array(tmp)
            scat=ax.scatter(tmp[0], tmp[1], label=label_[k], cmap='viridis', edgecolor='blacK')
        scat = ax.scatter(0, 0, label="current", c="crimson",s=250,marker="X") #空撃ちすることでリアルタイム分類時のset_datasに備える
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)
        plt.title('LDA', fontsize=20)
        plt.pause(0.05)
        tmp = input("start Test? y/n: ")
        if tmp == "y":
            Train(listener).Test_emg()

if __name__ == '__main__':
  main()