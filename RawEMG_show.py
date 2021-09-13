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


emg_train=np.array([[] for i in range(8)])
emg_test=np.array([[] for i in range(8)])
features_2d = deque(maxlen=6)
features_2d.append([0,0])
features_2d.append([0,0])
features_2d.append([0,0])
features_2d.append([0,0])
features_2d.append([0,0])

print(emg_train.shape )
emg_label=[]
emg_features=[]
features_label=[]
lda_finger=LinearDiscriminantAnalysis(n_components=2)
count=0
ovr_l = 32
win_l = 64
motion_predict=3 #初回はnomotion
ADDR = '127.0.0.1'
PORT_TO = 50007 #送信ポート
M_SIZE = 1024

ax=0
scat=0

def update_plot(scat):
    tmp = np.array([x for x in list(features_2d)])
    #print(tmp)
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
      global motion_predict
      motion_predict=lda_finger.predict(df.values)
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
        print("EMG record phase press space to record")
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
                    global emg_label
                    emg_label.extend([tmp for i in range(len(emg_data[0]))])
                    #print(emg_train)
                    #print(emg_label)
                    flag_start = False
                    tmp=input("exit training? y/n: ")
                    if tmp=="y":
                        break
            flag_pre=flag_pressed
            flag_pressed=False
            plt.pause(1.0 / 30)

    def Test_emg(self):
        global count,lda_finger,ax,features_2d,scat
        classname = ["fist", "grab", "nomotion", "spread", "lateral", "pinch", "current"]
        snd=socket(AF_INET,SOCK_DGRAM)
        count = 0
        self.listener.start_test()
        while True:
            update_plot(scat)
            msg=str(motion_predict)
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

#def class_test():
    
scatter=0

def main():
  # ovr_l=32
  # win_l=64
  class_n=6
  myo.init()
  hub = myo.Hub()
  listener = EmgCollector(512)
  #label_ = ["flexion", "extension", "pronation", "supination"]
  features_label=[]
  with hub.run_in_background(listener.on_event):

    # while True:
    #     tmp = input("record EMG? y/n: ")
    #     if tmp == "y":
    #         Train(listener).main()
    #     print("LDA training start")
    #     global emg_label
    #     for i in range(int(len(emg_train[0]) / ovr_l - 1)):
    #         tmp = i * ovr_l
    #         emg_features.append(feature_calc(emg_train[:,tmp:tmp+win_l],win_l))
    #         features_label.append(emg_label[tmp])
    #     df = pd.DataFrame(emg_features)
    #
    #     tmp=np.array(features_label)
    #     np.savetxt("features_label.txt", tmp, fmt='%s', delimiter=',')
    #     df.to_csv("EMG_features.csv")   #データ一回作っておこう8/22

        #lda_finger = LinearDiscriminantAnalysis(n_components=2)

        global lda_finger
        df = pd.read_csv('EMG_features.csv', header=0, index_col=0)
        features_label = np.loadtxt("features_label.txt")
        finger_motion = lda_finger.fit(df.values, features_label).transform(df.values)


        fig = plt.figure(figsize=(18, 12))
        global ax,scat
        ax = plt.subplot(121)
        label_ = ["fist", "grab", "nomotion", "spread", "lateral", "pinch"]
        for k in range(6):
            tmp = []
            tmp.append([finger_motion[j][0] for j in range(len(finger_motion)) if features_label[j] == (k + 1)])
            tmp.append([finger_motion[j][1] for j in range(len(finger_motion)) if features_label[j] == (k + 1)])
            tmp = np.array(tmp)
            scat=ax.scatter(tmp[0], tmp[1], label=label_[k], cmap='viridis', edgecolor='blacK')

        scat = ax.scatter(0, 0, label="current", c="crimson",s=250,marker="X") #空撃ちすることでリアルタイム分類時のset_datasに備える
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)
        plt.title('LDA', fontsize=20)
        #plt.ion()
        plt.pause(0.05)
        tmp = input("start Test? y/n: ")
        if tmp == "y":
            Train(listener).Test_emg()

if __name__ == '__main__':
  main()