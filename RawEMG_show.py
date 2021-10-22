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
import tkinter as tk

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

finger_2d = queue_init(2,6)
wrist_2d = queue_init(2,6)
finger_predict = queue_init(1,6)
wrist_predict = queue_init(1,6)

print(emg_train.shape )
finger_label=[]
wrist_label=[]
emg_features=[]
lda_finger=LinearDiscriminantAnalysis(n_components=2)
lda_wrist=LinearDiscriminantAnalysis(n_components=2)
count=0
ovr_l = 32
win_l = 64
do_record=True
class_n = 3  # 指
class_m = 4  # 手首

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
    print(results)
    return results

def update_plot(scat1,scat2):
    tmp = np.array([x for x in list(finger_2d)])
    tmp2 = np.array([x for x in list(wrist_2d)])
    scat1.set_offsets(tmp)
    scat2.set_offsets(tmp2)
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
      global scat_finger, ax1
      tmp=self.get_emg_queue()
      tmp=np.array([x[1] for x in tmp]).T
      test_features=[feature_calc(tmp, win_l)]
      df = pd.DataFrame(test_features)
      global finger_predict,wrist_predict
      finger_predict.append(lda_finger.predict(df.values))
      wrist_predict.append(lda_wrist.predict(df.values))
      #print(motion_predict)
      a= lda_finger.transform(df.values)
      b= lda_wrist.transform(df.values)
      #print(a)
      global finger_2d,wrist_2d
      finger_2d.append(a[0])
      wrist_2d.append(b[0])

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

    def Test_emg(self,feedback):
        global ax1,finger_2d,scat_finger,ax2,wrist_2d,scat_wrist,finger_predict,wrist_predict
        snd=socket(AF_INET,SOCK_DGRAM)
        self.listener.start_test()
        if feedback==1:
            while True:
                update_plot(scat_finger,scat_wrist)
                msg=str(finger_predict)+str(wrist_predict)
                snd.sendto(msg.encode(),(ADDR,PORT_TO))
                if keyboard.is_pressed("space"):  # スペースでテスト終了
                    self.listener.end()
                    break
        if feedback==0:
            fig = plt.figure(figsize=(18, 12))
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)
            motions=[]
            hist_finger=ax1.hist(hist_data(finger_predict,class_n))
            wrist_finger=ax2.hist(hist_data(wrist_predict,class_m))
            while True:
                ax1.cla()
                ax2.cla()
                ax1.set_title("finger motion")
                ax2.set_title("wrist motion")
                ax1.bar([1,2,3],hist_data(finger_predict, class_n))
                print(finger_predict, class_n)
                ax2.bar([1,2,3,4],hist_data(wrist_predict, class_m))
                ax1.set_ylim(0,6)
                ax1.set_xlim(0,class_n)
                ax2.set_ylim(0,6)
                ax2.set_xlim(0,class_m)
                plt.pause(1.0 / 30)


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
#------------------------------------設定ウィンドウのUI
    root = tk.Tk()
    root.geometry('300x200')
    root.title('初期数値')
    lbl1 = tk.Label(text='時間窓長さ')
    lbl1.place(x=20, y=70)
    ovr = tk.Entry(width=20)
    ovr.place(x=90, y=70)
    ovr.insert(tk.END,"30")
    button = tk.Button(text="go record")
    button.place(x=50, y=110)
    button2 = tk.Button(text="go test")
    button2.place(x=150, y=110)
    lbl2 = tk.Label(text='オーバーラップ')
    lbl2.place(x=20, y=95)
    win = tk.Entry(width=20)
    win.place(x=90, y=95)
    win.insert(tk.END, "60")
    def click_setting():
        global ovr_l,win_l,do_record
        ovr_l = int(ovr.get())
        win_l = int(win.get())
        do_record=True
        root.destroy()

    def click_setting2():
        global ovr_l, win_l,do_record
        ovr_l = int(ovr.get())
        win_l = int(win.get())
        do_record=False
        root.destroy()

    button["command"] = click_setting
    button2["command"] = click_setting2
    root.mainloop()
#-----------------------------------------
  # ovr_l=32
  # win_l=64
    myo.init()
    hub = myo.Hub()
    listener = EmgCollector(512)

    features_finger=[]
    features_wrist=[]
    with hub.run_in_background(listener.on_event):
        global lda_finger, lda_wrist
        if do_record==True:
            print("record")
            while True:
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


        if do_record==False:
            print("test")
            df = pd.read_csv('EMG_features.csv', header=0, index_col=0)
            features_finger = np.loadtxt("features_finger.txt")
            features_wrist = np.loadtxt("features_wrist.txt")

        finger_motion = lda_finger.fit(df.values, features_finger).transform(df.values)
        wrist_motion = lda_wrist.fit(df.values, features_wrist).transform(df.values)

        fig = plt.figure(figsize=(18, 12))
        global ax1,ax2,scat_finger,scat_wrist
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        label_ = ["nomotion","fist", "point"]
        label__= ["nomotion","flexion", "pronation", "supination"]
        for k in range(class_n):    #手描画
            tmp = []
            tmp.append([finger_motion[j][0] for j in range(len(finger_motion)) if features_finger[j] == (k + 1)])
            tmp.append([finger_motion[j][1] for j in range(len(finger_motion)) if features_finger[j] == (k + 1)])
            tmp = np.array(tmp)
            scat_finger=ax1.scatter(tmp[0], tmp[1], label=label_[k], cmap='viridis', edgecolor='blacK')
        ax1.set_title("finger_pattern")
        ax1.legend(labels=label_, fontsize=12)
        scat_finger = ax1.scatter(0, 0, label="current", c="crimson", s=250, marker="X") #空撃ちすることでリアルタイム分類時のset_datasに備える

        for k in range(class_m):
            tmp=[]
            tmp.append([wrist_motion[j][0] for j in range(len(wrist_motion)) if features_wrist[j] == (k + 1)])
            tmp.append([wrist_motion[j][1] for j in range(len(wrist_motion)) if features_wrist[j] == (k + 1)])
            tmp = np.array(tmp)
            scat_wrist = ax2.scatter(tmp[0], tmp[1], label=label__[k], cmap='viridis', edgecolor='blacK')
        ax2.legend(labels=label__, fontsize=12)
        ax2.set_title("wrist_pattern")
        scat_wrist = ax2.scatter(0, 0, label="current", c="crimson", s=250, marker="X")  # 空撃ちすることでリアルタイム分類時のset_datasに備える
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)
        #plt.title('LDA', fontsize=20)

    plt.pause(0.05)
    tmp = input("start Train with FeedBuck? y/n: ")
    if tmp == "y":
        Train(listener).Test_emg(1)
    if tmp =="n":
        plt.gca().clear()  # 確認で表示したクラスター図の削除
        Train(listener).Test_emg(0)
if __name__ == '__main__':
  main()
