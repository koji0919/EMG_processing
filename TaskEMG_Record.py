from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread
import myo
import time
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from socket import socket, AF_INET, SOCK_DGRAM

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
finger_predict = queue_init(1,10)
wrist_predict = queue_init(1,10)

testdata_2d_rec=[]  #評価タスク時のデータを格納

print(emg_train.shape )
finger_label=[]
emg_features=[]

task_data=[]
lda_finger=LinearDiscriminantAnalysis(n_components=2)
count=0
ovr_l = 20
win_l = 40
fps=0.001
queue_len=6
do_record=True
class_n = 10  # 指

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
    self.lock = Lock()
    self.time_pre=0
    self.emg_data_list = [[] for i in range(8)]
    self.emg_data_queue = queue_init(1,win_l)  #テスト時の取得筋電位データの数はここで決定
    self.start_time=time.time()
    self.sampleamount = 0

  def get_emg_queue(self):
    with self.lock:
      return list(self.emg_data_queue)

  def on_connected(self, event):
    event.device.stream_emg(True)

  def start(self):
      for i in range(ovr_l):
          self.emg_data_queue.append((0, [0, 0, 0, 0, 0, 0, 0, 0]))
      self.idle=True

  def predict(self):
      #global scat_finger, ax1
      tmp=self.get_emg_queue()
      tmp=np.array([x[1] for x in tmp]).T
      test_features=[feature_calc(tmp, win_l)]
      df = pd.DataFrame(test_features)
      global finger_predict,testdata_2d_rec
      lda_fingerresult=int(lda_finger.predict(df.values))
      finger_predict.append(lda_fingerresult)
      a= lda_finger.transform(df.values)
      global finger_2d
      finger_2d.append(a[0])
      testdata_2d_rec.append([lda_fingerresult,a[0][0],a[0][1]])

  def end(self):
      self.idle=False

  def on_emg(self, event):
    with self.lock:
        if self.idle:
            tmp=event.emg
            self.sampleamount+=1
            self.emg_data_queue.append((event.timestamp, tmp))
            if self.sampleamount==ovr_l:
                self.sampleamount=0
                thread = Thread(target=self.predict)
                thread.start()
                thread.join(0.0001)

class Record(object):
    def __init__(self,listener):
        self.n = listener.n
        self.listener = listener
        self.rcv=socket(AF_INET, SOCK_DGRAM)
        self.VR_test_start=0

    def main(self):
        print("wait msg")
        ADDR = ''
        PORT = 50004  # 受信ポート
        M_SIZE = 1024
        msg=0
        self.rcv.bind((ADDR, PORT))
        while True:
            msg, address = self.rcv.recvfrom(8192)
            msg_=int(msg[0] - 48)

            if self.VR_test_start==0 and msg_==1:
                self.VR_test_start=1
                global testdata_2d_rec
                testdata_2d_rec=[]
                print("msg received record start")
                self.listener.start()

            if self.VR_test_start==1 and msg_==9:
                self.VR_test_start=0
                self.listener.end
                print("msg received record finished")
                self.save_file(msg.decode()[1:]+".csv")

    def save_file(self,filename):
        df=pd.DataFrame(testdata_2d_rec)
        df.to_csv(filename)  # データを保存

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
        FEATURES.append(zero)   #ZC
        diff = np.diff(tmp, n=1)
        FEATURES.append(np.sum(np.abs(diff)))  # WL
        # freq = np.abs(np.fft.fft(tmp))  # 周波数領域
        # FEATURES.append(np.max(freq))  # PKF
        # FEATURES.append(np.mean(freq))  # MKF

    return FEATURES

def main():
#-----------------------------------------
    myo.init()
    hub = myo.Hub()
    listener = EmgCollector(512)

    with hub.run_in_background(listener.on_event):
        global lda_finger, lda_wrist
        print("Task")
        df = pd.read_csv('tmp.csv', header=0, index_col=0)
        features_finger = df.values[:, -1]
        df = df.values[:, 0:-1]
        finger_motion = lda_finger.fit(df, features_finger)
        Record(listener).main()


if __name__ == '__main__':
  main()