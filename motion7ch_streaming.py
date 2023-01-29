#   使用モジュール
import socket
import re
import json
import time
import matplotlib.pyplot as plt
from collections import deque

#   MR3からのHTTPstreaming用のクラス定義
class NoraxonHTTPClient:
    def __init__(self,port,host="127.0.0.1"):
        self.host=host
        self.port=port
        self.socket=0   #中身の仮置き

    #   ソケットの作成
    def make_socket(self):
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    #   MR3への接続
    def connect_MR3(self):
        self.socket.connect((self.host,self.port))

    #   MR3へリクエストを投げる
    def request2_MR3(self,cmd="samples"):
        GET_command = ('GET /'+cmd+'\nHost: %s\r\n\r\n' % (self.host))
        self.socket.send(GET_command.encode('utf-8'))


    #   MR3から帰ってきたデータを整形して受け取る
    def receive_from_MR3(self,msg_length):

        buffer=b""
        msg_fin= -1
        while msg_fin<0:
            chunk = self.socket.recv(msg_length)
            buffer+=chunk
            if len(chunk)<msg_length/2:
                msg_fin=1
        return buffer.decode("utf-8")


    #   初回のヘッダ情報を得るためのメソッド
    def initial_request(self):
        GET_command = ('GET /headers\nHost: %s\r\n\r\n' % (self.host))
        self.socket.send(GET_command.encode('utf-8'))

#   以下実行部分
if __name__ == "__main__":
    client=NoraxonHTTPClient(port=9220,host="127.0.0.1")
    client.make_socket()
    client.connect_MR3()
    client.initial_request()
    header_flag=True
    print("----------<header>-------------")
    while header_flag:
        msg = client.receive_from_MR3(4096 * 2)
        if "header" in msg:
            print(msg)
            header_flag=False
    print("----------</header>------------")
    queue1 = deque([0.0] * 200, maxlen=200)
    queue2 = deque([0.0] * 200, maxlen=200)
    queue3 = deque([0.0] * 200, maxlen=200)
    queue4 = deque([0.0] * 2000, maxlen=2000)

    fig, axs = plt.subplots(2,2,figsize=(12,6))
    axs = axs.flatten()
    lines = []
    index=0
    titles=["EMG.accel 1 Ax LT","EMG.accel 1 Ay LT","EMG.accel 1 Az LT","EMG.accel 3 Ax LT","EMG.accel 3 Ay LT","EMG.accel 3 Az LT"]
    for ax in axs:
        ax.set_xlim(0, 200)
        ax.set_ylim(-200, 200)
        ax.set_title(titles[index])
        lines.append(ax.plot([], [])[0])
        index+=1

    while True:
        # print("-----------------------------")
        client.request2_MR3()
        msg=client.receive_from_MR3(4096*2)
        if("channels" in msg and "index" in msg):
            # print(msg)
            msg = msg.replace("\n", "")
            msg=re.search("{.*]}", msg).group()

            if msg!=None:
                decoder = json.JSONDecoder()
                json_data = decoder.raw_decode(msg)
                print(json_data)
                for i in json_data[0]["channels"]:
                    # print(str(i["index"]) + " : " + str(i["samples"]))
                    if i["index"]==20:
                        queue1.extend(i["samples"])
                    if i["index"]==21:
                        queue2.extend(i["samples"])
                    if i["index"]==22:
                        queue3.extend(i["samples"])
                    if i["index"] == 23:
                        queue3.extend(i["samples"])

                    # if i["index"]==4:
                    #     queue4.extend(i["samples"])
                    # if i["index"]==5:
                    #     queue5.extend(i["samples"])
                    # if i["index"]==6:
                    #     queue6.extend(i["samples"])

            # Continuously add new data to the queue and update the plot
                # Add new data to the queue
            # Update the plot data
            for i in range(4):
                # Add new data to the queue
                queue = eval(f'queue{i + 1}')
                # Update the plot data
                lines[i].set_data(range(len(queue)), queue)
            # Redraw the plot
            fig.canvas.draw()
            plt.pause(0.01)

            # time.sleep(1)
            
            
# import tkinter as tk

# class App:
#     def __init__(self, master, n):
#         self.vars = [tk.IntVar() for _ in range(n)]
#         for i, var in enumerate(self.vars):
#             c = tk.Checkbutton(master, text=str(i), variable=var)
#             c.pack()
#         self.button = tk.Button(master, text="Get Selection", command=self.get_selection)
#         self.button.pack()
    
#     def get_selection(self):
#         selected = [i for i, var in enumerate(self.vars) if var.get()]
#         print(selected)

# root = tk.Tk()
# app = App(root, n=10)
# root.mainloop()
            
    
# import tkinter as tk
# import matplotlib
# import matplotlib.backends.backend_tkagg
# import matplotlib.pyplot as plt
# import random
# import time


# class App:
#     def __init__(self, master):
#         self.master = master
#         self.fig, self.ax = plt.subplots()
#         self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(self.fig, master=self.master)
#         self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
#         self.x = [0]
#         self.y = [0]
#         self.line, = self.ax.plot(self.x, self.y)
#         self.scale_x = tk.Scale(master, from_=0, to=100, orient="horizontal", command=self.set_xlim)
#         self.scale_x.pack()
#         self.scale_y = tk.Scale(master, from_=-100, to=100, orient="vertical", command=self.set_ylim)
#         self.scale_y.pack()
#         self.update()

#     def set_xlim(self, value):
#         value = float(value)
#         self.ax.set_xlim(0, value)
#         self.canvas.draw()

#     def set_ylim(self, value):
#         value = float(value)
#         self.ax.set_ylim(-value, value)
#         self.canvas.draw()

#     def update(self):
#         self.x.append(self.x[-1] + 1)
#         self.y.append(self.y[-1] + random.randint(-10, 10))
#         self.line.set_data(self.x, self.y)
#         self.ax.relim()
#         self.ax.autoscale_view()
#         self.canvas.draw()
#         self.master.after(100, self.update)


# root = tk.Tk()
# app = App(root)
# root.mainloop()
            
            
