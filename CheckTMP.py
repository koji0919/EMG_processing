from socket import socket, AF_INET, SOCK_DGRAM

ADDR = ''
PORT = 3333 #受信ポート
M_SIZE = 1024

rcv = socket(AF_INET, SOCK_DGRAM)
rcv.bind((ADDR, PORT))
a=0
b=2
while True:
    msg, address = rcv.recvfrom(8192)
    print(int(msg[0]-48)-int(msg[1]-48))

rcv.close()
