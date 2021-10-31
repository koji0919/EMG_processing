using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Text;

public class record_trigger : MonoBehaviour
{
    public string host = "127.0.0.1";
    public int port = 3333;
    private UdpClient client;
    private int a=10;
    string msg="";
    public int f=0,w=0;
    void Start()
    {
        client = new UdpClient();
        client.Connect(host, port);
    }

    // Update is called once per frame
    void Update()
    {
        a=a+1;
        Debug.Log(a);
        if(a>100){
        Debug.Log("------------------------");    
        msg=f.ToString()+w.ToString();
        byte[] dgram = Encoding.UTF8.GetBytes(msg);
        client.Send(dgram, dgram.Length);
        a=0;
        }
    }
        void OnApplicationQuit(){
        client.Close();
    }
}
