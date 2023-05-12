#!/usr/bin/env python

import asyncio
import websockets
import socket
from predict import p_detect
from Data_gen import dataset
import time
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


hostname = socket.gethostname()
IPAddr = get_ip()
print("Your Computer Name is: " + hostname)
print("Your Computer IP Address is: " + IPAddr)
print(
    "* Enter {0}:5000 in the app.\n* Press the 'Set IP Address' button.\n* Select the sensors to stream.\n* Update the 'update interval' by entering a value in ms.".format(IPAddr))

############## Main Working Code ##################################
#print("NAME-AKANSHA ,  AGE = 32 ")
#print("NAME-MOHIT ,    AGE = 22 ")
#print("NAME-PRAMAY ,   AGE = 12 ")
print("NAME-RAJ ,      AGE = 57 ")
#print("NAME-SUDEEP ,   AGE = 30 ")


#print("TYPE OF MOTION :: STATIONARY(standing,sitting,laying)")
print("TYPE OF MOTION ::  KINETIC(walking,jogging,running)")

print("TYPE OF CLASSIFIER :: DECISION TREE")
#print("TYPE OF CLASSIFIER :: RANDOM FOREST")
#print("TYPE OF CLASSIFIER :: KERNEL SVM")
#print("TYPE OF CLASSIFIER :: LINEAR SVM")
#print("TYPE OF CLASSIFIER :: NAIVE BAYES")




async def echo(websocket, path):
    start=time.time()
    start1=time.time()
    done=0
    line=1
    flag=1
    async for message in websocket:
        if path == '/accelerometer':
            message = await websocket.recv()
            #data = json.loads(message)
            f = open("acc.json", "a")
            # Initial Posture
            if (time.time()- start1)>20 and line==1 and flag==1:
                posture= p_detect()
                print ("Posture detected as::",posture)
                #dataset()
                flag=0
                line=2
            # Refresh Posture
            if (time.time()- start1)>25 and line==2:
            	print("*****Refreshing Posture Information*****")
            	start1=time.time()
            	flag=1
            	line=1
            mytime=(time.time()- start)
            if not f.closed :
                    #print(message,file=f)
                    f.write(message+"\n")
                    #print("Writing ACC File")
                   # print("Time elapsed:",mytime)
               		
                    if mytime>15:
                        f.close()
                        done=1
                                   
            if mytime>25 and done==1:
               f=open("acc.json","w")
               start=time.time()
               done=0
              
		#Gyroscope record
        if path == '/gyroscope':
            start=time.time()
            message = await websocket.recv()
            #data = json.loads(message)
            #print(data)
            f1 = open("gyro.json", "a")
            mytime=(time.time()- start)
            if not f1.closed :
                    #print(message,file=f1)
                    f1.write(message+"\n")
                    if mytime>15:
                        f1.close()
                        done=1
                        print ("Time elapsed:",mytime)
                        
            if mytime>25 and done==1:
               f1=open("gyro.json","w")
               start=time.time()
               done=0
        
asyncio.get_event_loop().run_until_complete(websockets.serve(echo, '0.0.0.0', 5000, max_size=1_000_000_000))
asyncio.get_event_loop().run_forever()
