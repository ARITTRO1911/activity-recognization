def myprocessor():
    import json
    #Acc code begin
    fp=open("../input/testtraindata/Test_Set/unknown.csv","w")
    #activity="standing"
    print("ax,ay,az,gx,gy,gz",file=fp)
    with open("acc.json") as data_file:
        my_list1 = [json.loads(line) for line in data_file]
    acclist = my_list1[ : 600]
    with open("gyro.json") as data_file2:
        my_list2 = [json.loads(line) for line in data_file2]
    gyrolist = my_list2[ : 600]
    count = 0
    while count <=599:
    
        acce = acclist[count]
        Xaxis = float(format(float(acce['x']),'.4f'))
        Yaxis = float(format(float(acce['y']),'.4f'))
        Zaxis = float(format(float(acce['z']),'.4f'))
        #Calculate Eucledian Distance of Accelerometer
        #EdA= math.sqrt(math.pow(float(Xaxis),2)+math.pow(float(Yaxis),2)+math.pow(float(Zaxis),2))
    
        gyro = gyrolist[count]
        pitch=float(format(float(gyro['x']),'.4f'))
        roll=float(format(float(gyro['y']),'.4f'))
        yaw=float(format(float(gyro['z']),'.4f'))
        #Calculate Eucledian Distance of Gyroscope
        #EdG= math.sqrt(math.pow(float(pitch),2)+math.pow(float(roll),2)+math.pow(float(yaw),2))
    
    # Start writing total 251 records in file starting from 300 records onwards 
        if count>299 and count < 550:       
        	#print count,Xaxis,",",Yaxis,",",Zaxis,",",pitch,",",roll,",",yaw
        	print(Xaxis,",",Yaxis,",",Zaxis,",",pitch,",",roll,",",yaw,file=fp)
        count=count+1
    
    fp.close()
    print ("Acceleration and Gyroscope data combined successfully")
    return