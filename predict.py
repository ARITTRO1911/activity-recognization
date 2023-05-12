def p_detect():
    import warnings
    import numpy as np
    import pandas as pd
    import os
    import time
    from sklearn.utils import shuffle
    from preprocessor import myprocessor
    import tensorflow as tf
    
    start=time.time()
    myprocessor()
    warnings.filterwarnings('ignore')
    os.listdir('../input/testtraindata/')
    base_train_dir = '../input/testtraindata/Train_Set/'
    base_test_dir = '../input/testtraindata/Test_Set/'
    
    test_data = pd.DataFrame(columns = ['ax','ay','az','gx','gy','gz'])
    files = os.listdir(base_test_dir)
    for f in files:
        df = pd.read_csv(base_test_dir+f)
        #df['activity'] = f.split('.')[0].split('_')[-1]
        test_data = pd.concat([test_data,df],axis = 0)
        
    test_data = shuffle(test_data)
    test_data.reset_index(drop = True,inplace = True)
    test_data.head()
    #Train Data
    train_data = pd.DataFrame(columns = ['activity','ax','ay','az','gx','gy'])
    train_folders = os.listdir(base_train_dir)
    
    for tf in train_folders:
        files = os.listdir(base_train_dir+tf)
        for f in files:
            df = pd.read_csv(base_train_dir+tf+'/'+f)
            train_data = pd.concat([train_data,df],axis = 0)
    train_data = shuffle(train_data)
    train_data.reset_index(drop = True,inplace = True)
    train_data.head()
    
    train_data['activity'] = train_data['activity'].str.strip()
    
    #train_dict = {'standing':1,'sitting':2,'walking':3,'laying':4}
    train_dict = {'standing':1,'sitting':2,'laying':3,'walking':4,'jogging':5,'running':6}
    train_data['activity'] = train_data['activity'].replace(train_dict)
    #test_data['activity'] = test_data['activity'].replace(train_dict)
    test_data.head()
    train_data.head()
    
    #prepare Data
    X_train = train_data.drop('activity',axis = 1)
    y_train = train_data['activity']
    X_test = test_data
    #y_test = test_data['activity']
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Fitting K-NN to the Training set(working on jogging and running but not walking)(sitting(working 100%)->sitting,standing(working ->100%)->standing,laying(working 100%)->laying)
    #from sklearn.neighbors import KNeighborsClassifier
    #classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
    #classifier.fit(X_train, y_train)
    
    #Fitting Decision tree Classification model on the training set(working on walking,jogging ,running(100%))(standing(not working 100%)->running,sitting-(working 100%)->sitting,laying (working100%)->laying)
    #from sklearn.tree import DecisionTreeClassifier
    #classifier = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
    #classifier.fit(X_train, y_train)   
   
    #Fitting RANDOM FOREST model on the training set( on walking->walking ,jogging->jogging,running->running)#combined random forest and decisin tree walking)( standing->running,sitting->sitting,jogging,running,laying(working fine100%)-laying)
    #from sklearn.ensemble import RandomForestClassifier
    #classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    #classifier.fit(X_train, y_train)
    
    #Fitting NAIVE BAYES CLASSIFIER MODEL on the training set(not working with walking,running but working with jogging)(working absolutely fine with standing ,sitting,laying )(walking->jogging,jogging->joggging,running->running)
    #from sklearn.naive_bayes import GaussianNB
    #classifier = GaussianNB()
    #classifier.fit(X_train, y_train)
    
    #Fitting KERNEL SVM model on the training set(not working with running,jogging,)//(working absolutely fine with standing and sitting and laying(sometimes))(walking->standing,jogging->standing,jogging,running->running)
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, y_train)
    
    #Fitting linear SVM model on the training set(working absolutely fine with standing ,sitting ,laying)(walking(not working->running),jogging(not working)->standing,running(not working) ->standing)
    #from sklearn.svm import SVC
    #classifier = SVC(kernel = 'linear', random_state = 0)
    #classifier.fit(X_train, y_train)
    
   
    
   #Trying with neural network(standing(not working->running,laying->,siting(not working100%)->laying(not working)->running jogging->standing,jogging,running->running )
    #from sklearn.neural_network import MLPClassifier
    #for i in 5, 10, 15:                    #### We are taking only one hidden layer, try with different number of layers
        #classifier = MLPClassifier(hidden_layer_sizes=(i,i,i),early_stopping=True,learning_rate='adaptive',learning_rate_init=0.003)
        #classifier.fit(X_train,y_train)
   
   
   
   #Trying with convolutional neural network   
    #from keras.models import Sequential
    #from keras.layers import Dense,LSTM,GRU,Dropout, Flatten
    #for s in 6,12,18:
        #classifier = Sequential()
        #classifier.add(LSTM(s, input_shape=(None, parameters), return_sequences= False))
	    #classifier.add(Dense(6, activation='sigmoid'))
	    #classifier.compile(optoimizer = 'adam' , loss='categorical_crossentropy', metrics=['accuracy'])

	    #print(model.summary())
	    #classifier.fit(X_train, y_train, epochs=2, batch_size=40)
    
    
    #Trying with convolutio neural network
    #from tensorflow.python import keras
    #from keras.models import Sequential
    #from keras.layers import Dense, Dropout , BatchNormalization
    #from sklearn.model_selection import train_test_split
    #from keras.utils import np_utils
    #from keras.optimizers import RMSprop, Adam
    
    #classifier = Sequential()

    #classifier.add(Dense(64, input_dim=X_train.shape[1] , activation='relu'))
    #classifier.add(Dense(64, activation='relu'))
    #classifier.add(BatchNormalization())
    #classifier.add(Dense(128, activation='relu'))
    #classifier.add(Dense(196, activation='relu'))
    #classifier.add(Dense(32, activation='relu'))
    #classifier.add(Dense(6, activation='sigmoid'))
    
    #classifier.compile(optimizer = Adam(lr = 0.0005),loss='categorical_crossentropy', metrics=['accuracy'])
    #print(model.summary())
    #classifier.fit(X_train, y_train , epochs=2 , batch_size = 256 )
   
    
   #trying with deep learning tutorial
   #from tensorflow.python import keras
   #from keras.models import Sequential
   #from keras.layers import Dense, Dropout , BatchNormalization
   #from sklearn.model_selection import train_test_split
   #from keras.utils import np_utils
   #from keras.optimizers import RMSprop, Adam
   
   #classifier = tf.keras.models.Sequential()
   #classifier.add(tf.keras.layers.Dense(units=6, activation='relu'))
   #classifier.add(tf.keras.layers.Dense(units=6, activation='relu'))
   #classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
   #classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
   #classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)
    
   
    
   
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    counts = np.bincount(y_pred)
    #print(counts)
    verdict=np.argmax(counts)
    #print(verdict)
    #choices = {1:'standing', 2:'sitting',3:'walking', 4:'laying'}
    choices = {1:'standing', 2:'sitting',3:'laying', 4:'walking', 5:'jogging', 6:'running'}
    result = choices.get(verdict, 'default')
    #print'The unknown activity is identified as::',result
    mytime=time.time()- start
    print ('Activity analysed in',mytime,' seconds')
    return result