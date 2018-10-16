# Lab1 House Sale Price Prediction 

#1 Load file

    # read data
    data_train = pd.read_csv('train-v3.csv')
    x_train = data_train.drop(['price','id'],axis=1).values
    y_train = data_train['price'].values
    y_train = y_train.reshape((-1, 1))

    data_valid = pd.read_csv('valid-v3.csv')
    x_valid = data_valid.drop(['price','id'],axis=1).values
    y_valid = data_valid['price'].values
    y_valid = y_valid.reshape((-1, 1))

    data_test = pd.read_csv('test-v3.csv')
    x_test = data_test.drop('id',axis=1).values
    
#2 Normalization    

    X_train = preprocessing.scale(x_train)
    X_valid = preprocessing.scale(x_valid)
    X_test = preprocessing.scale(x_test)

#3 Training Neural Network

    model = Sequential()
    
    model.add(Dense(units=32,input_dim=x_train.shape[1], kernel_initializer='normal',activation='relu'))
    model.add(Dense(units=128,kernel_initializer='normal',activation='relu'))
    model.add(Dense(units=256,kernel_initializer='normal',activation='relu'))
    model.add(Dense(units=128,kernel_initializer='normal',activation='relu'))
    model.add(Dense(units=128,kernel_initializer='normal',activation='relu'))
    model.add(Dense(units=32,kernel_initializer='normal',activation='relu'))

    model.add(Dense(units=x_train.shape[1], kernel_initializer='normal',activation='relu'))
    #output layer
    model.add(Dense(units=1, kernel_initializer='normal'))
    
#4 Predict Neural Network

    model.compile(loss='MAE',optimizer='adam')

    train_history = model.fit(x=X_train,y=y_train,validation_data=(X_valid,y_valid),epochs=256,batch_size=32)
    Y_predict = model.predict(X_test)
    np.savetxt('output.csv',Y_predict,delimiter=',')
    
