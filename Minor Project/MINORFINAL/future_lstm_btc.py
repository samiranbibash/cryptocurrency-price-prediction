
import pred
from pred import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
import minor 
from minor import *
from sklearn.preprocessing import MinMaxScaler

import math
from sklearn.metrics import mean_squared_error
from datasplit_bitcoin import *
def tempor():
    future_prediction(x_input,temp_input,steps,pred_days)
model = tf.keras.models.load_model('lstmmodel_btc')
model.summary()

def baklol():
    
    # EVALUATION ON TEST DATA
    model.evaluate(x_test, y_test)
    
    trainpredict = model.predict(x_train)
    testpredict = model.predict(x_test)
    
    #TRANSFORMING DATA BACK TO ORIGINAL FORM
    train_predict = scaler.inverse_transform(trainpredict)
    test_predict = scaler.inverse_transform(testpredict)
    hey_lol = scaler.inverse_transform(y_train.reshape(-1,1))
    hey_lol2 = scaler.inverse_transform(y_test.reshape(-1,1))
    
    #Calculate RMSE performance metrics
    
    math.sqrt(mean_squared_error(hey_lol,train_predict))
    
    #Test Data RMSE
    math.sqrt(mean_squared_error(hey_lol2,test_predict))
    def MAPE(Y_actual,Y_Predicted):
        mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
        return mape
    LR_MAPE= MAPE(hey_lol,train_predict)
    print("MAPE: ",LR_MAPE)
    #plt.plot(history.history['val_loss'], color = 'blue')
    #plt.title('model loss')
    #plt.xlabel('epochs')
   # plt.ylabel('validation loss')
    #plt.legend(loc='upper left')
    #plt.show()
    
    
    testpredict.shape
    
    # COPIED ACTUAL TEST DATA
    new_y = y_test
    
    #Converting data back to original form 
    new_y = scaler.inverse_transform(y_test.reshape(-1,1))
    
    # Predicting for the next n 'days'
    
    test_index = len(test_data) - steps
    
    test_data[test_index:]
    
    i = 0
    new_list = []
    x_input=test_data[test_index:].reshape(1,-1)
    print(x_input.shape)
    
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    
    pred_days = pred.b
    
    # PREDICTION FUNCTION FOR N DAYS
    def future_prediction(x_input, temp_input, steps, pred_days):
      lst_output=[]
      i=0
      while(i<pred_days):
          
          if(len(temp_input)>steps):
              x_input=np.array(temp_input[1:])
              print("{} day input {}".format(i,x_input))
              x_input=x_input.reshape(1,-1)
              x_input = x_input.reshape((1, steps, 1))
              next_day = model.predict(x_input)
              print("{} day output {}".format(i,next_day))
              temp_input.extend(next_day[0].tolist())
              temp_input=temp_input[1:]
              lst_output.extend(next_day.tolist())
              i=i+1 
          else:
              x_input = x_input.reshape((1,steps,1))
              next_day= model.predict(x_input)
              temp_input.extend(next_day[0].tolist())
              lst_output.extend(next_day.tolist())
              i=i+1
      return lst_output
    
    lst_output = future_prediction(x_input, temp_input, steps, pred_days)
    ytrainforplot = scaler.inverse_transform(y_train.reshape(-1,1))
    # print("\n \n \n \n \n \n \n \n \n TRAINING = ", ytrainforplot[:20])
    
    ytestforplot = scaler.inverse_transform(y_test.reshape(-1,1))
    # print("\n \n \n \n \n \n \n \n \n TEST = ", ytestforplot[:20])
    
    # CONVERTING THE OBTAINED RESULT INTO ORIGINAL FORM
    predicted_prices = scaler.inverse_transform(lst_output)
    
    
    allfinalprices = ytrainforplot
    actualprices_plot = np.append(allfinalprices, ytestforplot)
    print("\n \n \n \n \n ",actualprices_plot.shape)
    
    
    temp_index = np.linspace(actualprices_plot.shape[0]-1, actualprices_plot.shape[0]+9, num = 20) 
    #Plotting new data
    
    
    plt.title("Predicted Bitcoin Prices")
    
    plt.plot(actualprices_plot, label = 'Historical Prices')
    
    plt.plot(temp_index, predicted_prices, 'y-x', label = 'Predicted Prices')
    plt.legend()
    #real_y_value = scaler.inverse_transform(real_y_value)
    
    #real_y_value.shape
    
    #plt.plot(real_y_value)
    
    #newprices = np.append(real_y_value, newdata1)
    
    #newprices.shape
    return

def performance_plot():
    plt.plot(train_predict, 'r-o')
    plt.plot(y_train, 'b-x')