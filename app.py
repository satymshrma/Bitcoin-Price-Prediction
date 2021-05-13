import os
from flask import Flask, render_template, url_for, flash, request, redirect
#from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
#from flask_sqlalchemy import SQLAlchemy



#UPLOAD_FOLDER='/uploads'
#ALLOWED_EXTENSIONS={'csv'}

app = Flask(__name__)
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
#app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
#db= SQLAlchemy(app)

#class Todo(db.Model):
#    id=db.Column(db.integer,primary_key=True)




@app.route('/')

def index():
    return render_template('index.html')


'''
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return 
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
'''

@app.route('/prediction', methods=['POST'])
def predict():
    if request.method=='POST':
        #start predicting
        data = pd.read_csv('BTC-USD.csv', date_parser = True)
        data_training = data[data['Date']< '2020-01-01'].copy()
        data_test = data[data['Date']> '2020-01-01'].copy()
        training_data = data_training.drop(['Date', 'Adj Close'], axis = 1)
        scaler = MinMaxScaler()
        training_data = scaler.fit_transform(training_data)
        X_train = []
        Y_train = []
        for i in range(60, training_data.shape[0]):
            X_train.append(training_data[i-60:i])
            Y_train.append(training_data[i,0])
        X_train, Y_train = np.array(X_train), np.array(Y_train)
        regressor = Sequential()
        regressor.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
        regressor.add(Dropout(0.3))

        regressor.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
        regressor.add(Dropout(0.4))

        regressor.add(LSTM(units = 120, activation = 'relu'))
        regressor.add(Dropout(0.5))

        regressor.add(Dense(units =1))


        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        regressor.fit(X_train, Y_train, epochs = 20, batch_size =50)
        past_60_days = data_training.tail(60)
        df= past_60_days.append(data_test, ignore_index = True)
        df = df.drop(['Date', 'Adj Close'], axis = 1)
        inputs = scaler.transform(df)
        X_test = []
        Y_test = []
        for i in range (60, inputs.shape[0]):
            X_test.append(inputs[i-60:i])
            Y_test.append(inputs[i, 0])
        X_test, Y_test = np.array(X_test), np.array(Y_test)
        Y_pred = regressor.predict(X_test)
        scale = 1/5.18164146e-05
        Y_test = Y_test*scale
        Y_pred = Y_pred*scale

        #return render_template('index.html',Y_pred=Y_pred)

        plt.figure(figsize=(14,5))
        plt.plot(Y_test, color = 'red', label = 'Real Bitcoin Price')
        plt.plot(Y_pred, color = 'green', label = 'Predicted Bitcoin Price')
        plt.title('Bitcoin Price Prediction using RNN-LSTM')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig('plot.png', dpi=300, bbox_inches='tight')
        return render_template('index.html',Y_pred=plt.show())
#    else:
#        return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)