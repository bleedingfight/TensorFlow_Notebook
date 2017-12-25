import requests
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,LSTM
from keras.utils import np_utils
from os.path import exists
from keras.callbacks import EarlyStopping
def get_data(url):
    request = requests.get(url)
    name = url.split('/')[-1]
    if request.status_code== 200:
        with open(name,'w') as f:
            f.write(request.content.decode('utf-8'))
    else:
        print('sorry,network unstable')
def main():
    url = 'https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/10165151/macbeth.txt'
    name = url.split('/')[-1]
    if not exists(name):
        get_data(url)
    text = (open(name).read()).lower()
    unique_chars = sorted(list(set(text)))
    char_to_int = {}
    int_to_char = {}
    for i,c in enumerate(unique_chars):
        char_to_int.update({c:i})
        int_to_char.update({i:c})
    X = []
    Y = []
    for i in range(0,len(text)-50,1):
        sequence = text[i:i+50]
        label = text[i+50]
        X.append([char_to_int[char] for char in sequence])
        Y.append(char_to_int[label])
    X_modified = np.reshape(X,(len(X),50,1))
    X_modified = X_modified/float(len(unique_chars))
    Y_modified = np_utils.to_categorical(Y)
    if not exists('my.h5'):
        model = Sequential()
        model.add(LSTM(300,input_shape=(X_modified.shape[1],X_modified.shape[2]),
            return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(300))
        model.add(Dropout(0.2))
        model.add(Dense(Y_modified.shape[1],activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer = 'adam')
        model.fit(X_modified,Y_modified,epochs=10,batch_size=len(Y)//1000)
        EarlyStopping(monitor='val_loss',patience=0,verbose=0,mode='auto')
        model.save('my.h5')
    else:
        model = load_model('my.h5')
    start_index = np.random.randint(0,len(X)-1)
    new_string = X[start_index]
    for i in range(50):
        x = np.reshape(new_string,(1,len(new_string),1))
        x = x/float(len(unique_chars))
    pred_index = np.argmax(model.predict(x,verbose=0))
    char_out = int_to_char[pred_index]
    seq_in = [int_to_char[value] for value in new_string]
    new_string.append(pred_index)
    new_string = new_string[1:len(new_string)]
    print('----------------------input:---------------------------')
    print(''.join(seq_in))
    print('-----------------------output:-----------------------------')
    print(char_out)
if __name__ == '__main__':
    main()
