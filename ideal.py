import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.layers import Conv1D, Conv2D,MaxPooling1D, MaxPooling2D, Flatten 
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from kerastuner.tuners import RandomSearch, BayesianOptimization
import pickle
import sys
assert tf.test.is_gpu_available() == True

import data
import time
EPOCHS = 32 # 32 is ideal
BATCH_SIZE = 128 
LEARNING_RATE = 0.001
NAME = f"HOURLY_BTC_{data.SEQ_LEN}-SEQ-{data.FUTURE_PREDICT_PERIOD}-PERIOD-{BATCH_SIZE}-BATCH-{LEARNING_RATE}_Lr-BBANDS-{int(time.time())}"

LOG_DIR = f"KTUNER_logs"

def build():
    model = Sequential()

    model.add(LSTM(192, input_shape=(data.train_X.shape[1:]), return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(LSTM(64, input_shape=(data.train_X.shape[1:]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.7))

    model.add(Dense(2, activation="softmax"))
    
    return model

predict = False
try:
    if sys.argv[1] == "load":
        model = load_model('models/rnn_current')
        print("Model loaded successfully!")
    if len(sys.argv) >= 3 and sys.argv[2] == "predict":
        predict = True
except:
    model = build()
    print("Model not loaded, building from scratch.")

if predict == True and model is not None:
    
    prediction = model.predict(data.val_X)
    print(prediction)
    with open('prediction.pkl', 'wb') as f:
        pickle.dump(prediction, f)
        print("Saved predictions. :)")
else:
    opt = tf.keras.optimizers.Adam(lr=LEARNING_RATE, decay=1e-6)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    tb = TensorBoard(log_dir=f'logs/{NAME}', update_freq='epoch', profile_batch=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=2, min_delta=0.01)

    filepath = "RNN_Final-test"

    checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc',verbose=1, save_best_only=True,mode='max')) # saves only the best one
    history = model.fit(
      data.train_X, data.train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(data.val_X, data.val_y),
        callbacks=[tb, checkpoint, reduce_lr]
    )

    save = input("Save the model? (y/n)")

    try:
        if save == "y": 
            model.save("models/rnn_current")
            print("Model saved successfully!")
        else:
            print("Model not saved.")
    except:
        print("Model not saved, an error occured.")
