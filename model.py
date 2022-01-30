import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from kerastuner.tuners import RandomSearch, BayesianOptimization
#from kerastuner.hyperparameters import HyperParameters
import pickle
import sys

if len(sys.argv) < 2:
    print("Usage: python3.8 model.py <view/search>")
    exit()

assert tf.test.is_gpu_available() == True

import data
import time
EPOCHS = 20 
BATCH_SIZE = 128 
NAME = f"HOURLY_BTC_{data.SEQ_LEN}-SEQ-{data.FUTURE_PREDICT_PERIOD}-PERIOD-{BATCH_SIZE}-BATCH-{int(time.time())}"

LOG_DIR = f"KTUNER_logs/{NAME}"
MAX_TRIALS = 10 
EX_PER_TRIAL = 1

def build_model(hp):
    model = Sequential()

    model.add(LSTM(hp.Int("input_units", 32, 256, 32),
                   input_shape=(data.train_X.shape[1:]), 
                   return_sequences=True))
    model.add(Dropout(hp.Float("input_dropout", 0, 0.9, 1)))
    model.add(BatchNormalization())
   
    num_layers = hp.Int("num_lstm", 1, 3, 1)
    for i in range(0, num_layers):
        lstm_units = hp.Int(f"lstm{i}_units", 32, 256, 32)
        if i < num_layers-1:
            model.add(LSTM(lstm_units,
                           input_shape=(data.train_X.shape[1:]), 
                           return_sequences=True))
        else:
            model.add(LSTM(lstm_units, 
                           input_shape=(data.train_X.shape[1:]), 
                           return_sequences=False))

        model.add(Dropout(hp.Float(f"lstm{i}_dropout", 0, 0.9, 0.1)))
        model.add(BatchNormalization())

    num_dense = hp.Int("num_dense", 0, 3, 1)
    for i in range(0, num_dense):
        model.add(Dense(hp.Int(f"dense{i}_units", 32, 128, 32), activation="relu"))
        model.add(Dropout(hp.Float(f"dense{i}_dropout", 0, 0.9, 0.1)))

    model.add(Dense(2, activation="softmax"))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


if sys.argv[1] == "view":
    tuner = pickle.load(open("ktuner_obj.pkl","rb"))

    tuner.results_summary()
 
elif sys.argv[1] == "search":

    tb = TensorBoard(log_dir=f'logs/{NAME}', update_freq='epoch', profile_batch=0)

    filepath = "RNN_Final-test"
    checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc',
                                                          verbose=1, save_best_only=True,
                                                          mode='max')) # saves only the best one

    tuner = BayesianOptimization(
        build_model,
        objective="val_accuracy",
        max_trials=MAX_TRIALS,
        executions_per_trial=EX_PER_TRIAL,
        directory=LOG_DIR
    )

    tuner.search(x=data.train_X, y=data.train_y, 
                 epochs=EPOCHS, 
                 batch_size=BATCH_SIZE,
                 validation_data=(data.val_X, data.val_y))

    print(tuner.results_summary())

    with open('ktuner_obj.pkl', 'wb') as f:
        pickle.dump(tuner, f)
else:
    print("Usage: python3.8 model.py <view/search>")

#model = build_model()
# history = model.fit(
#  data.train_X, data.train_y,
#    batch_size=BATCH_SIZE,
#    epochs=EPOCHS,
#    validation_data=(data.val_X, data.val_y),
#    callbacks=[tb, checkpoint]
#)
