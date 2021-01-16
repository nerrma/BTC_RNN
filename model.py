import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

assert tf.test.is_gpu_available() == True

import data
import time
EPOCHS = 15
BATCH_SIZE = 64
NAME = f"HOURLY_BTC_{data.SEQ_LEN}-SEQ-{data.FUTURE_PREDICT_PERIOD}-BATCH-{BATCH_SIZE}-{int(time.time())}"

model = Sequential()

model.add(LSTM(256, input_shape=(data.train_X.shape[1:]), return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(LSTM(256, input_shape=(data.train_X.shape[1:]), return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(LSTM(128, input_shape=(data.train_X.shape[1:]), return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
tb = TensorBoard(log_dir=f'logs/{NAME}', update_freq='epoch', profile_batch=0)

filepath = "RNN_Final-test"
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc',
                                                      verbose=1, save_best_only=True,
                                                      mode='max')) # saves only the best one

history = model.fit(
    data.train_X, data.train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(data.val_X, data.val_y),
    callbacks=[tb, checkpoint]
)
