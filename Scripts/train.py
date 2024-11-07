from sklearn.model_selection import train_test_split # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import numpy as np # type: ignore
import os
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.callbacks import TensorBoard # type: ignore


actions = np.array(['assault', 'stabbing', 'gun violence'])

label_map = {label: num for num, label in enumerate(actions)} 

#print(label_map)

DATA_SET = os.path.join('DATA')

no_sequences = 30

sequence_length = 30

sequences, labels = [], []
for action in actions:
    for sequence in range (no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_SET, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

#print(np.array(sequences).shape)


# creating values to train the model

x = np.array(sequences)
y = to_categorical(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir = log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(actions.shape[0], activation = 'softmax'))

model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(x_train, y_train, epochs = 2000, callbacks = [tb_callback])


# Save the trained model

model.save('model.h5')