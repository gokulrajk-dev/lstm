import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

X_train = np.array([[[i], [i+1], [i+2]] for i in range(1, 51)]) 
y_train = np.array([[i+3, i+4] for i in range(1, 51)])

model = Sequential([
    LSTM(50, input_shape=(3, 1)),
    Dense(2)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=250, verbose=0)

X_test = np.array([[[21], [22], [23]]])
prediction = model.predict(X_test)
print("Predicted next two values:", prediction[0])
