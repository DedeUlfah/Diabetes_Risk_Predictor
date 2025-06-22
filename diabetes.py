from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from matplotlib import pyplot
from numpy import argmax
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization


file_path = r'C:\Users\user\Documents\tugasakhir\mini-projek\diabetes_dataset.csv'
df = read_csv(file_path)
df.head()

x = df.values[:, :-1]  
y = df.values[:, -1]  
x[0:4]
x = x.astype('float32')
x[0:4]
y

y = LabelEncoder().fit_transform(y)
y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print('Ukuran x train:', x_train.shape)
print('Ukuran y train:', y_train.shape)
print()
print('Ukuran x test:', x_test.shape)
print('Ukuran y test:', y_test.shape)

model = Sequential([
    Input(shape=(4,)),
    Dense(3, activation='relu'),
    Dense(3, activation='relu'),
    Dense(3, activation='softmax'),
], name='Sequential_API_1')

model = Sequential(name='sequential_API_2')
model.add(Input(shape=(4,)))
model.add(Dense(3, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(3, activation='softmax'))

input_layer = Input(shape=(4,))
hid_layer_1 = Dense(3, activation='relu')(input_layer)
hid_layer_2 = Dense(3, activation='relu')(hid_layer_1)
output_layer = Dense(3, activation='softmax')(hid_layer_2)
model = Model(inputs=input_layer, outputs=output_layer, name='Functional_API')

model.summary()

plot_model(model, 'model.png', show_shapes=True)
model.compile(
    optimizer = Adam(learning_rate=0.001),
    loss = SparseCategoricalCrossentropy(),
    metrics = ['Accuracy']
)
hist = model.fit(
    x = x_train,
    y = y_train,
    validation_data = (x_test, y_test),
    batch_size = 32,
    epochs = 200,
    verbose=2
)
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Accuracy')
pyplot.plot(hist.history['Accuracy'], label='train')
pyplot.plot(hist.history['val_Accuracy'], label='val')
pyplot.legend()
pyplot.show()

pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Loss')
pyplot.plot(hist.history['loss'], label='train loss')
pyplot.plot(hist.history['val_loss'], label='val loss')
pyplot.legend()
pyplot.show()

loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test Accuracy: {acc}')

# Input fitur dari pengguna
new_sepal_length = float(input('Input sepal length: '))
new_sepal_width = float(input('Input sepal width: '))
new_petal_length = float(input('Input petal length: '))
new_petal_width = float(input('Input petal width: '))

# Buat array NumPy dari input
new_data = [new_sepal_length, new_sepal_width, new_petal_length, new_petal_width]
new_data_np = np.array([new_data])  # Bentuk (1, 4)

# Prediksi
y_pred = model.predict(new_data_np)
y_class = argmax(y_pred)

# Tampilkan hasil
print(f'\nHasil Prediksi: {y_pred} (class {y_class}) \n')

if y_class == 0:
    print('Iris Setosa')
elif y_class == 1:
    print('Iris Versicolor')
elif y_class == 2:
    print('Iris Virginica')

model.save('model_diabetes.h5')

model = load_model('model_iris.h5')
new_data = [4.9, 3.0, 1.4, 0.2]
y_pred = model.predict(new_data_np)
print('\nPredicted:%s (class=%d)' % (y_pred, argmax(y_pred)))

model = Sequential(name='Dropout_Example')
model.add(Dense(100, input_shape=(10,)))
model.add(Dense(80))
model.add(Dropout(0.5))
model.add(Dense(30))
model.add(Dropout(0.4))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

model = Sequential(name='Batch_Normalization_Example')
model.add(Dense(100, input_shape=(10,)))
model.add(BatchNormalization())
model.add(Dense(80))
model.add(BatchNormalization())
model.add(Dense(30))
model.add(BatchNormalization())
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

