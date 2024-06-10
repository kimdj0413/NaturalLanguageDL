import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

data = pd.read_csv('Iris.csv', encoding='latin1')
# print(data[:5])
# print(data["Species"].unique(), sep="\n")

sns.set_theme(style="ticks", color_codes=True)
# g = sns.pairplot(data, hue="Species", palette="husl")
# sns.barplot(x='Species', y='SepalWidthCm', data=data, ci=None)
# data['Species'].value_counts().plot(kind='bar')
# plt.show()
data['Species'] = data['Species'].replace(['Iris-virginica','Iris-setosa','Iris-versicolor'],[0,1,2])
# data['Species'].value_counts().plot(kind='bar')
# plt.show()
data_X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
data_y = data['Species'].values
# print(data_X[:5])
# print(data_y[:5])
(X_train, X_test, y_train, y_test) = train_test_split(data_X, data_y, train_size=0.8, random_state=1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train[:5])
# print(y_test[:5])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_test, y_test))

epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()