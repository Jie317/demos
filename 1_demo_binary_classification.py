import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, Activation
from collections import Counter
from sklearn.metrics import roc_auc_score


# ================================================================================= #
# 1 prepare data
# ================================================================================= #1
data = np.random.randint(100, size=(100000, 10))
df = pd.DataFrame(data, columns=['f%d' % v for v in range(len(data[0]))])
df['label'] = 0
df.loc[(df.f0 > 67) & (df.f1 < 32) & (df.f2 % 7 < 3) & (df.f3 % 30 > 12), 'label'] = 1
print(Counter(df.label))
test = df.sample(frac=.3, random_state=0)
train = df.drop(test.index)
train_x = train.values[:, :-1]
train_y = train.values[:, -1:]
test_x = test.values[:, :-1]
test_y = test.values[:, -1:]


# ================================================================================= #
# 2 build model
# ================================================================================= #2

inp = Input(shape=(10, ))

out = Dense(64, activation='relu')(inp)
out = Dense(64, activation='relu')(out)
out = Dense(1, activation='sigmoid')(out)

model = Model(inp, out)

# ================================================================================= #
# 3 compile and train model
# ================================================================================= #3
model.compile(optimizer='adagrad', loss='binary_crossentropy',
              metrics=['binary_crossentropy', 'accuracy'])

tbCallback = TensorBoard(log_dir='tbGraph/', histogram_freq=1, write_graph=True)

model.fit(train_x, train_y,
      validation_data=(test_x, test_y),
      epochs=10,
      batch_size=256,
      callbacks=[tbCallback],
      shuffle=True,
      )


# ================================================================================= #
# 4 evaluate model
# ================================================================================= #4
print('Evaluating model')

scores = model.evaluate(test_x, test_y)

for n,v in zip([model.loss] + model.metrics, scores):
        print(n, v)


# ================================================================================= #
# 5 evaluate model on another metric
# ================================================================================= #5
print('Evaluating model with another metric')

y_proba = model.predict(test_x)

aucRoc = roc_auc_score(test_y, y_proba)

print('AUC ROC:', aucRoc)