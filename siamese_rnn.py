from keras.layers import Input, Convolution2D, Dense, Flatten, LSTM
from keras.model import Model, Sequential
import pickle
import theano
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

train_x =
train_y =
n_labels = len(set(train_y))
siamese_theano = pickle.load(open('3434987_shreq.pkl', 'r'))
n_samples, h, w = train_x.shape
x = Input(shape=(h, w))
h1 = Convolution2D(20, 3, 3, activation='sigmoid', border_mode='same')(x)
h2 = Convolution2D(50, 3, 3, activation='sigmoid', border_mode='same')(h1)
h2_f = Flatten()(h2)
h3 = Dense(500)(h2_f)
h4 = Dense(100)(h3)
h5 = Dense(2)(h4)


siamese = Model(input=x, output=h5)
siamese.compile(optimizer='adam',
                loss='hinge')

siamese.set_weight(siamese_theano)

get_feature100 = theano.function(siamese.layers[0].input,
                                  siamese.layers[-2].output)

feat_x = get_feature100(train_x)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(feat_x)

feat_neighbor = np.empty((n_samples, 6, feat_x.shape[1]))
for sample in xrange(n_samples):
    neighbor_x = knn.kneighbors(feat_x[sample])
    feat_neighbor[sample, :5, :] = neighbor_x
    feat_neighbor[sample, 5, :] = feat_x[sample]

rnn = Sequential()
rnn.add(LSTM(100, input_shape=(5, 100), activation='tanh'))
rnn.add(Dense(n_labels, activation='softmax'))
rnn.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

rnn.fit(feat_neighbor, train_y,
        batch_size=5,
        nb_epoch=10,
        verbose=1)

