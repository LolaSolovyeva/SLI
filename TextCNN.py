import json
import numpy as np
from gensim.models import Word2Vec
import itertools
import numpy as np
from keras.layers import *
from keras.models import *
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import imblearn.over_sampling as oversample
import itertools
import tcn

with open('data.json') as json_file:
    data = json.load(json_file)
    
PYTHON_VERSIONS = {shortned: actual for shortned, actual in {
    "2.0": 0,
    "2.1": 0,
    "2.2": 0,
    "2.3": 0,
    "2.4": 0,
    "2.5": 0,
    "2.6": 0,
    "2.7": 0,
    "3.0": 0,
    "3.1": 0,
    "3.2": 0,
    "3.3": 0,
    "3.4": 0,
    "3.5": 0,
    "3.6": 0,
    "3.7": 0,
    "3.8": 0,
    "3.9": 0,
    "3.10": 0,
    "3.11": 0
}.items()}

x=[]
y=[]
for version in data.keys():
    if version in PYTHON_VERSIONS.keys():
        x= np.concatenate((x, data[version]),axis=None)
        labels=[version]*len(data[version])
        y= np.concatenate((y, labels),axis=None)
        
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.4)


encode_labels = {k: v for k, v in zip(['2.0','2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7','3.0','3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','3.10','3.11'], range(20))}
labels=[]
for label in y_train:
    labels.append(encode_labels[label])



def create_w2v(sentences):
    w2v_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
    w2v_model.save("word2vecTextCNN.model")
sentences = []
for text in x_train:
    res=itertools.chain.from_iterable([piece.split("\n") for piece in text.split(" ")])
    sentences.append(list(filter(None, res)))
create_w2v(sentences)



w2v = Word2Vec.load("word2vecTextCNN.model")
print("Word2Vec loaded")

def word2token(word):
    try:
        return w2v.wv.key_to_index[word]
    except KeyError:
        return 0

MAX_SEQUENCE_LENGTH = 100
sentences = []
for text in x_train:
    res=itertools.chain.from_iterable([piece.split("\n") for piece in text.split(" ")])
    sequence = list(filter(None, res))[:MAX_SEQUENCE_LENGTH]
    sequence=[word2token(w) for w in sequence]
    if len(sequence) < MAX_SEQUENCE_LENGTH:
        sequence.extend([0]*(MAX_SEQUENCE_LENGTH-len(sequence)))
    sentences.append(sequence)
    
    
w2v_weights=w2v.wv.vectors
vocab_size, embedding_size = w2v_weights.shape
print("Vocabulary Size: {} - Embedding Dim: {}".format(vocab_size, embedding_size))

train_x = np.array(sentences)
train_y=np.array(labels)

smt=oversample.SMOTE()
x_train, y_train=smt.fit_resample(train_x,train_y)

#Early Stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model = Sequential()
model.add(Embedding(input_dim=vocab_size,
                    output_dim=embedding_size,
                    weights=[w2v_weights],
                    input_length=MAX_SEQUENCE_LENGTH,
                    mask_zero=True,
                    trainable=False))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPool1D())
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_x, train_y,  validation_split=0.2, epochs=100, batch_size=64,callbacks=[callback], verbose=1)

MAX_SEQUENCE_LENGTH = 100
sentences_test = []
for text in x_test:
    res=itertools.chain.from_iterable([piece.split("\n") for piece in text.split(" ")])
    sequence = list(filter(None, res))[:MAX_SEQUENCE_LENGTH]
    sequence=[word2token(w) for w in sequence]
    if len(sequence) < MAX_SEQUENCE_LENGTH:
        sequence.extend([0]*(MAX_SEQUENCE_LENGTH-len(sequence)))
    sentences_test.append(sequence)

labels=[]
for label in y_test:
    labels.append(encode_labels[label])

test_x = np.array(sentences_test)
test_y=np.array(labels)

predicted = model.predict(test_x)
results = model.evaluate(test_x, test_y)
pred_single = np.argmax(predicted, axis=1)
loss, test_acc = model.evaluate(test_x, test_y)
print('Loss: %.3f, Test accuracy: %.3f' % (loss, test_acc))
print(classification_report(test_y, pred_single,  target_names=['2.0','2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7','3.0','3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','3.10','3.11']))
#model.save('TextCNNW2V20.h5')


def most_probable(vectors):
    return np.argmax(vectors, axis=1)


c = ConfusionMatrixDisplay.from_predictions(y_true=test_y, y_pred=most_probable(predicted), normalize='true',
                                            cmap=plt.cm.Blues,
                                            display_labels=['2.0','2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7','3.0','3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','3.10','3.11'],  values_format='.2f')


fig = c.ax_.get_figure()
fig.set_figwidth(12)
fig.set_figheight(12)
#fig.savefig('TextCNN.png')


plt.clf()
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.ylabel("Loss value")
plt.xlabel("Number of epochs")
plt.legend()
plt.savefig('TextCNN_Loss.png')