import json
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import imblearn.over_sampling as oversample
import tensorflow as tf
from transformers import TFXLNetModel, XLNetTokenizer


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


xlnet_model = 'xlnet-base-cased'
xlnet_tokenizer = XLNetTokenizer.from_pretrained(xlnet_model)

def xlnet_encode(data,maximum_length) :
    input_ids = []

    for i in range(len(data)):
        encoded = xlnet_tokenizer.encode_plus(

        data[i],
        add_special_tokens=True,
        max_length=maximum_length,
        pad_to_max_length=True,

      )

        input_ids.append(encoded['input_ids'])
    return np.array(input_ids)

train_input_ids = xlnet_encode(x_train,60)
test_input_ids = xlnet_encode(x_test,60)

def create_model_xlnet(xlnet_model):
    word_inputs = tf.keras.Input(shape=(60,), name='word_inputs', dtype='int32')


    xlnet = TFXLNetModel.from_pretrained(xlnet_model)
    xlnet_encodings = xlnet(word_inputs)[0]

    # Collect last step from last hidden state (CLS)
    doc_encoding = tf.squeeze(xlnet_encodings[:, -1:, :], axis=1)

    doc_encoding = tf.keras.layers.Dropout(.1)(doc_encoding)

    outputs = tf.keras.layers.Dense(20, activation='sigmoid', name='outputs')(doc_encoding)

    model = tf.keras.Model(inputs=[word_inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

    

xlnet = create_model_xlnet(xlnet_model)
xlnet.summary()

smt=oversample.SMOTE()
x_train, y_train=smt.fit_resample(train_input_ids,labels)
y_train=np.array(y_train)
x_train= np.array(x_train)
#Early Stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

history = xlnet.fit(x_train,y_train,  validation_split=0.2, epochs=100,batch_size=64,  callbacks=[callback])

labels=[]
for label in y_test:
    labels.append(encode_labels[label])

test_y=np.array(labels)

predicted = xlnet.predict(test_input_ids)
loss, test_acc = xlnet.evaluate(test_input_ids, test_y)
pred_single = np.argmax(predicted, axis=1)
print('Loss: %.3f, Test accuracy: %.3f' % (loss, test_acc))
print(classification_report(test_y, pred_single,  target_names=['2.0','2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7','3.0','3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','3.10','3.11']))
xlnet.save('XLNet20.h5')


def most_probable(vectors):
    return np.argmax(vectors, axis=1)


c = ConfusionMatrixDisplay.from_predictions(y_true=test_y, y_pred=most_probable(predicted), normalize='true',
                                            cmap=plt.cm.Blues,
                                            display_labels=['2.0','2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7','3.0','3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','3.10','3.11'],  values_format='.2f')

fig = c.ax_.get_figure()
fig.set_figwidth(12)
fig.set_figheight(12)
fig.savefig('XLNet.png')

plt.clf()
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.ylabel("Loss value")
plt.xlabel("Number of epochs")
plt.legend()
plt.savefig('XLNet_Loss.png')

