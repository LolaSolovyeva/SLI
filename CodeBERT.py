import json
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from transformers import TFRobertaModel
import imblearn.over_sampling as oversample
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from transformers import AutoTokenizer


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


tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

def bert_encode(data,maximum_length) :
    input_ids = []
    attention_masks = []


    for i in range(len(data)):
        encoded = tokenizer.encode_plus(

        data[i],
        add_special_tokens=True,
        max_length=maximum_length,
        pad_to_max_length=True
      )

        input_ids.append(encoded['input_ids'])
    return np.array(input_ids)

train_input_ids= bert_encode(x_train,60)
test_input_ids  = bert_encode(x_test,60)


def create_model(bert_model):
    input_ids = tf.keras.Input(shape=(60,),dtype='int32',name="input_ids")

    output = bert_model(input_ids)
    output = output[1]

    output = tf.keras.layers.Dense(32,activation='relu')(output)
    output = tf.keras.layers.Dropout(0.2)(output)

    output = tf.keras.layers.Dense(20,activation='sigmoid')(output)
    model = tf.keras.models.Model(inputs = input_ids,outputs = output)
    model.compile(Adam(lr=6e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
    

codebert_model = TFRobertaModel.from_pretrained('microsoft/codebert-base')
model = create_model(codebert_model)
model.summary()

smt=oversample.SMOTE()
x_train, y_train=smt.fit_resample(train_input_ids,labels)
y_train=np.array(y_train)
x_train= np.array(x_train)

#Early Stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)


history = model.fit(x_train,y_train, validation_split=0.2, epochs=100,batch_size=64, callbacks=[callback])

labels=[]
for label in y_test:
    labels.append(encode_labels[label])

test_y=np.array(labels)

predicted = model.predict(test_input_ids)
loss, test_acc = model.evaluate(test_input_ids, test_y)
pred_single = np.argmax(predicted, axis=1)
print('Loss: %.3f, Test accuracy: %.3f' % (loss, test_acc))
print(classification_report(test_y, pred_single,  target_names=['2.0','2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7','3.0','3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','3.10','3.11']))
model.save('CodeBERT20.h5')


def most_probable(vectors):
    return np.argmax(vectors, axis=1)


c = ConfusionMatrixDisplay.from_predictions(y_true=test_y, y_pred=most_probable(predicted), normalize='true',
                                            cmap=plt.cm.Blues,
                                            display_labels=['2.0','2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7','3.0','3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','3.10','3.11'],  values_format='.2f')

fig = c.ax_.get_figure()
fig.set_figwidth(12)
fig.set_figheight(12)
fig.savefig('CodeBERT.png')

plt.clf()
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.ylabel("Loss value")
plt.xlabel("Number of epochs")
plt.legend()
plt.savefig('CodeBERT_Loss.png')
