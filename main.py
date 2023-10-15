import keras
import numpy as np
from transformers import TFBertModel, BertTokenizer

PYTHON_VERSIONS = {shortned: actual for shortned, actual in {
    0: "2.0",
    1: "2.1",
    2: "2.2",
    3: "2.3",
    4: "2.4",
    5: "2.5",
    6: "2.6",
    7: "2.7",
    8: "3.0",
    9: "3.1",
    10: "3.2",
    11: "3.3",
    12: "3.4",
    13: "3.5",
    14: "3.6",
    15: "3.7",
    16: "3.8",
    17: "3.9",
    18: "3.10",
    19: "3.11"
}.items()}


def main(filename):
    model = keras.models.load_model('models/BERT_LSTM20.h5', custom_objects={'TFBertModel': TFBertModel})
    print("Model loaded")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def bert_encode(data_d, maximum_length):
        encoded = tokenizer.encode_plus(

            data_d,
            add_special_tokens=True,
            max_length=maximum_length,
            pad_to_max_length=True,

            return_attention_mask=True,

        )
        return [encoded['input_ids']], [encoded['attention_mask']]

    file = open(filename, "r").read()
    encoded_text, mask = bert_encode(file, 60)
    res = model.predict([np.array(encoded_text), np.array(mask)])
    print(PYTHON_VERSIONS[np.argmax(res[0])])


if __name__ == '__main__':
    main("TCN.py")
