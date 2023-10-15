import textwrap
import keras
from transformers import TFBertModel
import json
from transformers import BertTokenizer
import numpy as np
import asttokens
import ast

model = keras.models.load_model('models/BERT_LSTM20.h5', custom_objects={'TFBertModel': TFBertModel})

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


with open('longdata.json') as json_file:
    data = json.load(json_file)

PYTHON_VERSIONS = {shortned: actual for shortned, actual in {
    "2.0": [],
    "2.1": [],
    "2.2": [],
    "2.3": [],
    "2.4": [],
    "2.5": [],
    "2.6": [],
    "2.7": [],
    "3.0": [],
    "3.1": [],
    "3.2": [],
    "3.3": [],
    "3.4": [],
    "3.5": [],
    "3.6": [],
    "3.7": [],
    "3.8": [],
    "3.9": [],
    "3.10": [],
    "3.11": []
}.items()}

for file in data.keys():
    if data[file]["label"] in PYTHON_VERSIONS.keys():
        PYTHON_VERSIONS[data[file]["label"]].append(data[file]["data"])

correctVersion = 0
version_to_accuracy = {}
for version in PYTHON_VERSIONS.keys():
    print(version)
    countCorrect = 0
    countFiles = 0
    version_to_accuracy[version] = 0
    for file in PYTHON_VERSIONS[version][:100]:
        countFiles += 1
        versions = []
        for line in file.split("\n"):
            encoded_text, mask = bert_encode(line, 60)
            for vector in model.predict([np.array(encoded_text), np.array(mask)]):
                versions.append(np.argmax(vector))
                break
        print(versions)
        if max(versions) == correctVersion:
            countCorrect += 1
    if countFiles > 0:
        version_to_accuracy[version] = countCorrect / countFiles
        print(version_to_accuracy)
    correctVersion += 1
print("Tested per line: ")
print(version_to_accuracy)

correctVersion = 0
version_to_accuracy = {}
for version in PYTHON_VERSIONS.keys():
    countCorrect = 0
    countFiles = 0
    version_to_accuracy[version] = 0
    for file in PYTHON_VERSIONS[version][:100]:
        countFiles += 1
        versions = []
        try:
            atok = asttokens.ASTTokens(file, parse=True)
            for n in ast.walk(atok.tree):
                if not isinstance(n, ast.Module) and not isinstance(n,
                                                                    ast.Name) and not isinstance(
                    n,
                    ast.Load) and not isinstance(
                    n, ast.Store) \
                        and not isinstance(n, ast.Constant):
                    encoded_text, mask = bert_encode(atok.get_text(n), 60)
                    for vector in model.predict([np.array(encoded_text), np.array(mask)]):
                        versions.append(np.argmax(vector))
                        break
            if max(versions) == correctVersion:
                countCorrect += 1
        except:
            continue
    if countFiles > 0:
        version_to_accuracy[version] = countCorrect / countFiles
    correctVersion += 1
print("Tested AST nodes: ")
print(version_to_accuracy)

correctVersion = 0
version_to_accuracy = {}
for version in PYTHON_VERSIONS.keys():
    print(version)
    countCorrect = 0
    countFiles = 0
    version_to_accuracy[version] = 0
    for file in PYTHON_VERSIONS[version][:100]:
        countFiles += 1
        versions = []
        for part in textwrap.wrap(file, 60):
            encoded_text, mask = bert_encode(part, 60)
            for vector in model.predict([np.array(encoded_text), np.array(mask)]):
                versions.append(np.argmax(vector))
                break
        print(versions)
        if max(versions) == correctVersion:
            countCorrect += 1
    if countFiles > 0:
        version_to_accuracy[version] = countCorrect / countFiles
        print(version_to_accuracy)
    correctVersion += 1
print("Tested per part: ")
print(version_to_accuracy)
