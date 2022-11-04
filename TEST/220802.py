import pandas as pd
import numpy as np
import time


def fix(item_name_list):
    for z in range(len(item_name_list)):
        if item_name_list[z] == "EFORLOOP":
            item_name_list[z] = "FORLOOP"
        if item_name_list[z] == "EFORLOOP-BODY":
            item_name_list[z] = "FORLOOP-BODY"
        if item_name_list[z] == "EFOR-MAINBLOCK":
            item_name_list[z] = "FOR-MAINBLOCK"
        if item_name_list[z] == "SETSUCESSCOUNT":
            item_name_list[z] = "SETSUCCESSCOUNT"
        if item_name_list[z] == "CONDITIONALLOGERROR":
            item_name_list[z] = "VALIDATIONCHECK"
        if item_name_list[z] == "VALIDATIONCHECK-BODY":
            item_name_list[z] = "0"
        if item_name_list[z] == "MAPPUT-BODY":
            item_name_list[z] = "0"
        if item_name_list[z] == "if":
            item_name_list[z] = "0"
        if item_name_list[z] == "ConditionalUDF-BODY":
            item_name_list[z] = "0"
        if item_name_list[z] == "DBOBJECTSETS-BODY":
            item_name_list[z] = "0"
        if item_name_list[z] == "FORLOOP-BODY":
            item_name_list[z] = "0"
    while "0" in item_name_list:
        item_name_list.remove("0")
    return item_name_list


# onehot encoding
def to_onehot(val, max_val):
    t = np.zeros((1, max_val))
    t[0, val] = 1
    return t


train = pd.read_csv("../Module/test_name4.csv", index_col=0, header=0)
print(train)
list_rule_block = []
for i in range(len(train)):
    pd_train = train.loc[i].tolist()
    new_list = [x for x in pd_train if pd.isnull(x) == False]

    new_list_2 = fix(new_list)
    # print(new_list)
    list_rule_block.append(new_list_2)
print("list_rule_book : ", list_rule_block)

list_rule = []
# len(item_list)

for i in list_rule_block:
    for j in i:
        list_rule.append(j)

# all cases of biz item
unique_list = set(list_rule)

list_block = list(unique_list)

print(unique_list, len(unique_list))

print("list_rule : ", list_rule)
dic_block_num = {}
for idx in range(len(list_block)):
    dic_block_num[list_block[idx]] = idx
print("dic_block_num : ", dic_block_num)

# make (x, y) set
len_seq = 3
print('length of sequence', len_seq)
dim_block = len(list_block)  # 31
x = np.zeros((0, len_seq, dim_block))
y = np.zeros((0, 1))
for idx_rule in range(180):
    blocks = list_rule_block[idx_rule]  # n by 1
    mat = np.zeros((len(blocks) - 1, dim_block))
    for i in range(mat.shape[0]):
        mat[i, :] = to_onehot(dic_block_num[blocks[i]], dim_block)
    for i in range(len(blocks) - len_seq):
        tmat = mat[i:(i + len_seq), :]
        x = np.append(x.flatten(), tmat.flatten())
        x = np.reshape(x, (-1, len_seq, dim_block))
        y = np.vstack((y, dic_block_num[blocks[i + len_seq]]))
start = time.time()

from tensorflow import keras
from tensorflow.python.keras import layers
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
'''
y = to_categorical(y, num_classes=None)

# model set
model = Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
# model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
# model.build(input_shape=(len_seq, dim_block))
# model.add(layers.LSTM(128, input_shape=(len_seq, dim_block), name='Layer1'))
model.add(LSTM(128, input_shape=(len_seq, dim_block), name='Layer1'))

# Add a Dense layer with 10 units.
# model.add(layers.Dense(128, activation='relu', name='Layer2'))
model.add(Dense(128, activation='relu', name='Layer2'))
# model.add(layers.Dense(128, activation='relu', name='Layer3'))
model.add(Dense(128, activation='relu', name='Layer3'))
# model.add(layers.Dense(dim_block, activation='softmax', name='Layer4'))
model.add(Dense(dim_block, activation='softmax', name='Layer4'))
'''
'''
class CustomLayer(tf.python.keras.layers.Layer):
    def __init__(self):
        super(CustomLayer, self).__init__()

    def get_config(self):
        config = super().get_config()

        return config
'''

# model.summary()
'''
# split train test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train_val, x_val, y_train_val, y_val = train_test_split(x_train, y_train, test_size=0.25)

# fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train_val, y_train_val, epochs=300, batch_size=32, validation_data=(x_val, y_val), verbose=0)
model.summary()
'''
'''
model_json = model.to_json()
with open("model.json", "w") as json_file :
    json_file.write(model_json)

model.save_weights('model.h5')
print("save model")
'''

'''
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('acc: ', mse)

model.save('lstm_model.h5')
print(x_test)
print(x_test.shape)
data_save = []
for i in range(x_test.shape[0]):
    ttt = []
    for j in range(len_seq):
        t1 = np.argmax(x_test[i, j, :])
        ttt.append(list_block[t1])
    data_save.append(ttt)

t1 = model.predict(x_test)
t1 = np.argmax(t1, axis=1)
answer = []
for i in range(x_test.shape[0]):
    answer.append(list_block[t1[i]])

t2 = np.argmax(y_test, axis=1)
correct = []
for i in range(x_test.shape[0]):
    correct.append(list_block[t2[i]])
'''
'''
f = open("C:\\Users\\become\\Desktop\\become log lstm\\test_data" + str(len_seq) + ".txt", 'a', encoding='utf-8')
for i in range(x_test.shape[0]):
    text = str()
    for j in range(len_seq):
        if j == (len_seq - 1):
            text = text + data_save[i][j]
        else:
            text = text + data_save[i][j] + ','
    text = text + '\n'
    f.write(text)
f.close()

f = open("C:\\Users\\become\\Desktop\\become log lstm\\answer_data" + str(len_seq) + ".txt", 'a', encoding='utf-8')
for i in range(x_test.shape[0]):
    text = answer[i] + '\n'
    f.write(text)
f.close()
'''
'''
f = open("C:\\Users\\become\\Desktop\\become log lstm\\correct_data" + str(len_seq) + ".txt", 'a', encoding='utf-8')
for i in range(x_test.shape[0]):
    text = correct[i] + '\n'
    f.write(text)
f.close()
'''
'''
end = time.time()
print("time : ", end - start)
'''