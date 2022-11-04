# -*- coding: utf-8 -*-

import os
os.chdir("C:\\Users\\become\\Desktop\\logdata")

# from google.colab import drive
# drive.mount('/content/drive')

import os
path = "../"
file_list = os.listdir(path)

print("file_list: {}".format(file_list))
print(len(file_list))

f2 = open("total_log.txt",'a')

# 3/31/08~23
line = None
for hour in range(8, 24):
  if hour < 10:
    f = open('atum-staserver_2022-03-31-0'+str(hour)+'.log', encoding='UTF-8')
  else:
    f = open('atum-staserver_2022-03-31-'+str(hour)+'.log', encoding='UTF-8')

  line = None
  while line != '':
    line = f.readline()
    if "[ML-DATA]" in line:
      f2.write(line)

  f.close()

# 4/1~4/3, 0~23
for day in range(1, 4):
  for hour in range(0, 24):
    if hour < 10:
      f = open('atum-staserver_2022-04-0'+str(day)+'-0'+str(hour)+'.log', encoding='UTF-8')
    else:
      f = open('atum-staserver_2022-04-0'+str(day)+'-'+str(hour)+'.log', encoding='UTF-8')

    line = None
    while line != '':
      line = f.readline()
      if "[ML-DATA]" in line:
        f2.write(line)

    f.close()

# 4/4, 0~17
for hour in range(0, 18):
  if hour < 10:
    f = open('atum-staserver_2022-04-04-0'+str(hour)+'.log', encoding='UTF-8')
  else:
    f = open('atum-staserver_2022-04-04-'+str(hour)+'.log', encoding='UTF-8')

  line = None
  while line != '':
    line = f.readline()
    if "[ML-DATA]" in line:
      f2.write(line)

  f.close()

f2.close()

f = open("total_log.txt")
f2 = open("id_item.txt", 'a')

line = None
while line != '':
  line = f.readline()
  t1 = line.split('RULEID=')
  if len(t1) == 1:
    break
  else:
    t2 = t1[1].split('] ')

  t3 = line.split('BIZITEM=')
  t4 = t3[1].split('] ')

  f2.write(t2[0]+','+t4[0]+'\n')
f.close()
f2.close()

f = open("id_item.txt")
list_rule = []
line = None
while line != '':
  line = f.readline()
  t = line.split(',')
  if len(t) == 1:
    break
  else:
    if t[0] in list_rule:
      continue
    else:
      list_rule.append( t[0] )

list_block = []
for idx in list_rule:
  list_block.append([])

f = open("id_item.txt")
line = None
while line != '':
  line = f.readline()
  t = line.split(',')
  if len(t) == 1:
    break
  else:
    list_idx = list_rule.index(t[0])
    list_block[list_idx].append(t[1])
f.close()

print(list_rule)
print(len(list_rule))
print()

total_block = []
for idx in range(len(list_rule)):
  total_block = total_block + list_block[idx]
  print(len(list_block[idx]))

print(len(total_block))
set_block = set(total_block)
print(len(set_block))

# GOAL: make (x, y) set from sequence
import numpy as np

# onehot encoding
def to_onehot(val, max_val):
  t = np.zeros((1, max_val))
  t[0, val] = 1
  return t

# load data
f = open("id_item.txt")

# find rule list and all blocks
list_rule = []
list_block = []
line = None
while line != '':
  line = f.readline()
  t = line.split(',')
  if len(t) == 1:
    break
  else:
    if t[0] not in list_rule:
      list_rule.append( t[0] )
    if t[1].strip() not in list_block:
      list_block.append( t[1].strip() )
f.close()

# dic for block
dic_block_num = {}
for idx in range(len(list_block)):
  dic_block_num[list_block[idx]] = idx

# find sequence for each rule
list_rule_block = []
for i in list_rule:
  list_rule_block.append([])
f = open("id_item.txt")
line = None
while line != '':
  line = f.readline()
  t = line.split(',')
  if len(t) == 1:
    break
  else:
    list_idx = list_rule.index(t[0])
    list_rule_block[list_idx].append(t[1].strip())
f.close()

# make (x, y) set
len_seq = 7
print('length of sequence', len_seq)
dim_block = len(list_block)  # 19
x = np.zeros((0, len_seq, dim_block))
y = np.zeros((0, 1))
for idx_rule in range(len(list_rule)):
  blocks = list_rule_block[idx_rule] # n by 1
  mat = np.zeros( (len(blocks)-1, dim_block) )
  for i in range(mat.shape[0]):
    mat[i, :] = to_onehot(dic_block_num[blocks[i]], dim_block)
  for i in range( len(blocks)-len_seq ):
    tmat = mat[i:(i+len_seq), :]
    x = np.append(x.flatten(), tmat.flatten())
    x = np.reshape(x, (-1, len_seq, dim_block))
    y = np.vstack((y, dic_block_num[blocks[i+len_seq]]))


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=None)

# model set
model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
# model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128, input_shape=(len_seq, dim_block), name='Layer1'))

# Add a Dense layer with 10 units.
model.add(layers.Dense(128, activation='relu', name='Layer2'))
model.add(layers.Dense(128, activation='relu', name='Layer3'))
model.add(layers.Dense(dim_block, activation='softmax', name='Layer4'))

model.summary()

# split train test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train_val, x_val, y_train_val, y_val = train_test_split(x_train, y_train, test_size=0.25)

# fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train_val, y_train_val, epochs=300, batch_size=32, validation_data=(x_val, y_val), verbose=0)

loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('acc: ', mse)

print(x_test.shape)

data_save = []
for i in range(x_test.shape[0]):
  ttt = []
  for j in range(len_seq):
    t1 = np.argmax( x_test[i, j, :] )
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

f = open("test_data"+str(len_seq)+".txt", 'a')
for i in range(x_test.shape[0]):
  text = str()
  for j in range(len_seq):
    if j == (len_seq-1):
      text = text + data_save[i][j]
    else:
      text = text + data_save[i][j] + ','
  text = text + '\n'
  f.write(text)
f.close()

f = open("answer_data"+str(len_seq)+".txt", 'a')
for i in range(x_test.shape[0]):
  text = answer[i] + '\n'
  f.write(text)
f.close()

f = open("correct_data"+str(len_seq)+".txt", 'a')
for i in range(x_test.shape[0]):
  text = correct[i] + '\n'
  f.write(text)
f.close()