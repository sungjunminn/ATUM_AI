from keras_preprocessing.sequence import pad_sequences
from konlpy.tag import Hannanum, Okt
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.text import Tokenizer
from keras import preprocessing
from keras.layers import Input, Embedding, Dense, Dropout
from keras.layers import Conv1D, GlobalMaxPool1D, concatenate
from keras.models import Model
train = pd.read_excel("Dataset.xlsx", header=0, index_col=None)
print(train['sentence'])
test = list(train['sentence'])
print(test)
print(str(test))

def okt_preprocessing(sequence, okt):
    new_sequence = okt.morphs(sequence, stem=False)
    return new_sequence


hannanum = Hannanum()
okt = Okt()
print(okt_preprocessing("인사이저는 보다 정확하고 의미있는 비즈니스 인사이트를 제공합니다", okt))

okt_list = []
for i in range(len(test)):
    okt_list.append(okt_preprocessing(test[i], okt))
print(okt_list)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(okt_list)
train_sequence = tokenizer.texts_to_sequences(okt_list)
print(train_sequence)
word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1
print('vocab_size : ', vocab_size)
X_train = pad_sequences(train_sequence, maxlen=10, padding='post')
print(X_train)
# ------------------
MAX_SEQ_LEN = 10
dropout_prob = 0.5
EMB_SIZE = 128
EPOCH = 5
VOCAB_SIZE = len(word_index) + 1
# ------------------


input_layer = Input(shape=(MAX_SEQ_LEN, ))
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layer)
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

conv1 = Conv1D(
    filters=128,
    kernel_size=3,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool1 = GlobalMaxPool1D()(conv1)

conv2 = Conv1D(
    filters = 128,
    kernel_size=4,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(
    filters = 128,
    kernel_size=5,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)

concat = concatenate([pool1, pool2, pool3])

hidden = Dense(128, activation=tf.nn.relu)(concat)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
logits = Dense(3, name='logits')(dropout_hidden)
predictions = Dense(3, activation=tf.nn.softmax)(logits)

model = Model(inputs=input_layer, outputs=predictions)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


'''
while 1:
    test2 = input()
    if test2 == 'break':
        break
    okt_list.append(preprocessing(test2, okt))
    tokenizer.fit_on_texts(okt_list)
    train_sequence = tokenizer.texts_to_sequences(okt_list)
    vocab_size = len(tokenizer.word_index) + 1
    print('vocab_size : ', vocab_size)
    X_train = pad_sequences(train_sequence, maxlen=10, padding='post')
    print(X_train)
    vectorizer = TfidfVectorizer()
    sp_matrix = vectorizer.fit_transform(okt_list)
    similar_vector_values = cosine_similarity(sp_matrix[-1], sp_matrix)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched == 0:
        print("일치하는 데이터 x")
    else:
        print(similar_sentence_number)
        print(test[similar_sentence_number])

'''
