import numpy as np
from keras.models import model_from_json
from tensorflow.python.keras.models import load_model
import time

start = time.time()
model = load_model('Module\\lstm_model.h5')
'''
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('model.h5')
'''


def LSTM_main(len_3_bizitem):
    # loaded_model.complie(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    # a = ['SQLEXECUTER', 'MULTIASSIGNUSERDEFINE', 'FORLOOP']
    a = len_3_bizitem
    print(a)

    def to_onehot(val, max_val):
        t = np.zeros((1, max_val))
        t[0, val] = 1
        return t

    dic_block_num = {'REPLYMSG': 0, 'FOR-MAINBLOCK': 1, 'INSERTBYKEY': 2,
                     'SETREPLYMSG': 3, 'IFBLOCK': 4, 'ELSEBLOCK': 5,
                     'ELSEIFBLOCK': 6, 'LISTADD': 7, 'LOGTRACE': 8,
                     'MAPPUT': 9, 'EVENTVALIDATION': 10, 'SETSUCCESSCOUNT': 11,
                     'LOOKUPBYKEY': 12, 'WHILELOOP': 13, 'LOOKUPBYCONDITION': 14,
                     'SQLEXECUTER': 15, 'WHILE-MAINBLOCK': 16, 'IF-MAINBLOCK': 17,
                     'DBOBJECTSETS': 18, 'DELETEBYKEY': 19, 'CONDITIONALUDF': 20,
                     'VALIDATIONCHECK': 21, 'SYSTEMLIBRARY': 22, 'EVENTDATAASSIGN': 23,
                     'MAPGET': 24, 'DBOBJECTCOPY': 25, 'UDF': 26,
                     'UPDATEBYKEY': 27, 'BATCHINSERT': 28, 'FORLOOP': 29,
                     'MULTIASSIGNUSERDEFINE': 30}
    blocks = a  # n by 1
    x = np.zeros((0, 3, 31))
    mat = np.zeros((3, 31))

    for i in range(mat.shape[0]):
        mat[i, :] = to_onehot(dic_block_num[blocks[i]], 31)
    print("mat : ", mat)
    tmat = mat[0:3, :]
    x = np.append(x.flatten(), tmat.flatten())
    x = np.reshape(x, (-1, 3, 31))
    # y = np.vstack((y, dic_block_num[blocks[i + len_seq]]))
    print(x)
    print(model.predict(x))

    print(np.argmax(model.predict(x), axis=1))
    b = np.argmax(model.predict(x), axis=1)
    return_bizitme = ''
    for key, value in dic_block_num.items():
        if value == b[0]:
            print(key)
            return_bizitme = key
    end = time.time()

    return return_bizitme
