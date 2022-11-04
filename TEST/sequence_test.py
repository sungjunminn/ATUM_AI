import pandas as pd
import numpy as np

train = pd.read_csv("C:\\Users\\become\\Desktop\\new_db_data\\ATUM_DEV_ENHANCE_MESSTD\\BR_TxnLotCreate.csv",
                    index_col=None, header=0)
test_list = []
total_list = []
visited = [False] * len(train)

SUPER_ID = train["SUPERITEMID"].tolist()


def check_child(ITEM_ID):
    global SUPER_ID
    print("check_child active")
    child_list = []

    for i in range(len(SUPER_ID)):
        if ITEM_ID == SUPER_ID[i]:
            child_list.append(i)
            print("child name : ", train["ITEMNAME"][i])
    print("check_list : ", child_list)
    return child_list


def dfs_test_2(index):
    global visited, SUPER_ID, test_test_list
    print("dfs_test_2 active", index)
    if index not in total_list:
        total_list.append(index)
        visited[index] = True
    index_list = check_child(train["ITEMID"][index])
    print("index : ", index, train["ITEMNAME"][index])
    print("index_list : ", index_list)
    if 0 < len(index_list) <= 1:

        #index_list.append(SUPER_ID.index(train["ITEMID"][index]))
        visited[SUPER_ID.index(train["ITEMID"][index])] = True
        total_list.append(SUPER_ID.index(train["ITEMID"][index]))
        for i in index_list:
            a = check_child(train['ITEMID'][i])
            if len(a) > 0:
                return dfs_test_2(SUPER_ID.index(train["ITEMID"][i]))
                #return map(dfs_test_2(SUPER_ID.index(train["ITEMID"][i])), a)
    elif len(index_list) > 1:
        for j in index_list:
            b = check_child(train['ITEMID'][j])
            if len(b) > 0:
                return map(dfs_test_2(SUPER_ID.index(train["ITEMID"][j])), b)


while 1:
    for i in range(len(train)):

        if i not in total_list:
            total_list.append(i)
            new_index = check_child(train["ITEMID"][i])
            if len(new_index) > 0:
                for j in range(len(new_index)):
                    dfs_test_2(new_index[j])
        else:
            continue
        print("total_list : ", total_list)
    print("total : ", total_list)

    if len(total_list) >= len(train):
        break

name_list = []
for i in range(len(total_list)):
    name_list.append(train['ITEMNAME'][total_list[i]])

item_name_list = name_list
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
while "0" in item_name_list:
    item_name_list.remove("0")

print(item_name_list)

train2 = pd.read_csv("C:\\Users\\become\\Desktop\\Final_Rule_Sequence.csv", index_col="INDEX", header=0 )
pd_train2 = train2.loc[27]

a = pd_train2.tolist()
print(a)