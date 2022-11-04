import pandas as pd
import numpy as np
import os
import os

train = pd.read_csv("C:\\Users\\become\\Desktop\\new_db_data\\ATUM_PRD_MESSTD_MESSTD_V2\\BR_ExeTxnWOModify.csv",
                    index_col=None, header=0)
'''
train = pd.read_csv("C:\\Users\\become\\Desktop\\test_csv.csv",
                    index_col=None, header=0)
'''
test_list = []


def test_algorithm():
    to_list = train['SUPERITEMID'].to_list()
    global test_list
    for i in range(len(train)):
        # print(train['ITEMID'][i])
        a = train['ITEMID'][i]
        if a in to_list:
            test_list.append(i)
            print(a)
            print(train[train['ITEMID'] == a])
            print("\n")
            # print(train['SUPERITEMID'][i])


visited = [False] * len(train)
to_list = train['SUPERITEMID'].to_list()


def dfs(train, v, visited):
    global test_list, to_list
    visited[v] = True

    for i in range(v, len(train)):
        a = train['ITEMID'][i]

        check_visited(i)
        print(visited)
        print(test_list)
        if not visited[i]:
            if a in to_list:
                check_count = check_child(a)
                if check_count == 1:
                    b = train.index[train['SUPERITEMID'] == a].tolist()
                    check_visited(b[0])
                    dfs(train, b[0], visited)
                    continue
                else:
                    b = train.index[train['SUPERITEMID'] == a].tolist()
                    check_visited(b[-1])
                    for j in range(len(b), -1):
                        dfs(train, b[j], visited)

            else:
                visited[i] = True
                pass


def check_child(a):
    b = train.index[train['SUPERITEMID'] == a].tolist()
    return len(b)


def check_visited(i):
    global visited
    if i not in test_list:
        test_list.append(i)
        # visited[i] = True
    return test_list


# test_algorithm()
# dfs(train, 0, visited)
# print(visited)
# print(test_list)
total_list = []

SUPER_ID = train["SUPERITEMID"].tolist()


def dfs_test(ITEM_ID):
    global SUPER_ID

    index = []
    for k in range(len(SUPER_ID)):

        if ITEM_ID == SUPER_ID[k]:
            index.append(k)

    index = sorted(index, reverse=False)

    return index


def dfs_test_2(index):
    global visited, SUPER_ID

    if index not in total_list:
        total_list.append(index)

    if train["ITEMID"][index] in SUPER_ID:
        index_list = dfs_test(train["ITEMID"][index])
        print("index_list", index_list)
        # index_list.append(SUPER_ID.index(train["ITEMID"][index]))
        visited[SUPER_ID.index(train["ITEMID"][index])] = True
        return dfs_test_2(SUPER_ID.index(train["ITEMID"][index]))


while 1:
    for i in range(len(train)):
        print(total_list)
        if i not in total_list:
            total_list.append(i)
            new_index = dfs_test(train["ITEMID"][i])

            for j in range(len(new_index)):
                dfs_test_2(new_index[j])
        else:
            continue
    print("total : ", total_list)
    if len(total_list) >= len(train):
        break
print(total_list)
'''
import os
import pandas as pd
item_list = []

os.chdir("C:\\Users\\become\\Desktop\\DB 데이터")
test_list = os.listdir()

new_biz_item_list = pd.read_csv("C:\\Users\\become\\Desktop\\mldata\\old_recent_bizitem.csv", index_col=None, header=0)

old_biz = new_biz_item_list["old_item"].tolist()
new_biz = new_biz_item_list["recent_item"]
new_biz = new_biz.dropna()
new_biz = new_biz.tolist()
a = set(old_biz).intersection(set(new_biz))
print(a)
'''
'''
for k in test_list:
    os.chdir("C:\\Users\\become\\Desktop\\DB 데이터\\"+k)

    tlist = os.listdir()

    file_list = []
    #print(tlist)
    for x in tlist:
        if ".csv" in x:
            file_list.append(x)

    for i in file_list:
        df = pd.read_csv(i)
        test_a = df["ITEMNAME"].values.tolist()
        item_list.append(test_a)
'''