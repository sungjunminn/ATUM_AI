
import pandas as pd
import numpy as np
from anytree import Node, RenderTree
'''
train = pd.read_csv("C:\\Users\\become\\Desktop\\new_db_data\\ATUM_DEV_ENHANCE_MESSTD\\TxnLotCreate.csv",
                    index_col=None, header=0)

SUPER_ID = train["SUPERITEMID"].tolist()

global_list = []
check_list = []
# test_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 47, 48, 49, 50, 51, 52, 53, 60, 61, 62, 63, 64, 65, 66, 67,
#                   90, 91, 92, 93, 94, 95, 96, 103, 104, 105, 106, 116, 117, 118, 119, 120, 121, 122]
test_index_list = [6]

'''
'''
def total_tree(list):
    global global_list
    global check_list

    def check_child(ITEM_ID):
        global SUPER_ID
        child_list = []

        for i in range(len(SUPER_ID)):
            if ITEM_ID == SUPER_ID[i]:
                child_list.append(i)

        return child_list


    for i in range(len(train)):
        print(i, check_child(train['ITEMID'][i]))


    def treetest(input):

        if len(check_list) > 0:
            check_list.remove(check_list[0])
        if input not in global_list:
            global_list.append(input)
        a = check_child(train['ITEMID'][input])
        # print("test: ", input, check_list, a)
        if len(a) >= 0:
            # global_list.append(a[0])
            for j in range(len(a)):
                # if input not in global_list:
                global_list.insert(global_list.index(input) + j + 1, a[j])
                check_list.append(a[j])
                # print("global", global_list)
                # print("check", check_list)
            if len(check_list) == 0 and len(a) == 0:
                return
            else:
                return treetest(check_list[0])
        else:
            return

    # print(check_list)

    for i in list:
        treetest(i)
    print(global_list)
    print(len(global_list))
    return_list = []
    for i in range(len(global_list)):
        print(i, train['ITEMNAME'][global_list[i]], global_list[i])
        return_list.append(train['ITEMNAME'][global_list[i]])
    return return_list
'''
'''
a = total_tree(test_index_list)
print("결과", a)
'''