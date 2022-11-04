import pandas as pd


def total_tree(list, dataset, superID):
    global_list = []
    check_list = []

    def check_child(ITEM_ID):

        child_list = []

        for i in range(len(superID)):
            if ITEM_ID == superID[i]:
                child_list.append(i)

        return child_list

    def treetest(input):

        if len(check_list) > 0:
            check_list.remove(check_list[0])
        if input not in global_list:
            global_list.append(input)
        a = check_child(dataset['ITEMID'][input])
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
        print(i, dataset['ITEMNAME'][global_list[i]], global_list[i])
        return_list.append(dataset['ITEMNAME'][global_list[i]])
    return return_list

