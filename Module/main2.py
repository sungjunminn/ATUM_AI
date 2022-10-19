def test2(pID, rID):
    import pandas as pd
    import pymysql
    from Module.sequence_Tree import total_tree
    from Module.fix_sequence import fix
    from Module.LSTM import LSTM_main
    conn = pymysql.connect(host='192.168.0.28', port=3306,
                           user='ATUM_PRD_MESSTD_ADMIN', password='become1!', database='ATUM_PRD_MESSTD')
    cur = conn.cursor()
    '''
    projID = "MESSTD_V2"
    ruleID = "BR_DspTxnTrackOut"
    '''
    projID = pID
    ruleID = rID
    # sql = f"SELECT * FROM RULE_CURRENTITEM rc WHERE PROJECTID = '{projID}' AND RULEID = '{ruleID}'"
    sql = f"SELECT A.* FROM (SELECT (SELECT SEQNUM FROM RULE_BIZLOGICITEMGROUP C WHERE C.PROJECTID = '{projID}' AND C.RULEID ='{ruleID}' AND C.VERSION = '0.0.1' AND C.ITEMGROUPID = A.ITEMGROUPID) AS GROUP_SEQ, ITEMID, LEVEL, SUPERITEMID, SEQNUM, ITEMNAME FROM RULE_BIZLOGICITEM A WHERE A.PROJECTID = '{projID}' AND A.RULEID = '{ruleID}' AND A.VERSION = '0.0.1' AND A.ITEMNAME NOT LIKE '%HEAD' AND A.ITEMNAME NOT LIKE '%TAIL' AND A.CREATEDATE >= '2021-09-01 00:00:00.000' AND CASE 'TotalView' WHEN 'TotalView' then 1 = 1 ELSE A.ITEMGROUPID = 'TotalView' END ) A ORDER BY GROUP_SEQ, LEVEL, CASE WHEN LEVEL = 0 THEN SEQNUM END ASC, CASE WHEN LEVEL != 0 THEN SEQNUM END ASC;"

    cur.execute(sql)

    result = cur.fetchall()

    col_name = ["GROUP_SEQ", "ITEMID", "LEVEL", "SUPERITEM", "SEQNUM", "ITEMNAME"]
    df = pd.DataFrame(result, columns=col_name)
    print(df)
    SUPER_ID = df['SUPERITEM'].tolist()
    level_list = df['LEVEL'].tolist()
    level_0_list = []
    for i in range(len(level_list)):
        if level_list[i] == 0:
            level_0_list.append(i)
    sequence = total_tree(list=level_0_list, dataset=df, superID=SUPER_ID)
    new_sequence = fix(item_name_list=sequence)
    new_sequence = list(new_sequence)
    print("rule sequence : ", new_sequence)
    count = len(new_sequence)
    print(new_sequence)
    for i in new_sequence:
        if i == 'WHILE-MAINBLOCK':
            count -= 1
        if i == 'IF-MAINBLOCK':
            count -= 1
        if i == 'FOR-MAINBLOCK':
            count -= 1
    print(count)
    if count < 3:
        return 'biz item을 더 추가하세요.'
    train = pd.read_csv("Module\\test_name4.csv", header=0, index_col='INDEX')
    # print(train)
    item_list = []
    for i in range(len(train)):
        pd_train = train.loc[i].tolist()
        new_list = [x for x in pd_train if pd.isnull(x) == False]

        # print(new_list)
        item_list.append(new_list)
    item_col_list = []
    # len(item_list)

    for i in item_list:
        for j in i:
            item_col_list.append(j)

    # all cases of biz item
    unique_list = set(item_col_list)
    unique_list = list(unique_list)
    # print(len(unique_list), unique_list)
    import numpy as np

    min_freq = 2
    min_conf = 0.2

    test_count = 0
    total_list_1 = []
    total_list_2 = []
    total_list_3 = []
    for carda in range(1, 4):
        for cardb in range(1, 4):
            # set data structure
            list_a = []
            list_b = []
            list_freq = []
            list_conf = []

            # for a rule (= biz item list)
            for candi_rule in item_list:
                # all cases of association rule
                for i in range(len(candi_rule) - carda - cardb):
                    a = candi_rule[i:i + carda]
                    b = candi_rule[i + carda:i + carda + cardb]

                    if a in list_a and b in list_b:
                        # find exact index and increase number
                        for ii in range(len(list_a)):
                            if list_a[ii] == a and list_b[ii] == b:
                                list_freq[ii] = list_freq[ii] + 1
                                break

                    else:
                        # append
                        list_a.append(a)
                        list_b.append(b)
                        list_freq.append(1)
                        list_conf.append(0)
            # print(len(list_a), end=',')

            # remove association rules that are less than minimum frequency
            numbers = np.array(list_freq)
            del_tuple = np.where(numbers < min_freq)
            aa = del_tuple[0].tolist()
            for i in reversed(range(len(aa))):
                del list_a[aa[i]]
                del list_b[aa[i]]
                del list_freq[aa[i]]
                del list_conf[aa[i]]
            # print(len(list_a), end=',')

            # calculate confidence
            # 1. count number of a
            num_a = [0 for i in range(len(list_a))]
            for candi_rule in item_list:
                # all cases of association rule
                for i in range(len(candi_rule) - carda - cardb):
                    a = candi_rule[i:i + carda]
                    if a in list_a:
                        tidx = list_a.index(a)
                        num_a[tidx] = num_a[tidx] + 1

            # A can be redundant
            for i in range(len(list_a)):
                for j in range(i + 1, len(list_a)):
                    if list_a[i] == list_a[j] and num_a[j] == 0:
                        num_a[j] = num_a[i]

            # 2. fill in confidence variable
            for i in range(len(list_a)):
                list_conf[i] = list_freq[i] / num_a[i]

            # remove association rules that are less than minimum confidence
            numbers = np.array(list_conf)
            del_tuple = np.where(numbers < min_conf)
            aa = del_tuple[0].tolist()
            for i in reversed(range(len(aa))):
                del list_a[aa[i]]
                del list_b[aa[i]]
                del list_freq[aa[i]]
                del list_conf[aa[i]]
            # print(len(list_a))

            # print('size of a: {}, size of b: {}, num of rules: {}'.format(carda, cardb, len(list_a)))
            if len(list_a) != 0:
                test_count += len(list_a)
                for i in range(len(list_a)):
                    # print('{},{},{},{:.3f}'.format(list_a[i], list_b[i], list_freq[i], list_conf[i]))
                    test_list = []
                    if len(list_a[i]) == 1:
                        # print("length 1 : ", list_a[i])
                        test_list.append(list_a[i])
                        test_list.append(list_b[i])
                        test_list.append(list_freq[i])
                        test_list.append(list_conf[i])
                        total_list_1.append(test_list)
                    elif len(list_a[i]) == 2:
                        # print("length 2 : ", list_a[i])
                        test_list.append(list_a[i])
                        test_list.append(list_b[i])
                        test_list.append(list_freq[i])
                        test_list.append(list_conf[i])
                        total_list_2.append(test_list)
                    elif len(list_a[i]) == 3:
                        # print("length 3 : ", list_a[i])
                        test_list.append(list_a[i])
                        test_list.append(list_b[i])
                        test_list.append(list_freq[i])
                        test_list.append(list_conf[i])
                        total_list_3.append(test_list)

    # print(test_count)

    col_name = ["list_a", "list_b", "list_freq", "list_conf"]
    pd_total_list_1 = pd.DataFrame(total_list_1, columns=col_name)
    pd_total_list_2 = pd.DataFrame(total_list_2, columns=col_name)
    pd_total_list_3 = pd.DataFrame(total_list_3, columns=col_name)

    # list_test_text = list(test_text)
    # list_test_text = df["ITEMNAME"][1:4].tolist()
    # new_sequence.reverse()
    # list_test_text = new_sequence[:3]  # 테스트 끝나면 아래 주석 라인과 교체
    recomend_text = ''
    for k in range(0, len(new_sequence) - 3):
        recomend_text += new_sequence[k+2] + "/"
        list_test_text = new_sequence[k:k+3]
        # list_test_text.reverse()
        print("last 3 bizitem :", list_test_text)
        lstm_biztiem = LSTM_main(len_3_bizitem=list_test_text)
        return_list = []
        lstm_matching_list = []
        for i in range(len(pd_total_list_3)):

            if pd_total_list_3['list_a'][i] == list_test_text:
                print(pd_total_list_3['list_a'][i], pd_total_list_3['list_b'][i], pd_total_list_3['list_freq'][i],
                      pd_total_list_3['list_conf'][i])
                list_three = [pd_total_list_3['list_b'][i], pd_total_list_3['list_freq'][i],
                              pd_total_list_3['list_conf'][i]]
                return_list.append(list_three)
        tl3_list = pd_total_list_3['list_b'].tolist()
        for i in tl3_list:
            lstm_matching_list.append(i[0])
        list_test_text = list_test_text[1:]
        print("last 2 bizitem : ", list_test_text)
        for i in range(len(pd_total_list_2)):

            if pd_total_list_2['list_a'][i] == list_test_text:
                print(pd_total_list_2['list_a'][i], pd_total_list_2['list_b'][i], pd_total_list_2['list_freq'][i],
                      pd_total_list_2['list_conf'][i])
                list_two = [pd_total_list_2['list_b'][i], pd_total_list_2['list_freq'][i], pd_total_list_2['list_conf'][i]]
                return_list.append(list_two)
        tl2_list = pd_total_list_2['list_b'].tolist()
        for i in tl2_list:
            lstm_matching_list.append(i[0])
        list_test_text = list_test_text[1:]
        print("last 1 bizitem : ", list_test_text)
        for i in range(len(pd_total_list_1)):

            if pd_total_list_1['list_a'][i] == list_test_text:
                print(pd_total_list_1['list_a'][i], pd_total_list_1['list_b'][i], pd_total_list_1['list_freq'][i],
                    pd_total_list_1['list_conf'][i])
                list_one = [pd_total_list_1['list_b'][i], pd_total_list_1['list_freq'][i], pd_total_list_1['list_conf'][i]]
                return_list.append(list_one)
        tl1_list = pd_total_list_1['list_b'].tolist()
        for i in tl1_list:
            lstm_matching_list.append(i[0])

        lstm_matching_list = set(lstm_matching_list)
        lstm_matching_list = list(lstm_matching_list)
        lstm_matching_list = fix(item_name_list=lstm_matching_list)
        print("lstm_matching_list : ", lstm_matching_list)
        print("return_list : ", return_list)

        print("LSTM test : ", lstm_biztiem)
        TNF = False
        if lstm_biztiem in lstm_matching_list:
            TNF = True
        print("TNF : ", TNF)

        if len(return_list) == 0 and TNF == False:
            return '추천 아이템 x'
        else:
            comparison_name_list = []
            comparison_conf_list = []

            for j in range(len(return_list)):
                comparison_name_list.append(return_list[j][0][0])
                comparison_conf_list.append(return_list[j][2])
            np_comparison_conf_list = np.array(comparison_conf_list)
            s = np_comparison_conf_list.argsort()

            list_s = s.tolist()

            sort_list = []
            for i in range(len(list_s)):
                sort_list.append(comparison_name_list[list_s[i]])

            return_item = []
            return_item_text = ""
            for k in sort_list[::-1]:
                if k not in return_item:
                    return_item.append(k)

            for i in return_item:
                return_item_text += (str(i) + ", ")

            return_item_text = return_item_text.rstrip(", ")
            if len(return_list) == 0 and TNF:
                return_item_text += lstm_biztiem
            else:
                if lstm_biztiem not in return_item and TNF:
                    return_item_text += (", " + lstm_biztiem)
            print(return_item_text)
            recomend_text += return_item_text + "//"
        print(recomend_text)
    return recomend_text
