import pymysql
import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Module.main import test1
from Module.sequence_Tree import total_tree
from Module.fix_sequence import fix

def toNLP(sentence, pID, rID):

    conn = pymysql.connect(host='192.168.0.28', port=3306,
                           user='ATUM_DEV_ENHANCE', password='become1!', database='ATUM_DEV_ENHANCE')
    cur = conn.cursor()

    sql = f"SELECT * FROM RULE_TRAININGDATA"

    cur.execute(sql)

    result = cur.fetchall()

    col_name = ["CHATBOTID", "ANSWERID", "QUESTION", "ANSWERTYPE", "ANSWERDATA", "KEYWORD1",
                "KEYWORD2", "KEYWORD3", "KEYWORD4", "KEYWORD5", "DESCRIPTION"]
    df = pd.DataFrame(result, columns=col_name)
    test11 = list(df['QUESTION'])
    print(test11)

    def preprocessing(sequence, method):
        new_sequence = method.morphs(sequence, stem=False)
        return new_sequence

    okt = Okt()

    NLP = preprocessing(sentence, okt)

    test_sentence = ""

    # 띄어쓰기가 안된 문장을 띄어쓰고 오른쪽 공백 제거
    for i in range(len(NLP)):
        test_sentence += (NLP[i] + " ")
    test_sentence = test_sentence.rstrip()
    print(test_sentence)
    list_NLP = list(test_sentence)

    Upper_sentence = ""

    # 문장 내 영어가 있을 경우 대문자화
    for i in range(len(list_NLP)):
        if str(list_NLP[i]).isalpha():
            list_NLP[i] = str(list_NLP[i]).upper()
        Upper_sentence += list_NLP[i]

    test11.append(Upper_sentence)

    # 문장 추가된 리스트를 TfidfVectorizer, cosine_similarity 수행
    vectorizer = TfidfVectorizer()
    sp_matrix = vectorizer.fit_transform(test11)
    print(sp_matrix)
    # 추가된 문장은 리스트의 맨 끝에 추가되므로 [-1]번째 벡터와 전체 희소행렬을 비교함
    similar_vector_values = cosine_similarity(sp_matrix[-1], sp_matrix)
    # 같은 문장끼리 비교할 경우 코사인 유사도가 1이 나오므로 같은 문장 제외 가장 높은 유사도 인덱스 추출
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    # 만약 관련된 문장이 아예 없을 경우 모든 코사인 유사도가 0이 됨
    matched_vector = similar_vector_values.flatten()
    print(matched_vector)
    matched_vector.sort()
    vector_matched = matched_vector[-2]
    print(vector_matched)

    if vector_matched == 0:
        text = "일치하는 데이터 x"
        print(text)
        print(sentence)
        return {"ANSWERID":"", "ANSWERDATA":text}

    else:
        text = test11[similar_sentence_number]
        print("유사한 문장 인덱스 : ", similar_sentence_number)
        print("유사한 문장 : ", text)
        print(sentence)
        test_index = test11.index(text)
        print(df['ANSWERDATA'][test_index])
        print(df['ANSWERID'][test_index])
        if df['ANSWERDATA'][test_index] == "1":

            conn = pymysql.connect(host='192.168.0.28', port=3306,
                                   user='ATUM_DEV_ENHANCE', password='become1!', database='ATUM_DEV_ENHANCE')
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
                return {"ANSWERID":"", "ANSWERDATA":'biz item을 더 추가하세요.'}
            else:
                a = test1(pID, rID)
            return {"ANSWERID":"", "ANSWERDATA": a}
        else:
            return {"ANSWERID":df['ANSWERID'][test_index], "ANSWERDATA":""}