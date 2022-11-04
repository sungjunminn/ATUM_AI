from konlpy.tag import Okt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

train = pd.read_excel("Dataset.xlsx", header=0, index_col=None)
print(train['sentence'])
test = list(train['sentence'])
print(test)


def preprocessing(sequence, method):
    new_sequence = method.morphs(sequence, stem=False)
    return new_sequence


okt = Okt()

while 1:
    test_input = input()

    if test_input == 'break':
        break

    start = time.time()

    NLP = preprocessing(test_input, okt)

    test_sentence = ""

    # 띄어쓰기가 안된 문장을 띄어쓰고 오른쪽 공백 제거
    for i in range(len(NLP)):
        test_sentence += (NLP[i] + " ")
    test_sentence = test_sentence.rstrip()

    list_NLP = list(test_sentence)

    Upper_sentence = ""

    # 문장 내 영어가 있을 경우 대문자화
    for i in range(len(list_NLP)):
        if str(list_NLP[i]).isalpha():
            list_NLP[i] = str(list_NLP[i]).upper()
        Upper_sentence += list_NLP[i]

    test.append(Upper_sentence)
    print(test)

    # 문장 추가된 리스트를 TfidfVectorizer, cosine_similarity 수행
    vectorizer = TfidfVectorizer()
    sp_matrix = vectorizer.fit_transform(test)

    # 추가된 문장은 리스트의 맨 끝에 추가되므로 [-1]번째 벡터와 전체 희소행렬을 비교함
    similar_vector_values = cosine_similarity(sp_matrix[-1], sp_matrix)
    # 같은 문장끼리 비교할 경우 코사인 유사도가 1이 나오므로 같은 문장 제외 가장 높은 유사도 인덱스 추출
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    # 만약 관련된 문장이 아예 없을 경우 모든 코사인 유사도가 0이 됨
    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched == 0:
        print("일치하는 데이터 x")
    else:
        print("유사한 문장 인덱스 : ", similar_sentence_number)
        print("유사한 문장 : ", test[similar_sentence_number])
    end = time.time()
    print("실행 시간 : ", end - start)
