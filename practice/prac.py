def checkvalue(num):
    a = int(num**0.5)
    if num == 1:
        return False
    else:
        for i in range(2, a+1):
            if num % i == 0:
                return False
        return True

lst = []

for i in list(range(2, 246912)):
    if checkvalue(i):
        lst.append(i)

while True:
    M = int(input())
    cnt = 0
    if M == 0:
        break
    for i in lst:
        if M < i <= M*2:
            cnt += 1
    print(cnt)








# while True:
#     M = int(input())
#     cnt = 0
#     if M == 0:
#         break
#     for num in range(M+1,2*M+1):
#         for k in range(2, int(num**0.5)+1): #제곱근까지만 나누기
#             if num % k == 0:
#                 break
#         else:
#             cnt += 1
#
#     print(cnt)

