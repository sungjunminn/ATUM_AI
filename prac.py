def d(i):
    result = []
    for i in range(0, 9):
        i += 1
        b = i * 2
        result.append(b)
    return result


result = 0
for i in range(1, 1000):
    if i % 3 == 0 or i % 5 == 0:
        result += i


def getTotalPage(m,n):
    if m % n == 0:
        return m//n
    else:
        return m//n + 1
# print(getTotalPage(30,10))


#1
a = 'a:b:c:d'
b = a.split(':')
c = '#'.join(b)

#2
a = {'A':90, 'B':80}
a.get('C',70)

#3 extend를 사용하면 주소값이 유지됨
a = [1,2,3]
a = a + [4,5]
# print(id(a))

a = [1,2,3]
a.extend([4, 5])
# print(id(a))


#4
A = [20, 55, 67, 82, 45, 33, 90, 87, 100, 25]

result = 0
while A:
    mark = A.pop()
    if mark >= 50:
        result += mark
# print(result)


#5
def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    return fib(n-2) + fib(n-1)


#6
'''user_input = input("숫자입력")

numbers = user_input.split(',')
total = 0
for i in numbers:
    total += int(i)'''


#7
'''gugu_input = input("숫자를 입력하슈(2~9) : ")
dan = int(gugu_input)

for i in range(1,10):
    print(dan*i, end=' ')'''


#8
'''
import os
os.chdir('abc.txt')
f = open('abc.txt', 'r')
lines = f.readlines()
f.close()

lines.reverse()

f = open('abc.txt', 'w')
for line in lines:
    line = line.strip()
    f.write(line)
    f.write('\n')
f.close()
'''

#9
'''
import os
os.chdir('sample.txt')
f = open('sample.txt')
lines = f.readline()
f.close()

total = 0 
for line in lines:
    score = int(line)
    total += score
average = total / len(lines)

f = open('result.txt', 'w')
f.write(str(average))
f.close()
'''

#10
class Calculator:
    def __init__(self, numlist):
        self.numlist = numlist

    def sum(self):
        result = 0
        for num in self.numlist:
            result += num
        return result

    def avg(self):
        total = self.sum()
        return total/len(self.numlist)

cal1 = Calculator([1,2,3,4,5])


#11
data = '4546793'
def DashInsert(a):
    num = a.split()
    if a // 2 == 0:
        num


a = sum(x for x in range(1000) if x%3 == 0 or x%5 == 0)

sum = 0
for i in range(1000):
    if i%3 == 0 or i%5 == 0:
        sum += i


def tab_to_space(text):
    return text.replace("\t", " " * 4)








