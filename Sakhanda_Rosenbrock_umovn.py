import math
import numpy as np
import matplotlib.pyplot as plt

def proverka(x):
    # pass
    # if 1 <= x[0] <= 3 and -1 <= x[1] <= 1:
    #     if x[1] <= (1 - (x[0] - 2)**2):
    # if 0 <= x[0] <= 2 and -1 <= x[1] <= 1:
    #     if x[1] <= (1 - (x[0] - 1)**2):
    #         if x[1] <= 0.5 - (x[0] - 1 )**2:
    # if -1 <= x[0] <= 1 and -1 <= x[1] <= 1:
    #     if x[1] <= (1 - x[0] **2):
    if x[0] >= -x[1] :
            return 1
    else:
        return 0

number_func = 0
def func(x):
    global number_func
    number_func = number_func + 1
    return (10*(x[0] - x[1])**2 + (x[0] - 1)**2)**4
    # return ((x[0] - 1)**2 + (x[1] - 1)**2)

S1 = [1, 0]
S2 = [0, 1]
x0 = [-0.6, 0.7]
y0 = func(x0)
epsilon = 0.01
epsilon1 = 0.01
lambda_0 = []
x_r = [x0]
y_r = [y0]
a = 3
b = -0.5
sucs = 2

def sven(x0, S):
    global s
    x = [x0]
    y = [y0]
    l = [0]
    s = 0
    # dl = 0.01*(math.sqrt(x0[0]**2 + x0[1]**2))/math.sqrt(S[0]**2 + S[1]**2)
    dl = 0.1*math.sqrt(S[0]**2 + S[1]**2)/(math.sqrt(x0[0]**2 + x0[1]**2))
    # dl = 0.001
    f1 = func([x0[0] + dl*S[0], x0[1] + dl*S[1]])
    f2 = func([x0[0] - dl*S[0], x0[1] - dl*S[1]])
    if f1 <= y0:
        s = 1
        x.append([x0[0] + dl*S[0], x0[1] + dl*S[1]])
        y.append(f1)
    if f2 <= y0:
        s = -1
        x.append([x0[0] - dl*S[0], x0[1] - dl*S[1]])
        y.append(f2)
    l.append(dl)
    while y[-2] >= y[-1]:
        dl = s*dl*2
        l.append(dl)
        x.append([x[-1][0] + dl*S[0], x[-1][1] + dl*S[1]])
        y.append(func(x[-1]))
    x.append([(x[-1][0] + x[-2][0])/2, (x[-1][1] + x[-2][1])/2])
    y.append(func(x[-1]))
    n = y.index(min(y))
    if len(x) >= n+2:
        return [x[n-1], x[n+1]]
    else:
        return [x[n-1], x[n]]

def gold(x0,epsilon,S):
    a = sven(x0, S)[0]
    b = sven(x0, S)[1]
    L = [abs(b[0] - a[0]), abs(b[1] - a[1])]
    l1 = [a[0] + 0.382*L[0], a[1] + 0.382*L[1]]
    l2 = [a[0] + 0.618*L[0], a[1] + 0.618*L[1]]
    f1 = func([x0[0] + l1[0]*S[0], x0[1] + l1[1]*S[1]])
    f2 = func([x0[0] + l2[0]*S[0], x0[1] + l2[1]*S[1]])
    while L[0] > epsilon1 and L[1] > epsilon1:
        if f1 > f2:
            a = l1
            b = b
        elif f1 < f2:
            a = a
            b = l2
        L = [abs(b[0] - a[0]), abs(b[1] - a[1])]
        l1 = [a[0] + 0.382*L[0], a[1] + 0.382*L[1]]
        l2 = [a[0] + 0.618*L[0], a[1] + 0.618*L[1]]
        f1 = func([x0[0] + l1[0]*S[0], x0[1] + l1[1]*S[1]])
        f2 = func([x0[0] + l2[0]*S[0], x0[1] + l2[1]*S[1]])
    return L

def pauel(x0, S):
    l1 = sven(x0, S)[0]
    l3 = sven(x0, S)[1]
    l2 = [(l1[0] + l3[0])/2, (l1[1] + l3[1])/2]
    l = [l2]
    f1 = func(l1)
    f2 = func(l2)
    f3 = func(l3)
    y = [f2]
    if f1 <= f3:
        l.append(l1)
        y.append(f1)
        dl = [-abs(l2[0] - l1[0]), -abs(l2[1] - l1[1])]
    elif f1 >= f3:
        l.append(l3)
        y.append(f3)
        dl = [abs(l2[0] - l3[0]), abs(l2[1] - l3[1])]
    l1 = l[0]
    l2 = l[-1]
    while y[-2] > y[-1]:
        if abs(y[-2] - y[-1]) > epsilon:
            dl = [2*dl[0], 2*dl[1]]
            l.append([l2[0] + dl[0], l2[1] + dl[1]])
            y.append(func(l[-1]))
        else:
            break
    l.append([(l[-1][0] + l[-2][0])/2, (l[-1][1] + l[-2][1])/2])
    y.append(func(l[-1]))
    n = y.index(min(y))
    return [abs(l[n-1][0] - l[n][0]), abs(l[n-1][1] - l[n][1])]

n = 1
# lambda_0 = [pauel(x0, S1)[0], pauel(x0, S2)[1]]
while True:
    x0 = x_r[-1]
    x_temp = x0
    y_temp = y_r[-1]
    # lambda_0 = [pauel(x0, S1)[0], pauel(x0, S2)[1]]
    # lambda_0 = [0.0001, 0.0001]
    lambda_0 = [gold(x0,epsilon,S1)[0], gold(x0,epsilon,S2)[1]]
    while True:
        temp = func([x_temp[0] + S2[0]*lambda_0[1], x_temp[1] + S2[1]*lambda_0[1]])
        # print(temp, y_temp)
        if temp <= y_temp and proverka([x_temp[0] + S2[0]*lambda_0[1], x_temp[1] + S2[1]*lambda_0[1]]):
            y_temp = temp
            x_temp = [x_temp[0] + S2[0]*lambda_0[1], x_temp[1] + S2[1]*lambda_0[1]]
            lambda_0 = [lambda_0[0], a*lambda_0[1]]
        else:
            sucs = sucs - 1
            lambda_0 = [lambda_0[0], b*lambda_0[1]]
        temp = func([x_temp[0] + S1[0]*lambda_0[0], x_temp[1] + S1[1]*lambda_0[0]])
        if temp <= y_temp and proverka([x_temp[0] + S1[0]*lambda_0[0], x_temp[1] + S1[1]*lambda_0[0]]):
            y_temp = temp
            x_temp = [x_temp[0] + S1[0]*lambda_0[0], x_temp[1] + S1[1]*lambda_0[0]]
            lambda_0 = [a*lambda_0[0], lambda_0[1]]
        else:
            sucs = sucs - 1
            lambda_0 = [b*lambda_0[0], lambda_0[1]]
        if sucs == 0 :
            x_r.append(x_temp)
            y_r.append(y_temp)
            break
        else:
            sucs = 2
            n = n + 1


    if abs(y_r[-1] - y_r[-2]) <= epsilon and math.sqrt((x_r[-1][0] - x_r[-2][0])**2 + (x_r[-1][0] - x_r[-2][0])**2) <= epsilon:
    # if math.sqrt((x_r[-1][0] - x_r[-2][0])**2 + (x_r[-1][0] - x_r[-2][0])**2) <= epsilon:
        print(number_func)
        print(x_r[-1], y_r[-1])
        break
    temp = math.sqrt((x_r[-1][0] - x_r[-2][0])**2 + (x_r[-1][1] - x_r[-2][1])**2)
    S1 = [(x_r[-1][0] - x_r[-2][0])/temp, (x_r[-1][1] - x_r[-2][1])/temp]
    temp = -(S2[0]*S1[0] + S2[1]*S1[1])/(S1[0]*S1[0] + S1[1]*S1[1])
    S2 = [S2[0] + temp*S1[0], S2[1] + temp*S1[1]]

# print(x_r)
x = np.array(x_r)[:,0]
y = np.array(x_r)[:,1]
plt.plot(x, y, '-o')
plt.plot(x_r[-1][0], x_r[-1][1], '-x')
plt.plot(x_r[0][0], x_r[0][1], '-x')
plt.plot(1, 1, '-x')

m = 0
x = np.arange(m-1, m+1, 0.0001)
y_ = []
x_ = [x]
def h(x):
    y = 1 - (x - m )**2
    for i in range(len(x)):
        y_.append(math.sqrt(y[i]))
    return y_

def f(x):
    y = 1 - (x - m)**2
    for i in range(len(x)):
        y_.append(-1*math.sqrt(y[i]))
    return y_

def f1(x):
    for i in range(len(x)):
        y_.append(-1*x[i])
    return y_

# y = h(x)
# y_ = []
# y1 = f(x)
#
# m = 1
# x2 = np.arange(m-0.7071, m+0.7071, 0.0001)
# y_ = []
# x_ = [x]
# def h2(x):
#     y = 0.5 - (x - m )**2
#     for i in range(len(x)):
#         y_.append(math.sqrt(y[i]))
#     return y_
#
# def f2(x):
#     y = 0.5 - (x - m)**2
#     for i in range(len(x)):
#         y_.append(-1*math.sqrt(y[i]))
#     return y_
# y2 = h2(x2)
# y_ = []
# y3 = f2(x2)

y = f1(x)
plt.plot(x, y)
# plt.plot(x, y, x, y1)
# plt.plot(x2, y2, x2, y3)
# plt.plot(x, y)
plt.show()
