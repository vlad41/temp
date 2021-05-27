import math
import numpy as np

number_func = 0
def func(x):
    global number_func
    number_func = number_func + 1
    return (10*(x[0] - x[1])**2 + (x[0] - 1)**2)**4
    # return ((x[0] - 1)**2 + (x[1] - 1)**2)

S1 = [1, 0]
S2 = [0, 1]
x0 = [1.2, 0]
y0 = func(x0)
epsilon = 0.001
epsilon1 = 0.001
lambda_0 = []
x_r = [x0]
y_r = [y0]
a = 4
b = -0.5
sucs = 2

def sven(x0, S):
    global s
    x = [x0]
    y = [y0]
    l = [0]
    s = 0
    # dl = 0.01*(math.sqrt(x0[0]**2 + x0[1]**2))/math.sqrt(S[0]**2 + S[1]**2)
    dl = 0.001*math.sqrt(S[0]**2 + S[1]**2)/(math.sqrt(x0[0]**2 + x0[1]**2))
    # dl = 0.01
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
    lambda_0 = [pauel(x0, S1)[0], pauel(x0, S2)[1]]
    # lambda_0 = [0.0001, 0.0001]
    # if n == 1:
    #     lambda_0 = [pauel(x0, S1)[0], pauel(x0, S2)[1]]
    # lambda_0 = [gold(x0,epsilon,S1)[0], gold(x0,epsilon,S2)[1]]
    n= 2
    while True:
        temp = func([x_temp[0] + S2[0]*lambda_0[1], x_temp[1] + S2[1]*lambda_0[1]])
        if temp <= y_temp:
            y_temp = temp
            x_temp = [x_temp[0] + S2[0]*lambda_0[1], x_temp[1] + S2[1]*lambda_0[1]]
            lambda_0 = [lambda_0[0], a*lambda_0[1]]
        else:
            sucs = sucs - 1
            lambda_0 = [lambda_0[0], b*lambda_0[1]]
        temp = func([x_temp[0] + S1[0]*lambda_0[0], x_temp[1] + S1[1]*lambda_0[0]])
        if temp <= y_temp:
            y_temp = temp
            x_temp = [x_temp[0] + S1[0]*lambda_0[0], x_temp[1] + S1[1]*lambda_0[0]]
            lambda_0 = [a*lambda_0[0], lambda_0[1]]
        else:
            sucs = sucs - 1
            lambda_0 = [b*lambda_0[0], lambda_0[1]]
        if sucs == 0:
            x_r.append(x_temp)
            y_r.append(y_temp)
            break
        else:
            sucs = 2
    # if abs(y_r[-1] - y_r[-2]) <= epsilon and math.sqrt((x_r[-1][0] - x_r[-2][0])**2 + (x_r[-1][0] - x_r[-2][0])**2) <= epsilon:
    if math.sqrt((x_r[-1][0] - x_r[-2][0])**2 + (x_r[-1][0] - x_r[-2][0])**2) <= epsilon:
        print(number_func)
        print(x_r, y_r)
        break
    temp = math.sqrt((x_r[-1][0] - x_r[-2][0])**2 + (x_r[-1][1] - x_r[-2][1])**2)
    S1 = [(x_r[-1][0] - x_r[-2][0])/temp, (x_r[-1][1] - x_r[-2][1])/temp]
    temp = -(S2[0]*S1[0] + S2[1]*S1[1])/(S1[0]*S1[0] + S1[1]*S1[1])
    S2 = [S2[0] + temp*S1[0], S2[1] + temp*S1[1]]
