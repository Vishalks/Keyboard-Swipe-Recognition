import random
import numpy as np

class Line(object):

    def __init__(self, data):
            self.first, self.second = data

    def slope(self):
            '''Get the slope of a line segment'''
            (x1, y1), (x2, y2) = self.first, self.second
            #try:
            if x2-x1 == 0:
                return 0
            return (float(y2)-y1)/(float(x2)-x1)
            #except ZeroDivisionError:
                    # line is vertical
                    #return None

    def yintercept(self, slope):
            '''Get the y intercept of a line segment'''
            if slope != None:
                    x, y = self.first
                    return y - slope * x
            else:
                    return None

    def solve_for_y(self, x, slope, yintercept):
            '''Solve for Y cord using line equation'''
            #if slope != None and yintercept != None:
            return round(float(slope) * x + float(yintercept), 2)
            #else:
                    #raise Exception('Can not solve on a vertical line')

    def solve_for_x(self, y, slope, yintercept):
            '''Solve for X cord using line equatio'''
            if slope != 0 and slope:
                    return float((y - float(yintercept))) / float(slope)
            else:
                    raise Exception('Can not solve on a horizontal line')

data = ((1,1), (2,3))

line = Line(data)

m = line.slope()

#print(m)

c = line.yintercept(m)

#print(c)


#xlist = np.linspace(1,10,4).tolist()

#ylist = []

#for x in xlist:
#    ylist.append(line.solve_for_y(x, m, c))

#print(xlist)
#print(ylist)
import math

#cordlist = [(1,2), (2,2), (-2,2), (-2,-2), (-3,-2)]

cordlist = [(170.0,50.0), (225.0,85.0), (135.0,50.0), (275.0,50.0), (330.0,85.0)]
dist = []
for i in range(len(cordlist) - 1):
    dist.append(math.hypot(cordlist[i][0] - cordlist[i+1][0], cordlist[i][1] - cordlist[i+1][1]))

cordsum = round(sum(dist), 2)
print(cordsum)
for i in range(len(dist)):
    dist[i] = (dist[i]/cordsum * 100)

absdist = [math.floor(x) for x in dist]
#print(absdist)
xlist = []
x = []
y = []

#data = ((1,1), (2,3))
#line = Line(data)
#m = line.slope()
#c = line.yintercept(m)
#for x in xlist:
#    ylist.append(line.solve_for_y(x, m, c))
plots = []
t = round(cordsum/99,2)
print(t)
#x = x1 + (x2-x1) * t
#y = y1 + (y2-y1) * t
remX = 0
remY = 0
'''
for i in range(len(dist)):
    x1 = cordlist[i][0]
    x2 = cordlist[i+1][0]
    y1 = cordlist[i][1]
    y2 = cordlist[i+1][1]
    plots.append((x1, y1))
    for j in range(absdist[i]):
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        plots.append((x, y))
        t = t * 2
    plots.append((x2, y2))
'''
'''
for i in range(len(dist)):
    data = ((cordlist[i][0],cordlist[i][1]), (cordlist[i+1][0],cordlist[i+1][1]))
    line = Line(data)
    m = line.slope()
    c = line.yintercept(m)
    x = np.linspace(cordlist[i][0], cordlist[i+1][0], absdist[i]).tolist()
    xlist += x
    print(m,c)
    for j in range(absdist[i]):
        y.append(line.solve_for_y(x[j], m, c))
'''
step = t
xDiff = 0.0
lastX = 0
for i in range(len(dist)):
    data = ((cordlist[i][0],cordlist[i][1]), (cordlist[i+1][0],cordlist[i+1][1]))
    line = Line(data)
    m = line.slope()
    c = line.yintercept(m)
    x1 = cordlist[i][0]
    x2 = cordlist[i+1][0]
    y1 = cordlist[i][1]
    y2 = cordlist[i+1][1]
    oldx1 = x1
    if i != 0:
        if x1 > x2:
            x1 = round(x1-(t-xDiff), 2)
        else:
            x1 = round(x1 + (t - xDiff), 2)
    x = x1
    y = y1
    if x2 > oldx1:
        while x < x2:
            lastX = x
            y = line.solve_for_y(x, m, c)
            print(x)
            plots.append((x,y))
            x = round(x + step, 2)
    else:
        while x >= x2:
            lastX = x
            y = line.solve_for_y(x, m, c)
            print(x)
            plots.append((x,y))
            x = round(x - step, 2)
    if x2 > oldx1:
        xDiff = round(x2 - lastX, 2)
    else:
        xDiff = round(lastX - x2, 2)
    print(xDiff)
    #print(lastX)

#print(xlist.__len__())
print(len(plots))
#for i,j in zip(xlist,y):
#   plots.append((i,j))
#for k in range(len(plots)  -1):
#    print(plots[k][0], plots[k+1][0])
#    print(plots[k][0] - plots[k+1][0])
#print(plots[0][0], plots[15][0], plots[16][0])
#print(plots)
#import matplotlib.pyplot as plt
#plt.scatter(*zip(*plots))
#plt.scatter(*zip(*cordlist), c='r')
#plt.show()

import numpy as np
import matplotlib.pyplot as plt
import math

x = [170.0, 225.0, 135.0, 275.0, 330.0]
y = [50.0, 85.0, 50.0, 50.0, 85.0]
# find lots of points on the piecewise linear curve defined by x and y
M = 100
t = np.linspace(0, len(x), M)
x = np.interp(t, np.arange(len(x)), x)
y = np.interp(t, np.arange(len(y)), y)
tol = 1.5
i, idx = 0, [0]
while i < len(x):
    total_dist = 0
    for j in range(i+1, len(x)):
        total_dist += math.sqrt((x[j]-x[j-1])**2 + (y[j]-y[j-1])**2)
        if total_dist > cordsum:
            idx.append(j)
            break
    i = j+1

xn = x[idx]
yn = y[idx]
fig, ax = plt.subplots()
ax.plot(x, y, '.')
#ax.scatter(xn, yn)
print(len(x))
#ax.set_aspect('equal')
plt.show()

for k in range(len(x) - 1):
    print(math.hypot(x[k] - x[k+1], y[k] - y[k+1]))

import matplotlib.pyplot as plt
import numpy as np
#x = [170.0, 225.0, 135.0, 275.0, 330.0]
x = [280.3999996185303, 279.3999996185303, 278.3999996185303, 277.3999996185303, 274.3999996185303, 272.3999996185303, 269.3999996185303, 268.3999996185303, 268.3999996185303]
#y = [50.0, 85.0, 50.0, 50.0, 85.0]
y = [116.9749984741211, 117.9749984741211, 117.9749984741211, 117.9749984741211, 119.9749984741211, 120.9749984741211, 120.9749984741211, 121.9749984741211, 121.9749984741211]
centroid = (sum(x) / len(x), sum(y) / len(y))
print(centroid)
for i in range(len(x)):
    print(str(x[i]) + ': ' + str(y[i]))
dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
print(dr)
print(len(dr))
print(len(x))
r = np.zeros_like(x)
r[1:] = np.cumsum(dr) # integrate path
r_int = np.linspace(0, r.max(), 100) # regular spaced path
x_int = np.interp(r_int, r, x) # interpolate
y_int = np.interp(r_int, r, y)

#plt.subplot(1,2,1)
#plt.plot(x, y, 'o-')
#plt.title('Original')
#plt.axis([-32,32,-32,32])

#plt.subplot(1,2,2)
plt.plot(x_int, y_int, 'o-')
plt.title('Interpolated')
print(len(x_int))
#plt.axis([-32,32,-32,32])
plt.show()

#for k in range(len(x_int)):
#    print(x_int[k] ,y_int[k])