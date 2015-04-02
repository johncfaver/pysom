#!/usr/bin/env python

#Generate a 2D self-organizing map of 3D RGB color space.
#Generate a test set of colors and predict their positions in the 2D representation.

from random import random
from matplotlib import pyplot as plt

import som

ntrain=500
ntest= 50

#Generate training data in RGB space (random colors)
color_data = [ [ random()*255 for i in xrange(3) ] for i in xrange(ntrain) ]

#Create map object
mymap = som.map(color_data, nnodes=900)

#Train map
mymap.train(nsteps=1000, progress=True)

#Generate test set of random colors, predict their map coordinates.
test_x      = []
test_y      = []
test_color  = []
for i in xrange(ntest):
    t = [ random()*255 for j in xrange(3) ]
    #Predict position in map
    pred = mymap.classify(t)
    test_color.append([ j/255. for j in t ])
    test_x.append(pred[0])
    test_y.append(pred[1])

#Generate plots
nodexs = [ i.lowCoords[0] for i in mymap.nodes ]
nodeys = [ i.lowCoords[1] for i in mymap.nodes ]
nodecolors = []
for i in xrange(mymap.nnodes):
    c = mymap.unnormalizeDataPoint(mymap.nodes[i].highCoords)
    c = [ c[i]/255. if (c[i] > 0) else 0. for i in xrange(len(c)) ]
    nodecolors.append(c)

#Plot nodes
plt.scatter(nodexs,nodeys,c=nodecolors,s=200,marker='s')
#Plot predicted positions of test data
plt.scatter(test_x,test_y,c=test_color,s=50,marker='o')
plt.savefig('colors_example.png')

