#!/usr/bin/env python

import sys
import random 
import math

def meanAndStd(x):
    """Return mean and standard devation of a list."""
    N = len(x)
    m = sum(x)/float(N)
    v = sum([ math.pow(x[i]-m,2.0) for i in xrange(N) ])/(N-1)
    return m, math.sqrt(v)

def squareDistance(x,y):
    """Return squared Euclidian distance between points x and y."""
    return sum([ math.pow(x[i]-y[i],2) for i in xrange(len(x)) ])

def distance(x,y):
    """Return Euclidian distance between points x and y."""
    return math.sqrt(squareDistance(x,y))

class map():
    def __init__(self, data, nnodes=100, dimension=2, normalize=True):
        """Set up a self-organizing map to model input data.
        Input data should be a list of lists (2D array-like)
        with no headers.
        """
        self.data           = data            
        self.ndata          = len(data)
        self.nnodes         = nnodes
        self.lowDimension   = dimension
        self.highDimension  = len(self.data[0])
        self.mapLength      = int(math.pow(nnodes,1./dimension))
        self.normalize      = normalize
        self.ignoreColumns  = []
        self.nodes          = []
        
        #Check input data
        self.checkInputs() 
        #Normalize input data 
        if self.normalize:
            self.normalizeData()
        #Set up the map.
        self.initializeMap()
   
    def checkInputs(self):
        """Convert input data to float, and check other variables."""
        print "Checking input data...",
        assert(self.ndata > 0), "No input data!"
        for row in self.data:
            assert(len(row) == self.highDimension), "Input data is not a regular matrix!"
            row = [ float(i) for i in row ]
        assert(self.nnodes > 0), "No nodes!"
        assert(self.lowDimension > 0 and self.lowDimension < self.highDimension), "Bad dimensions!"
        print "Done."
        print "Ready to map {}-D data to {}-D representation.".format(self.highDimension, self.lowDimension)
        print "Using {} nodes.".format(self.nnodes)
        print "Training set size: {}".format(self.ndata)
        print "Map length: {}".format(self.mapLength)


    def normalizeData(self):
        """Normalize input data by converting to z-values,
        i.e. (x-m)/s  where m is the mean and s is the standard deviation.
        If s == 0 then we will be ignoring this descriptor in the future.
        Save the m and s values so we can transform back to original data scale.
        """
        print "Normalizing Data...",
        self.originalData = self.data[:]
        self.normalizationParameters = []
        for d in xrange(self.highDimension):
            dmean, dstd = meanAndStd([ x[d] for x in self.originalData ])
            self.normalizationParameters.append((dmean,dstd))
            if dstd == 0.:
                self.ignoreColumns.append(d)
            else:
                for row in self.data:
                    row[d] = (row[d]-dmean)/dstd
        print 'Done.'
        if(len(self.ignoreColumns)>0):
            print 'Column(s) {} do not vary and will be ignored.'.format(self.ignoreColumns)

    def normalizeDataPoint(self, x):
        """Scale input data point to match normalized data"""
        x2 = x[:]
        for d in xrange(self.highDimension):
            if d in self.ignoreColumns:
                continue
            else:
                u = self.normalizationParameters[d][0]
                s = self.normalizationParameters[d][1]
                x2[d] = (x[d]-u)/s
        return x2

    def unnormalizeDataPoint(self, x):
        """Scale normalized data point to original input data"""
        x2 = x[:]
        for d in xrange(self.highDimension):
            if d in self.ignoreColumns:
                continue
            else:
                u = self.normalizationParameters[d][0]
                s = self.normalizationParameters[d][1]
                x2[d] = (x[d]*s)+u
        return x2

    def initializeMap(self):
        #Assume a 2D map for now
        self.initialize2DMap()

    def initialize2DMap(self):
        """Randomly set intial node positions in higher dimensional space,
        and place into regular square grid in 2D space. Initial positions in 
        high-dimensional space will be random but within 2 sigma.
        """
        for inode in xrange(self.nnodes):
            highCoords = [ random.random()*2.*random.choice([-1.,1.]) for i in xrange(self.highDimension) ]
            lowX = inode % self.mapLength
            lowY = inode / self.mapLength
            lowCoords = [lowX,lowY]
            self.nodes.append(node(highCoords,lowCoords,index=inode))
    
    def findBestMatch(self, x):
        """Return index of node nearest data point x"""
        distance_min = 1.e50
        index_min = -1
        for node in self.nodes:
            tdist = node.highSquareDistanceFrom(x)
            if tdist < distance_min:
                distance_min = tdist
                index_min = node.index
        if index_min == -1:
            raise Exception, "Couldn't find a close node!"
        return index_min

    def train(self, nsteps=100, progress=False):
        """Fit the map to the data.
        Search radius in low-dimensional space will decay over time as:
                r(t) = radius0*exp(-istep/nsteps*tscale)
        Learning rate will decrease over time as:
                L(t) = learn0*exp(-istep/nsteps)
        Node influence will decrease over time as (Gaussian):
                P(t) = exp(d^2/(2*r(t)^2))
                where d is the distance between nodes
        """
        print 'Training map with {} steps...'.format(nsteps)
        #Initial parameters
        radius0 = self.mapLength/2.
        radius0sq = math.pow(radius0,2)
        tscale  = math.log(radius0)
        learn0  = 0.1
        for istep in xrange(nsteps):
            if progress:
                if istep % int((nsteps/10.)) == 0:
                    print '{}% done'.format(float(istep)/nsteps*100)
            randomDataPoint = random.choice(self.data)
            closestNodeIndex = self.findBestMatch(randomDataPoint)
            searchRadius2 =  math.pow(radius0*math.exp(-float(istep)/nsteps*tscale),2)
            learningRate = learn0*math.exp(-float(istep)/nsteps)
            if searchRadius2 <= 1.0:
                break
            #print 'Step {}, searchRadius2={}, learningRate={}'.format(istep,searchRadius2,learningRate)
            for node in self.nodes:
                nodeDistance = node.lowSquareDistanceFrom(self.nodes[closestNodeIndex].lowCoords)
                if nodeDistance < searchRadius2:
                    translationVector = [randomDataPoint[i]-node.highCoords[i] for i in xrange(self.highDimension)]
                    translationVector = [ i*learningRate*math.exp(-nodeDistance/2./searchRadius2) for i in translationVector]
                    node.translate(translationVector)
        print 'Done.'

    def printGrid(self, filename=None):
        """Print grid to file or screen."""
        if filename is None:
            if self.normalize:
                for node in self.nodes:
                    print "Node {} at {} has value {}".format(node.index,node.lowCoords,self.unnormalizeDataPoint(node.highCoords))
            else:
                for node in self.nodes:
                    print "Node {} at {} has value {}".format(node.index,node.lowCoords,node.highCoords)
        else:
            with open(filename,'w') as f:
                if self.normalize:
                    for node in self.nodes:
                        f.write("Node {} at {} has value {}\n".format(node.index,node.lowCoords,self.unnormalizeDataPoint(node.highCoords)))
                else:
                    for node in self.nodes:
                        f.write("Node {} at {} has value {}\n".format(node.index,node.lowCoords,node.highCoords))

    def updateU_values(self, radius=2.):
        """Update u_values for each node.
        For each node, find nodes within radius in low-dimensional space, 
        and compute the average distance from this node to those nodes in
        the high-dimensional space.
        """
        radius2 = math.pow(radius,2)
        for node in self.nodes:
            node.u_value = 0.
            nneighbors = 0
            for node2 in self.nodes:
                if node == node2:
                    continue
                if any([ abs(node2.lowCoords[i]-node.lowCoords[i]) > radius for i in xrange(self.lowDimension) ]):
                    continue
                if squareDistance(node.lowCoords, node2.lowCoords) <= radius2:
                    node.u_value += distance(node.highCoords, node2.highCoords)
                    nneighbors += 1
            node.u_value /= nneighbors


    def classify(self, x):
        """Return map coordinates for new point x"""
        assert( len(x) == self.highDimension), "Wrong dimension for input data, received {} but requires {}".format(len(x), self.highDimension)
        if self.normalize:
            closestNodeIndex = self.findBestMatch(self.normalizeDataPoint(x))
        else:
            closestNodeIndex = self.findBestMatch(x)
        #print "Assigned input to node {}".format(closestNodeIndex)
        return self.nodes[closestNodeIndex].lowCoords

class node():
    def __init__(self, highCoords, lowCoords, index=None):
        """Each node consists of coordinates in the higher dimensional space
        and the lower dimensional space. Each node is also characterized by 
        a u_value, which is the average distance between this node and its
        low-dimensional neighbors in the high-dimensional space.
        """
        self.highCoords = highCoords
        self.highDimension = len(highCoords)
        self.lowCoords = lowCoords
        self.lowDimension = len(lowCoords)
        self.u_value = 0.
        self.index = index

    def lowDistanceFrom(self, x):
        """Return distance from point x in low dimensional space."""
        return distance(self.lowCoords,x)

    def lowSquareDistanceFrom(self, x):
        """Return squre distance from point x in low dimensional space."""
        return squareDistance(self.lowCoords,x)

    def highDistanceFrom(self, x):
        """Return distance from point x in high dimensional space."""
        return distance(self.highCoords,x)

    def highSquareDistanceFrom(self,x):
        """Return squre distance from point x in high dimensional space."""
        return squareDistance(self.highCoords,x)

    def translate(self, x):
        self.highTranslate(x)

    def highTranslate(self, x):
        """Translate this node in the high dimensional space by x."""
        assert( len(x) == len(self.highCoords)), "Bad node translation vector"
        #print "Moving node {} by {}".format(self.index,x)
        for i in xrange(self.highDimension):
            self.highCoords[i] += x[i]

    def lowTranslate(self, x):
        """Translate this node in the low dimensional space by x."""
        assert( len(x) == len(self.lowCoords)), "Bad node translation vector"
        #print "Moving node {} by {}".format(self.index,x)
        for i in xrange(self.lowDimension):
            self.lowCoords[i] += x[i]

   
