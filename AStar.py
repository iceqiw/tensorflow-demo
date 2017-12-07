import numpy as np
import queue


class Node:
    def __init__(self, x, y, parent=None, g=0, h=0):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = g
        self.h = h

    def getF(self):
        return self.g+self.h

weight=100
hight=100

map = np.zeros((weight, hight))
openSet = {}
closedSet = {}

DIRECT_VALUE = 10
startPoint = Node(2, 2)
endPoint = Node(98, 98)

# print(map[100,100])


def canAddNodeToOpen(x, y):
    if x < 0 or y < 0 or x > weight or y > hight:
        return False
    if map[x-1, y-1] == 1:
        return False
    if isCoordInClose(x, y):
        return False
    return True


def isCoordInClose(x, y):
    if closedSet.get((x, y)):
        return True
    return False


def isEndPointSet():
    return closedSet.get((endPoint.x, endPoint.y))


def moveNodes():
    while openSet:
        if isEndPointSet():
            drawPath()
            break
        current = getSmallOne()
        closedSet[(current.x, current.y)] = current

        addNeighborNodeInOpen(current)


def addNeighborNodeInOpen(current):
    addNeighborNodeInOpenStep(current, current.x - 1, current.y, DIRECT_VALUE)
    addNeighborNodeInOpenStep(current, current.x, current.y - 1, DIRECT_VALUE)
    addNeighborNodeInOpenStep(current, current.x + 1, current.y, DIRECT_VALUE)
    addNeighborNodeInOpenStep(current, current.x, current.y + 1, DIRECT_VALUE)


def getSmallOne():
    sd = sorted(openSet.items(), key=lambda d: d[1].getF())
    val = openSet.pop(sd[0][0])
    return val


def addOpenSet(node):
    openSet[(node.x, node.y)] = node


def addNeighborNodeInOpenStep(current, x, y, value):
    if canAddNodeToOpen(x, y):
        G = current.g + value
        child = findNodeInOpen(x, y)
        if not child:
            H = calcH(endPoint.x, endPoint.y, x, y)
            child = Node(x, y, current, G, H)
            addOpenSet(child)
        elif child.g > G:
            child.g = G
            child.parent = current
            addOpenSet(child)


def findNodeInOpen(x, y):
    return openSet.get((x, y))


def drawPath():
    end = closedSet.get((endPoint.x, endPoint.y))
    while end:
        map[end.x-1, end.y-1] = 5
        end = end.parent
    print(map)


def start():
    openSet[(startPoint.x, startPoint.y)] = startPoint
    moveNodes()


def calcH(ex, ey, x, y):
    return (abs(ex - x) + abs(ey - y))


if __name__ == '__main__':
    start()