from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, maxDisappeared=20):
        self.nextObjectID = 0
        self.objects = OrderedDict()#质心坐标的映射
        self.disappeared = OrderedDict()#消失帧数的映射
        self.maxDisappeared = maxDisappeared

    def register(self, centroid,rect):
        if len(self.objects) >2:
            print("检测到超过两个人，退出程序")
            return
        self.objects[self.nextObjectID] = (rect, centroid)#为新对象分配id，存储质心
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1#递增可用id

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) >2:
            rects = rects[:2]
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1#没有检测到矩形框，增加消失计数
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)#超过阈值注销该对象
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, w, h)) in enumerate(rects):
            #计算质心坐标
            inputCentroids[i] = (int(startX + w / 2.0), int(startY + h / 2.0))

        #如果当前无对象，注册所有新检测的质心
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i],rects[i])
        else:
            #获取已跟踪对象的id和质心(上一帧的位置)
            objectIDs = list(self.objects.keys())
            objectCentroids = [val[1] for val in self.objects.values()]
            #计算旧质心，新质心四个点之间的距离
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()#每个旧点对应的最小距离
            cols = D.argmin(axis=1)[rows]#最小距离对应的新点
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                objectID = objectIDs[row]
                self.objects[objectID] = (rects[col], inputCentroids[col])#更新box和质心
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            #未匹配的旧id
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            #未匹配的新点
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            #旧id比新点多--有人走了
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:#有人来了
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col])
        return self.objects