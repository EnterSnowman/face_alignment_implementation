import cv2 as cv

forest = cv.ml.RTrees_create()
forest.setMaxDepth(6)
forest.setMaxNumOfTreesInTheForest(6)
# print(help(cv.ml_RTrees))
print(forest.getMaxDepth())

