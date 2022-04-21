from util.dataStructures import Properties


class MaterialManager(list):

    def __init__(self, matProps):
        matType = matProps.type
        material = getattr(__import__('.materials.' + matType, globals(), locals(), matType, 0), matType)
        self.mat = material(matProps)
        self.iIter = -1

    def reset(self):
        self.iIter = -1

    def getStress(self, kinematic, iSam=-1):
        if iSam == -1:
            self.iIter += 1
            iSam = self.iIter
        self.mat.setIter(iSam)
        return self.mat.getStress(kinematic)

    def commitHistory(self):
        self.mat.commitHistory()
