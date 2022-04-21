import numpy as np
from materials.MaterialManager import MaterialManager


class Element(list):
    dofTypes = []

    def __init__(self, elnodes, props):
        super().__init__(self, elnodes)
        self.history = []
        self.current = []

        for name, val in props:
            if name == 'material':
                self.matProps = val
                self.mat = MaterialManager(self.matProps)
            else:
                setattr(self, name, val)

    def dofCount(self):
        return len(self) * len(self.dofTypes)

    def getNodes(self):
        return self

    def getType(self):
        return self

    def appendNodalOutput(self, outputNames, globdat, outmat, outw=None):

        if outw == None:
            outw = np.ones(outmat.shape[0])

        for i, name in enumerate(outputNames):
            if not hasattr(globdat, name):
                globdat.outputNames.append(name)

                setattr(globdat, name, np.zeros(len(globdat.nodes)))
                setattr(globdat, name + 'Weights', np.zeros(len(globdat.nodes)))

            outMat = getattr(globdat, name)
            outWeights = getattr(globdat, name + 'Weights')
            indi = globdat.nodes.getIndices(self)

            outMat[indi] += outmat[:, i]
            outWeights[indi] += outw

    def setHistoryParameter(self, name, val):
        self.current[name] = val

    def getHistoryParameter(self, name):
        return self.history[name]

    def commitHistory(self):
        self.history = self.current.copy()
        self.current = {}

        if hasattr(self, "mat"):
            self.mat.commitHistory()

    def commit(self, elemdat):
        pass
