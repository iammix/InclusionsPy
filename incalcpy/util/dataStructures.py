import numpy as np


class Properties(object):
    def __init__(self, dictionary=[]):
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

    def __str__(self) -> str:
        myStr = ""
        for att in dir(self):
            if att.startswith('__') or att.startswith('_'):
                continue
            myStr += f"Attribute: {att}\n"
            myStr += f"{getattr(self, att)} \n"
        return myStr

    def __iter__(self) -> iter:
        propertyList = []
        for att in dir(self):
            if att.startswith('__') or att.startswith('_'):
                continue
            propertyList.append((att, getattr(self, att)))
        return iter(propertyList)

    def store(self, key, val):
        setattr(self, key, val)


class GlobalData(Properties):
    def __init__(self, nodes, elements, dofs):
        super().__init__(self, {'nodes': nodes, 'elements': elements, 'dofs': dofs})
        self.state = np.zeros(len(self.dofs))
        self.Dstate = np.zeros(len(self.dofs))
        self.fint = np.zeros(len(self.dofs))
        self.fhat = np.zeros(len(self.dofs))
        self.velocity = np.zeros(len(self.dofs))
        self.acceleration = np.zeros(len(self.dofs))

        self.cycle = 0
        self.iiter = 0
        self.time = 0.0

        self.outputNames = []
        
    # TODO: Complete the code to read from file
    def readFromFile(self, fname):
        print("Reading External Forces . . . ")
        with open(fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('<ExternalForces>'):
                    pass


    def resetNodalOutput(self):
        for outputName in self.outputNames:
            delattr(self, outputName)
            delattr(self, f"{outputName}Weights")
        self.outputNames = []


class ElementData():

    def __init__(self, elstate, elDstate):
        nDof = len(elstate)
        self.state = elstate
        self.Dstate = elDstate
        self.stiff = np.zeros(shape=(nDof, nDof))
        self.fint = np.zeros(shape=nDof)
        self.mass = np.zeros(shape=(nDof, nDof))
        self.lumped = np.zeros(shape=nDof)
        self.outlabel = []

    def __str__(self) -> str:
        return self.state
