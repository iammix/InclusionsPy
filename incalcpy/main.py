import numpy as np
from util.dataStructures import Properties
from fem.NodeSet import NodeSet


class Pile:
    def __init__(self, s, r, Ep, gp, pile_length):
        self.s = s
        self.r = r
        self.Ep = Ep
        self.gp = gp
        self.Apile = np.pi * r ** 2
        self.Wp = gp * self.Apile

    def create_elements(self):
        props = Properties()
        props.TrussP = Properties({'type': 'Truss',
                                   'E': self.Ep,
                                   'Area': self.Apile})

    def create_nodeSet(self):
        P_nodes = NodeSet()

