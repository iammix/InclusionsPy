import numpy as np
from util.dataStructures import Properties
from fem.NodeSet import NodeSet
from fem.ElementSet import ElementSet
from fem.DofSpace import DofSpace
from util.dataStructures import GlobalData


class Pile:
    def __init__(self, s, r, Ep, gp, pile_length, mesh=0.10, first_pile_length=0):
        self.s = s
        self.r = r
        self.Ep = Ep
        self.gp = gp
        self.mesh = mesh
        self.pile_length = pile_length
        self.first_pile_length = first_pile_length

        self.Apile = np.pi * r ** 2
        self.Wp = gp * self.Apile

    def create_elements(self):
        self.props = Properties()
        self.props.TrussP = Properties({'type': 'Truss',
                                        'E': self.Ep,
                                        'Area': self.Apile})

    def create_nodeSet(self):
        self.P_nodes = NodeSet()
        length = 0
        node_tag = 1
        while length <= self.pile_length:
            self.P_nodes.add(node_tag, [length, 0])
            node_tag += 1
            length += self.mesh

    def create_elementSet(self):
        self.P_elements = ElementSet(self.P_nodes, self.props)
        # TODO in case the first_pile_length is 0
        if self.first_pile_length != 0:
            length = 0
            node_tag = 1
            while length <= self.pile_length:
                if length <= self.first_pile_length:
                    self.P_elements.add(node_tag, 'TrussFirstP', [node_tag, node_tag + 1])
                    length += self.mesh
                else:
                    self.P_elements.add(node_tag, 'TrussP', [node_tag, node_tag + 1])
                    length += self.mesh
        else:
            pass

    def create_dofSpace(self):
        self.P_dofs = DofSpace(self.P_elements)
        self.P_globaldat = GlobalData(self.P_nodes, self.P_elements, self.Pdofs)

