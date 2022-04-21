import numpy as np
from util.itemList import itemList
import re, sys


class NodeSet(itemList):

    def __init__(self):
        self.rank = -1

    def getNodeCoords(self, nodeIDs):
        return np.array(self.get(nodeIDs))

    def readFromFile(self, fname):

        print("  Reading nodes ................")

        fin = open(fname)

        while True:

            line = fin.readline()

            if line.startswith('<Nodes>') == True:

                while True:
                    line = fin.readline()

                    if line.startswith('</Nodes>') == True:
                        return

                    line = re.sub('\s{2,}', ' ', line)
                    a = line.split(';')

                    for a in a[:-1]:
                        b = a.strip().split(' ')

                        if b[0].startswith("//") or b[0].startswith("#"):
                            break
                        if len(b) > 1 and type(eval(b[0])) == int:
                            if self.rank == -1:
                                self.rank = len(b) - 1

                            self.add(eval(b[0]), [eval(crd) for crd in b[1:]])


            elif line.startswith('gmsh') == True:
                ln = line.replace('\n', '').replace('\t', '').replace(' ', '').replace('\r', '').replace(';', '')
                ln = ln.split('=', 1)
                self.readGmshFile(ln[1][1:-1])
                return

    def readGmshFile(self, fname):

        fin = open(fname)

        while True:

            line = fin.readline()

            if line.startswith('$MeshFormat') == True:

                while True:
                    line = fin.readline()
                    line = re.sub('\s{2,}', ' ', line)

                    a = line.split(';')
                    b = a[0].strip().split(' ')

                    if eval(b[0]) < 2.0:
                        print("error")
                        sys.exit()

                    break

            if line.startswith('$Nodes'):

                nNodes = eval(fin.readline())

                for i in range(nNodes):
                    line = fin.readline()
                    line = re.sub('\s{2,}', ' ', line)
                    b = line.strip().split(' ')

                    if len(b) > 1 and type(eval(b[0])) == int:
                        self.add(eval(b[0]), [eval(crd) for crd in b[1:3]])

            if line.startswith('$EndNodes'):
                return

            # ------

    def __repr__(self):
        return f"Nodeset contains {len(self)} nodes.\n"
