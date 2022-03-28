import numpy as np


class Properties:
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
        self.Dstate = np.zeros(dofs)
