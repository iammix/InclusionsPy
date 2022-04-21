class itemList(dict):

    def add(self, ID, item):
        if ID in self:
            raise RuntimeError(f'ID {ID} already exists in {type(self).__name__}')
        self[ID] = item

    def get(self, IDs):

        if isinstance(IDs, int):
            return self[IDs]
        elif isinstance(IDs, list):
            return [self[ID] for ID in IDs]

        raise RuntimeError('Illegal argument for itemList.get')

    def getIndices(self, IDs):

        if isinstance(IDs, int):
            return list(self.keys()).index(IDs)
        elif isinstance(IDs, list):
            return [list(self.keys()).index(ID) for ID in IDs]

        raise RuntimeError('illegal argument for itemList.getIndices')
