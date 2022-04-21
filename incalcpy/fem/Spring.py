from util.transformations import toElementCoordinates, toGlobalCoordinates
import numpy as np
from Element import Element


class Spring(Element):
    dofTypes = ['u', 'v']

    def __init__(self, elnodes, props):
        super().__init__(self, elnodes, props)

    def getTangentStiffness(self, elemdat):
        # Compute the current state vector

        a = toElementCoordinates(elemdat.state, elemdat.coords)
        Da = toElementCoordinates(elemdat.Dstate, elemdat.coords)

        # Compute the elongation of the spring
        elong = a[2] - a[0]

        # Compute the force in the spring
        Fs = elong * elemdat.props.k

        # Compute the element internal force vector in the element coordinate system
        elFint = np.array([-Fs, 0., Fs, 0])

        # Determine the element tangent stiffness in the element coordinate system
        elKbar = np.zeros((4, 4))

        elKbar[:2, :2] = elemdat.props.k * np.eye(2)
        elKbar[:2, 2:] = -elemdat.props.k * np.eye(2)

        elKbar[2:, :2] = elKbar[:2, 2:]
        elKbar[2:, 2:] = elKbar[:2, :2]

        # Rotate element tangent stiffness to the global coordinate system
        elemdat.stiff = toGlobalCoordinates(elKbar, elemdat.coords)
        elemdat.fint = toGlobalCoordinates(elFint, elemdat.coords)

    # ------------------------------------------------------------------

    def getInternalForce(self, elemdat):
        # Compute the current state vector

        a = toElementCoordinates(elemdat.state, elemdat.coords)
        Da = toElementCoordinates(elemdat.Dstate, elemdat.coords)

        # Compute the elongation of the spring
        elong = a[2] - a[0]

        # Compute the force in the spring
        Fs = elong * elemdat.props.k

        # Compute the element internal force vector in the element coordinate system
        elFint = np.array([-Fs, 0., Fs, 0])

        # Rotate element fint to the global coordinate system
        elemdat.fint = toGlobalCoordinates(elFint, elemdat.coords)
