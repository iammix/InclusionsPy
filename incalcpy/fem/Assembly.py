from numpy import zeros, ones, ix_, append, repeat, array
from scipy.sparse import coo_matrix
from util.dataStructures import Properties
from util.dataStructures import elementData

def assembleArray(props, globdat, rank, action):


    # A = zeros( len(globdat.dofs) * ones(2,dtype=int) )
    B = zeros(len(globdat.dofs) * ones(1, dtype=int))

    val = array([], dtype=float)
    row = array([], dtype=int)
    col = array([], dtype=int)

    nDof = len(globdat.dofs)

    if action != 'commit':
        globdat.resetNodalOutput()

    # Loop over the element groups
    for elementGroup in globdat.elements.iterGroupNames():

        # Get the properties corresponding to the elementGroup
        el_props = getattr(props, elementGroup)

        # Loop over the elements in the elementGroup
        for iElm, element in enumerate(globdat.elements.iterElementGroup(elementGroup)):

            # Get the element nodes
            el_nodes = element.getNodes()

            # Get the element coordinates
            el_coords = globdat.nodes.getNodeCoords(el_nodes)

            # Get the element degrees of freedom
            el_dofs = globdat.dofs.get(el_nodes)

            # Get the element state
            el_a = globdat.state[el_dofs]
            el_Da = globdat.Dstate[el_dofs]

            # Create the an element state to pass through to the element
            # el_state = Properties( { 'state' : el_a, 'Dstate' : el_Da } )
            elemdat = elementData(el_a, el_Da)

            elemdat.coords = el_coords
            elemdat.nodes = el_nodes
            elemdat.props = el_props
            elemdat.iElm = iElm

            if hasattr(element, "matProps"):
                elemdat.matprops = element.matProps

            if hasattr(element, "mat"):
                element.mat.reset()

            # Get the element contribution by calling the specified action
            getattr(element, action)(elemdat)

            for label in elemdat.outlabel:
                element.appendNodalOutput(label, globdat, elemdat.outdata)

            # Assemble in the global array
            if rank == 1:
                B[el_dofs] += elemdat.fint
            elif rank == 2 and action is "getTangentStiffness":
                # A[ix_(el_dofs,el_dofs)] += elemdat.stiff

                row = append(row, repeat(el_dofs, len(el_dofs)))

                for i in range(len(el_dofs)):
                    col = append(col, el_dofs)

                val = append(val, elemdat.stiff.reshape(len(el_dofs) * len(el_dofs)))

                #B[el_dofs] += elemdat.fint
            elif rank == 2 and action is "getMassMatrix":

                row = append(row, repeat(el_dofs, len(el_dofs)))

                for i in range(len(el_dofs)):
                    col = append(col, el_dofs)

                val = append(val, elemdat.mass.reshape(len(el_dofs) * len(el_dofs)))

                B[el_dofs] += elemdat.lumped
    #    else:
    #      raise NotImplementedError('assemleArray is only implemented for vectors and matrices.')

    if rank == 1:
        return B
    elif rank == 2:
        return coo_matrix((val, (row, col)), shape=(nDof, nDof)), B


##########################################
# Internal force vector assembly routine #
##########################################

def assembleInternalForce(props, globdat):
    return assembleArray(props, globdat, rank=1, action='getInternalForce')


##########################################
# External force vector assembly routine #
##########################################

def assembleExternalForce(props, globdat):
    return globdat.fhat + assembleArray(props, globdat, rank=1, action='getExternalForce')


#############################################
# Tangent stiffness matrix assembly routine #
#############################################

def assembleTangentStiffness(props, globdat):
    return assembleArray(props, globdat, rank=2, action='getTangentStiffness')


#############################################
# Mass matrix assembly routine              #
#############################################

def assembleMassMatrix(props, globdat):
    return assembleArray(props, globdat, rank=2, action='getMassMatrix')


def commit(props, globdat):
    return assembleArray(props, globdat, rank=0, action='commit')


def getAllConstraints(props, globdat):
    # Loop over the element groups
    for elementGroup in globdat.elements.iterGroupNames():

        # Get the properties corresponding to the elementGroup
        el_props = getattr(props, elementGroup)

        # Loop over the elements in the elementGroup
        for element in globdat.elements.iterElementGroup(elementGroup):
            # Get the element nodes
            el_nodes = element.getNodes()

            elemdat.nodes = el_nodes
            elemdat.props = el_props

            # Get the element contribution by calling the specified action
            getattr(element, 'getConstraints', None)(elemdat)
