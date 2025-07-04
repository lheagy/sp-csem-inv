from scipy.spatial import cKDTree


def match_values_nearest(e3d_values, edges):
    """
    Find the nearest edges to ensure we order the jsp or pore-pressure gradient
    values in the manner that discretize.TreeMesh is expecting

    Parameters
    ----------
    e3d_values : (*, 4) numpy.ndarray
        An array containing [x, y, z, j] where j is just a single component (e.g. jx, jy, or jz)

    edges : (*, 3) numpy.ndarray
        Array of the discretize mesh edge locations [x, y, z] for the appropriate edge grid (e.g. mesh.edges_x for jx or mesh.edges_y for jy)

    Returns
    -------

    numpy.ndarray
        Array of the j values in the order expected by the discretize mesh.

    """

    # separate coordinates and values
    j_coords = e3d_values[:, :3]
    j_values = e3d_values[:, 3]

    # build a KDTree on J_X coordinates
    tree = cKDTree(j_coords)

    # query the nearest neighbor for each edge point
    _, indices = tree.query(edges)

    # get the values from J_X corresponding to nearest neighbors
    matched_values = j_values[indices]

    return matched_values
