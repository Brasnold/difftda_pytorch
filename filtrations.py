import gudhi as gd
import numpy as np


def rips_percistence_pairs(DX, dim=1, max_edge_length=12):
    rc = gd.RipsComplex(distance_matrix=DX, max_edge_length=max_edge_length)

    # for H_1 it's sufficent to consider the 2-skeleton of the filtration
    st = rc.create_simplex_tree(max_dimension=2)

    # gudhi requets it before calling persistence_pairs
    st.persistence()

    pairs = st.persistence_pairs()

    start_indices = []
    end_indices = []


    for s1, s2 in pairs:
        if len(s1) == (dim + 1) and len(s2) > 0:  # only elements of H_dim (holes for dim=1)
            assert (len(s2) == (dim + 2))

            l1, l2 = np.array(s1), np.array(s2)

            # NOTE:
            # np.argmax gives the flattened index of the first occurency of the maximum distance
            # so it determinate a default pre-order among vertex based on their index
            # that come into play for the subgradient of the max function defyning the filtrations
            #
            # For dim = 1 there is only a pair in s1 so it does an useless computation.

            i1 = [s1[v] for v in np.unravel_index(np.argmax(DX[l1, :][:, l1]), [len(s1), len(s1)])]
            i2 = [s2[v] for v in np.unravel_index(np.argmax(DX[l2, :][:, l2]), [len(s2), len(s2)])]

            start_indices.append(i1)
            end_indices.append(i2)

    return np.array(start_indices), np.array(end_indices)


def cubical_persistence_pairs(x):
    cc = gd.CubicalComplex(dimensions=x.shape, top_dimensional_cells=x.flatten())
    cc.persistence()

    cof = cc.cofaces_of_persistence_pairs()[0][0]

    start_indices = [np.unravel_index(cof[idx, 0], x.shape) for idx in range(len(cof))]
    end_indices = [np.unravel_index(cof[idx, 1], x.shape) for idx in range(len(cof))]

    return np.array(start_indices), np.array(end_indices)