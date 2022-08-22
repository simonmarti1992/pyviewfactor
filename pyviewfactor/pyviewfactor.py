""" Main functions"""

import scipy.integrate 
import pandas as pd
import numpy as np 
import pyvista as pv
from numba import njit

def fc_unstruc2poly(mesh_unstruc):
    """Convenience conversion function from UnstructuredGrid to PolyData

    Parameters
    ----------
    * **mesh_unstruc** : *pyvista.UnstructuredGrid*
    
        > Unstructured Pyvista Grid.

    Returns
    -------
    * *pyvista.PolyData*
    
        > The same mesh converted to a surface pyvista.PolyData.

    Examples
    --------
    >>> import pyviewfactor as pvf
    >>> import pyvista as pv
    >>> sphere = pv.Sphere(radius=0.5, center=(0, 0, 0))
    >>> subset = sphere.extract_cells(10)
    >>> subsetPoly = fc_unstruc2poly(subset)
    >>> subsetPoly
    PolyData (0x1fdd9786040)
      N Cells:    1
      N Points:    3
      X Bounds:    -5.551e-17, 3.617e-02
      Y Bounds:    0.000e+00, 4.682e-02
      Z Bounds:    -5.000e-01, -4.971e-01
      N Arrays:    0

    """
    vertices = mesh_unstruc.points
    faces = mesh_unstruc.cells
    return pv.PolyData(vertices, faces)


def get_visibility(c1,c2):
    """Facets visibility:
    
    A test to check if two facets can "see" each other, taking the normals 
    into consideration (no obstruction tests, only normals orientations).

    Parameters
    ----------
    * **c1** : *pyvista.PolyData*
    
        > PolyData facet (pyvista format).
        
    * **c2** : *pyvista.PolyData*
    
        > PolyData facet (pyvista format).

    Returns
    -------
    * *bool*
    
        > True is the facets "see" each other, False else.
    
    Examples
    --------
    >>> import pyvista as pv
    >>> import pyviewfactor as pvf
    >>> tri1 = pv.Triangle([[0.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 0.0, 1.0]])
    >>> tri2 = pv.Triangle([[1.0, 0.0, 0.0],[1.0, 1.0, 0.0],[0.0, 1.0, 0.0]])
    >>> pvf.get_visibility(tri1, tri2)
    True

    """
    # les centres des cellules
    center1 = c1.cell_centers().points
    center2 = c2.cell_centers().points
    n21 = center1-center2
    # les normales
    norm1 = c1.cell_normals
    norm2 = c2.cell_normals
    # les produits scalaires
    pd_scal_sup = np.einsum('ij,ij->i', n21, norm2)
    pd_scal_inf = np.einsum('ij,ij->i', n21, norm1)
    # test de visibilite 
    vis=False
    if (pd_scal_sup > 0 and pd_scal_inf < 0):
        vis = True
    return vis

def get_visibility_raytrace(face1,face2,obstacle):
    """Raytrace between face1 and face2
    
    A test to check if there is an obstruction between two facets.

    Parameters
    ----------
    * **face1** : *pyvista.PolyData*
    
        > face1 to be checked for obstruction.
        
    * **face2** : *pyvista.PolyData*
    
        > face2 to be checked for obstruction.
        
    * **obstacle** : *pyvista.PolyData*
        
        > The mesh inbetween, composing the potential obstruction.

    Returns
    -------
    * *bool*
        
        > True if no obstruction, False else.
        
    Examples
    --------
    >>> import pyvista as pv
    >>> import pyviewfactor as pvf
    >>> tri1 = pv.Triangle([[0.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 0.0, 1.0]])
    >>> tri2 = pv.Triangle([[1.0, 0.0, 0.0],[1.0, 1.0, 0.0],[0.0, 1.0, 0.0]])
    >>> obstacle = pv.Circle(radius=3.0)
    >>> obstacle.translate([0, 0, 0.5], inplace = True)
    >>> pvf.get_visibility_raytrace(tri2, tri1, obstacle)
    False

    """
    # Define line segment
    start = face1.cell_centers().points[0]
    stop =  face2.cell_centers().points[0]
    # Perform ray trace
    points, ind = obstacle.ray_trace(start, stop, first_point=False)
    # if you work with a single cell
    if obstacle.n_cells==1:
        if ind.size==0:# if intersection: there is an obstruction
            return True 
        else:# if not, the faces can see each other
            return False
    # if face1 and face2 are contained in the obstacle mesh
    else:
        if len(ind)>3: # if intersection: there is an obstruction
            return False
        else: # if not, the faces can see each other
            return True

def trunc(values, decs=0):
    """Return values with *decs* decimals. 
    
    A function to truncate decimals in floats.

    Parameters
    ----------
    * **values** : *float*, or *numpy.array* (floats)
    
        A float value with decimals, or a numpy.array of floats
        
    * **decs** : *int*, optional
    
        The number of decimals to keep. The default is 0.

    Returns
    -------
    * *float*
    
        > The same flaot truncated with *decs* decimals, or a the same
        numpy.array of floats truncated.
    
    Examples
    --------
    >>> import pyvista as pv
    >>> import pyviewfactor as pvf
    >>> a = 1.23456789
    >>> pvf.trunc(a,2)
    1.23
    >>> tri1 = pv.Triangle([[0.111111, 1.111111, 1.111111],
                        [1.222222, 1.222222, 1.222222],
                        [1.333333, 0.333333, 1.333333]])
    >>> trunc(tri1.points,2)
    pyvista_ndarray([[0.11, 1.11, 1.11],
                     [1.22, 1.22, 1.22],
                     [1.33, 0.33, 1.33]])

    """
    return np.trunc(values*10**decs)/(10**decs)

@njit # numba's just in time compilation offers a x2 speedup...
def integrand(x , y, norm_q_carree, norm_p_carree, scal_qpq, \
              scal_qpp, scal_pq, norm_qp_carree):
    """
    Return the integrand for a pair of edges of two facets for the view factor
    computation.
    
    Used in the *compute_viewfactor* function. 

    """
    return np.log(y**2*norm_q_carree + x**2*norm_p_carree - 2*y*scal_qpq + \
                  2*x*scal_qpp - 2*x*y*scal_pq + norm_qp_carree)*scal_pq

def compute_viewfactor(cell_1,cell_2):
    """
    View factor computation between cell1 and cell2

    Parameters
    ----------
    * **cell_1** : *pyvista.PolyData* facet
	
        > The first cell.
		
    * **cell_2** : *pyvista.PolyData* facet
	
        > The second cell.

    Returns
    -------
    * *float*
	
        > The view factor from **cell_2** to **cell_1**.
    
    Examples
    --------
    >>> import pyvista as pv
    >>> import pyviewfactor as pvf
    >>> tri1 = pv.Triangle([[0.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 0.0, 1.0]])
    >>> tri2 = pv.Triangle([[1.0, 0.0, 0.0],[1.0, 1.0, 0.0],[0.0, 1.0, 0.0]])
    >>> pvf.compute_viewfactor(tri1, tri2)
    0.07665424316999997

    """
    FF = 0
    #travail sur la cellule 1
    cell_1_poly = cell_1
    cell_1 = cell_1.cast_to_unstructured_grid()
    cell_1_points = cell_1.cell_points(0)
    cell_1_points_roll = np.roll(cell_1_points, -1, axis=0)
    vect_dir_elts_1 = cell_1_points_roll - cell_1_points
    cell_1_points = trunc(cell_1_points, decs=4)
    #travail sur la cellule 2
    cell_2_poly = cell_2
    cell_2 = cell_2.cast_to_unstructured_grid()
    cell_2_points = cell_2.cell_points(0)
    cell_2_points_roll = np.roll(cell_2_points, -1, axis=0)
    vect_dir_elts_2 = cell_2_points_roll - cell_2_points
    cell_2_points = trunc(cell_2_points, decs=4)
    n_cols = np.shape(cell_2_points)[0]
    n_rows = np.shape(cell_1_points)[0]
    nb_sommets_1 = n_rows
    nb_sommets_2 = n_cols
    #calcul de la dataframe avec tous les vecteurs
    mat_vectors = np.zeros((n_rows,n_cols))
    vectors = pd.DataFrame(mat_vectors, dtype = object)
    for row in range(n_rows) :
        #on calcule les coordonnées des vecteurs partant du sommet i de la 
        # cellule i et allant vers les différents sommets j de la cellule j
        coord_repeat = np.tile(cell_1_points[row], (nb_sommets_2, 1))
        vect_sommets_1_to_2 = cell_2_points - coord_repeat
        # On transforme les matrices en liste de tuple pour obtenir les 
        # coordonnées de vecteurs sous forme de triple
        coord_vect = list(tuple(map(tuple, vect_sommets_1_to_2)))
        #On stocke ensuite les coord dans le DataFrame
        vectors.at[row] = pd.Series(coord_vect)
    vect_sommets_extra = vectors
    vect_sommets_intra_1 = vect_dir_elts_1
    vect_sommets_intra_2 = vect_dir_elts_2
    #calcul des constantes pour lintegrale 
    area = cell_2_poly.compute_cell_sizes(area = True)['Area']
    A_q = area[0]
    constante = 4*np.pi*A_q
    err = 0
    s_i = 0
    s_j = 0
    arr_test = np.argwhere((cell_2_points[:, None, :] == cell_1_points[:, :]).all(-1))
    nbre_sommets_partages = np.shape(arr_test)[0]
    if nbre_sommets_partages == 0:
            #dans ce cas il n'y a aucun sommet partage 
                for n in range(nb_sommets_2):
                    p_n_np1 = tuple(vect_sommets_intra_2[n,:])
                    norm_p_carree = np.dot(p_n_np1, p_n_np1)
                    for m in range(nb_sommets_1):
                        q_m_mp1 = tuple(vect_sommets_intra_1[m, :])
                        norm_q_carree = np.dot(q_m_mp1, q_m_mp1)
                        qm_pn = vect_sommets_extra.loc[m, n]
                        norm_qp_carree = np.dot(qm_pn, qm_pn)
                        scal_qpq = np.dot(qm_pn, q_m_mp1)
                        scal_qpp = np.dot(qm_pn, p_n_np1)
                        scal_pq = np.dot(q_m_mp1, p_n_np1)
                        s_j, err = scipy.integrate.dblquad(integrand, 0, 1, lambda x : 0, lambda x : 1, args = (norm_q_carree, norm_p_carree, scal_qpq, scal_qpp, scal_pq, norm_qp_carree,))
                        s_i += round(s_j/constante, 11)
                        err += err/(nb_sommets_1 + nb_sommets_2)
    else: 
    # dans ce cas les cellules ne partagent que 1 côté
    # on décide alors de les 'décoller'
        for sommet_j in cell_2_points[:, :]:
            # on les décale en leur appliquant le vecteur reliant les 
            # centroïdes des cellules
            # sommet_j += np.dot(normals[index[0]], 0.001)
            sommet_j += np.dot(cell_1_poly.face_normals[0], 0.001) # LE BON
        #On doit alors réécrire les vecteurs allant d'une cellule à l'autre
        for row in range(n_rows):
            # on calcule les coordonnées des vecteurs partant du sommet i de la
            # cellule i et allant vers les différents sommets j de la cellule j
            coord_repeat = np.tile(cell_1_points[row], (n_cols, 1))
            vect_sommets_i_to_j = cell_2_points - coord_repeat
            # On transforme les matrices en liste de tuple pour obtenir les 
            # coordonnées de vecteurs sous forme de triple
            coord_vect = list(tuple(map(tuple, vect_sommets_i_to_j)))
            #On stocke ensuite les coord dans le DataFrame
            vectors.at[row] = pd.Series(coord_vect)
        #puis on fait comme ci il n'y avait pas de sommet partagé !
        for n in range(nb_sommets_2):
            p_n_np1 = tuple(vect_sommets_intra_2[n, :])
            norm_p_carree = np.dot(p_n_np1, p_n_np1)
            for m in range(nb_sommets_1):
                q_m_mp1 = tuple(vect_sommets_intra_1[m, :])
                norm_q_carree = np.dot(q_m_mp1, q_m_mp1)
                qm_pn = vect_sommets_extra.loc[m, n]
                norm_qp_carree = np.dot(qm_pn, qm_pn)
                scal_qpq = np.dot(qm_pn, q_m_mp1)
                scal_qpp = np.dot(qm_pn, p_n_np1)
                scal_pq = np.dot(q_m_mp1, p_n_np1)
                s_j, err = scipy.integrate.dblquad(integrand, 0, 1, lambda x: 0, lambda x: 1, args=(norm_q_carree, norm_p_carree, scal_qpq, scal_qpp, scal_pq, norm_qp_carree,))
                s_i += round(s_j/constante, 11)
                err += err/(nb_sommets_1 + nb_sommets_2)
    if s_i > 0:
        FF = s_i
    return FF
#~~~~~~~~~~~~~~~~~~~~~~~~~~ End Of File ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
