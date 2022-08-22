# -*- coding: utf-8 -*-
import pyvista as pv
import numpy as np
from pyviewfactor import  compute_viewfactor
from tqdm import tqdm # fancy progress bar

def fc_unstruc2poly(mesh_unstruc):
    """
    A function to convert unstructuredgrid to polydata.
    :param mesh_unstruc: unstructured pyvista grid
    :type mesh_unstruc: pv.UnstructuredGrid
    :return: pv.PolyData
    :rtype: pv.PolyData

    """
    vertices = mesh_unstruc.points
    faces = mesh_unstruc.cells
    return pv.PolyData(vertices, faces)

# create a raw sphere with pyvista
sphere = pv.Sphere(radius=50, center=(0, 0, 0), direction=(0, 0, 1),
                 theta_resolution=9, phi_resolution=9,
                 start_theta=0, end_theta=360, 
                 start_phi=0, end_phi=180)
# and put the normals inwards please
sphere.flip_normals()

# let us chose a cell to compute view factors to
chosen_face = sphere.extract_cells(10)
# convert to PolyData
chosen_face = fc_unstruc2poly(chosen_face)
# "one array to contain them all"
F = np.zeros(sphere.n_cells)

# now let us compute the view factor to all other faces
# (with a fancy progress bar -> tqdm)
for i in tqdm(range(sphere.n_cells), total=sphere.n_cells):
    face = sphere.extract_cells(i) # other facet
    face = fc_unstruc2poly(face) # convert to PolyData
    F[i] = compute_viewfactor(face, chosen_face) # compute VF

print("Complementarity check: \n(is \sum_{i=0}^n F_i =? 1)", np.sum(F))

# put the scalar values in the geometry
sphere.cell_data["F"] = F
sphere.save("./sphere.vtk") # ... and save.