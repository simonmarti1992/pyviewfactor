# -*- coding: utf-8 -*-
import numpy as np
import pyvista as pv
from tqdm import tqdm
import pyviewfactor as pvf

# read the geometry
mesh = pv.read("./built_envmt.vtk")
meshpoly = pvf.fc_unstruc2poly(mesh) # convert to polydata for obstruction check

# identify who is who
i_wall = np.where(mesh["wall_names"]=='wall')[0]
i_sky = np.where(mesh["wall_names"]=='sky')[0]
i_building1 = np.where(mesh["wall_names"]=='building1')[0]
i_building2 = np.where(mesh["wall_names"]=='building2')[0]

# get the different elements
wall = mesh.extract_cells(i_wall)
sky = mesh.extract_cells(i_sky)
building1 = mesh.extract_cells(i_building1)
building2 = mesh.extract_cells(i_building2)

# convert to polydata
wall = pvf.fc_unstruc2poly(wall)

Fsky=0
# for all cells constituting the ensemble
for patch in tqdm(i_sky):
    sky= mesh.extract_cells(patch) # extract one cell
    sky = pvf.fc_unstruc2poly(sky) # convert to polydata
    if pvf.get_visibility(sky, wall): # if the can see each other...
        if pvf.get_visibility_raytrace(sky, wall, meshpoly): # ... if no obstruction
            Fsky += pvf.compute_viewfactor(sky,wall) # compute and increment view factor : F_i->(j+k) = F_i->j + F_i->k

# same for building 1
Fbuilding1 = 0
for patch in tqdm(i_building1):
    bldng1 = mesh.extract_cells(patch)
    bldng1 = pvf.fc_unstruc2poly(bldng1)
    if pvf.get_visibility(bldng1, wall):
        if pvf.get_visibility_raytrace(bldng1, wall, meshpoly):
            Fbuilding1 += pvf.compute_viewfactor(bldng1,wall)

# same for building 2
Fbuilding2 = 0
for patch in tqdm(i_building2):
    bldng2 = mesh.extract_cells(patch)
    bldng2 = pvf.fc_unstruc2poly(bldng2)
    if pvf.get_visibility(bldng2, wall):
        if pvf.get_visibility_raytrace(bldng2, wall, meshpoly):
            Fbuilding2 += pvf.compute_viewfactor(bldng2,wall)

# complementarity implies \sigma F_i = 1 : compute viewfactor to ground
Fground = 1-Fbuilding1-Fbuilding2-Fsky

print('\n----------------------')
print('Wall to environment view factors :')
print('\tSky ', round(Fsky,4))
print('\tBuilding 1 ', round(Fbuilding1, 4))
print('\tBuilding 2 ', round(Fbuilding2, 4))
print('Ground view factor :')
print('\tGround ', round(Fground, 4))
