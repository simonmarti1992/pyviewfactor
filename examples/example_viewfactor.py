# -*- coding: utf-8 -*-

import pyvista as pv
from pyviewfactor import get_visibility, compute_viewfactor


pointa = [1, 0, 0]
pointb = [1, 1, 0]
pointc = [0, 1, 0]
pointd = [0, 0, 0]
rectangle1 = pv.Rectangle([pointa, pointb, pointc, pointd])

pointa = [1, 0, 1]
pointb = [1, 1, 1]
pointc = [0, 1, 1]
pointd = [0, 0, 1]
liste_pts = [pointa, pointb, pointc, pointd]
liste_pts.reverse()
rectangle2 = pv.Rectangle(liste_pts)


if get_visibility(rectangle1 , rectangle2):
    F=compute_viewfactor(rectangle1, rectangle2)
    print("VF = ", F)
else:
    print("Not facing each other")
    
