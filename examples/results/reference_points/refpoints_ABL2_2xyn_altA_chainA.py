from pymol import *
import os
from pymol.cgo import *

cmd.load("/home/dominique/Documents/data/kinsim/20190724_full/raw/KLIFS_download/HUMAN/ABL2/2xyn_altA_chainA/pocket.mol2", "protein")

cmd.show("cartoon", "protein")
cmd.hide("lines", "protein")
cmd.color("gray", "protein")

cmd.set("cartoon_transparency", 0.5, "protein")
cmd.set("opaque_background", "off")

cmd.set_color("hinge_color", [1.0, 0.0, 1.0])
cmd.set_color("dfg_color", [0.25, 0.41, 0.88])
cmd.color("hinge_color", "protein and resi 362+363+364")
cmd.color("dfg_color", "protein and resi 427+428+429")

obj_refpoints_ABL2_2xyn_altA_chainA = [

	COLOR, 1.0, 0.65, 0.0,
	SPHERE, 0.615132186544339, 20.76112324159022, 36.753074235474074, 1,
	COLOR, 1.0, 0.0, 1.0,
	SPHERE, 1.8207666666666666, 22.0525, 41.8871, 1,
	SPHERE, 7.781, 21.5903, 44.3638, 0.5,
	SPHERE, -0.6448, 23.8535, 46.1493, 0.5,
	SPHERE, -1.6739, 20.7137, 35.1482, 0.5,
	COLOR, 0.25, 0.41, 0.88,
	SPHERE, 8.180033333333332, 19.509033333333335, 34.7477, 1,
	SPHERE, 14.9378, 16.8877, 38.9983, 0.5,
	SPHERE, 8.2016, 22.7536, 31.2782, 0.5,
	SPHERE, 1.4007, 18.8858, 33.9666, 0.5,
	COLOR, 0.0, 1.0, 0.0,
	SPHERE, -0.17346666666666652, 15.1489, 41.06463333333333, 1,
	SPHERE, 6.4446, 8.2389, 42.8178, 0.5,
	SPHERE, -3.8262, 21.8791, 45.4292, 0.5,
	SPHERE, -3.1388, 15.3287, 34.9469, 0.5,
]
cmd.load_cgo(obj_refpoints_ABL2_2xyn_altA_chainA, "refpoints_ABL2_2xyn_altA_chainA")