from pymol import *
import os
from pymol.cgo import *

cmd.load("/home/dominique/Documents/data/kinsim/20190724_full/raw/KLIFS_download/HUMAN/EGFR/3w32_chainA/pocket.mol2", "protein")

cmd.show("cartoon", "protein")
cmd.hide("lines", "protein")
cmd.color("gray", "protein")

cmd.set("cartoon_transparency", 0.5, "protein")
cmd.set("opaque_background", "off")

cmd.set_color("hinge_color", [1.0, 0.0, 1.0])
cmd.set_color("dfg_color", [0.25, 0.41, 0.88])
cmd.color("hinge_color", "protein and resi 791+792+793")
cmd.color("dfg_color", "protein and resi 855+856+857")

obj_refpoints_EGFR_3w32_chainA = [

	COLOR, 1.0, 0.65, 0.0,
	SPHERE, 1.1640752932551355, 20.801586730205273, 36.20067683284458, 1,
	COLOR, 1.0, 0.0, 1.0,
	SPHERE, 2.0429, 21.301233333333332, 41.536, 1,
	SPHERE, 7.7156, 20.2023, 43.4006, 0.5,
	SPHERE, 0.1417, 22.8441, 46.1515, 0.5,
	SPHERE, -1.7286, 20.8573, 35.0559, 0.5,
	COLOR, 0.25, 0.41, 0.88,
	SPHERE, 7.6964999999999995, 20.543633333333332, 32.303333333333335, 1,
	SPHERE, 12.1579, 15.7003, 35.9584, 0.5,
	SPHERE, 9.5766, 26.8005, 27.5058, 0.5,
	SPHERE, 1.355, 19.1301, 33.4458, 0.5,
	COLOR, 0.0, 1.0, 0.0,
	SPHERE, -0.6187333333333336, 15.5668, 39.08783333333333, 1,
	SPHERE, 5.0947, 9.2101, 36.3726, 0.5,
	SPHERE, -3.41, 21.6235, 45.5136, 0.5,
	SPHERE, -3.5409, 15.8668, 35.3773, 0.5,
]
cmd.load_cgo(obj_refpoints_EGFR_3w32_chainA, "refpoints_EGFR_3w32_chainA")