from pymol import *
import os
from pymol.cgo import *

cmd.load("/home/dominique/Documents/data/kinsim/20190724_full/raw/KLIFS_download/HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2", "protein")

cmd.show("cartoon", "protein")
cmd.hide("lines", "protein")
cmd.color("gray", "protein")

cmd.set("cartoon_transparency", 0.5, "protein")
cmd.set("opaque_background", "off")

cmd.set_color("hinge_color", [1.0, 0.0, 1.0])
cmd.set_color("dfg_color", [0.25, 0.41, 0.88])
cmd.color("hinge_color", "protein and resi 127+128+129")
cmd.color("dfg_color", "protein and resi 194+195+196")

obj_refpoints_AAK1_4wsq_altA_chainB = [

	COLOR, 1.0, 0.65, 0.0,
	SPHERE, 1.2933751497005992, 20.792858233532986, 36.40460965568866, 1,
	COLOR, 1.0, 0.0, 1.0,
	SPHERE, 1.9762000000000002, 21.758433333333333, 41.80473333333333, 1,
	SPHERE, 7.6657, 21.2202, 43.7273, 0.5,
	SPHERE, -0.306, 23.3326, 46.1435, 0.5,
	SPHERE, -1.4311, 20.7225, 35.5434, 0.5,
	COLOR, 0.25, 0.41, 0.88,
	SPHERE, 7.4102, 20.057466666666667, 33.52073333333333, 1,
	SPHERE, 13.3361, 16.395, 37.1039, 0.5,
	SPHERE, 7.3139, 24.4201, 29.9088, 0.5,
	SPHERE, 1.5806, 19.3573, 33.5495, 0.5,
	COLOR, 0.0, 1.0, 0.0,
	SPHERE, -0.7358333333333333, 15.8926, 39.50346666666667, 1,
	SPHERE, 5.0183, 9.9363, 37.8002, 0.5,
	SPHERE, -3.7619, 21.9317, 45.3376, 0.5,
	SPHERE, -3.4639, 15.8098, 35.3726, 0.5,
]
cmd.load_cgo(obj_refpoints_AAK1_4wsq_altA_chainB, "refpoints_AAK1_4wsq_altA_chainB")