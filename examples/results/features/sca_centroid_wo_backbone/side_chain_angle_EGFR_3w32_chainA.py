from pymol import *
import os
from pymol.cgo import *

cmd.fetch("3w32")
cmd.remove("solvent")
cmd.remove("3w32 and not chain A")
cmd.remove("all and not (resi 716+717+718+719+720+721+722+723+724+725+726+727+728+742+743+744+745+746+747+758+759+760+761+762+763+764+765+766+767+768+769+770+772+773+774+775+776+777+778+779+780+787+788+789+790+791+792+793+794+795+796+797+798+799+800+801+802+803+804+827+828+829+830+831+832+833+834+835+836+837+838+839+840+841+842+843+844+845+853+854+855+856+857+858+859)")

cmd.pseudoatom(object="angle_716", pos=[15.017999649047852, 46.242000579833984, 16.0939998626709], label=110.2)
obj_716 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 15.428999900817871, 46.391998291015625, 14.621000289916992, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 15.017999649047852, 46.242000579833984, 16.0939998626709, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 14.588599967956544, 48.4422004699707, 17.080199813842775, 0.2,
]
cmd.load_cgo(obj_716, "716")

cmd.pseudoatom(object="angle_717", pos=[20.10700035095215, 45.00899887084961, 14.449000358581543], label=127.8)
obj_717 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 18.67799949645996, 44.44200134277344, 14.362000465393066, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 20.10700035095215, 45.00899887084961, 14.449000358581543, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 20.523000081380207, 44.984999338785805, 14.047333399454752, 0.2,
]
cmd.load_cgo(obj_717, "717")

cmd.pseudoatom(object="angle_718", pos=[17.937999725341797, 39.849998474121094, 15.720999717712402], label=119.0)
obj_718 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 18.797000885009766, 41.07400131225586, 16.066999435424805, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 17.937999725341797, 39.849998474121094, 15.720999717712402, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 16.475249767303467, 39.90749931335449, 15.957000255584717, 0.2,
]
cmd.load_cgo(obj_718, "718")

obj_719 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 22.38800048828125, 39.99100112915039, 15.276000022888184, 0.2,
]
cmd.load_cgo(obj_719, "719")

cmd.pseudoatom(object="angle_720", pos=[26.257999420166016, 39.736000061035156, 12.494999885559082], label=111.2)
obj_720 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 25.090999603271484, 38.8129997253418, 12.855999946594238, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 26.257999420166016, 39.736000061035156, 12.494999885559082, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 26.032499313354492, 40.309499740600586, 12.145999908447266, 0.2,
]
cmd.load_cgo(obj_720, "720")

obj_721 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 26.597999572753906, 35.39899826049805, 12.11299991607666, 0.2,
]
cmd.load_cgo(obj_721, "721")

cmd.pseudoatom(object="angle_722", pos=[29.472000122070312, 32.207000732421875, 9.869999885559082], label=180.0)
obj_722 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 28.982999801635742, 33.64400100708008, 9.715999603271484, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 29.472000122070312, 32.207000732421875, 9.869999885559082, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 29.472000122070312, 32.207000732421875, 9.869999885559082, 0.2,
]
cmd.load_cgo(obj_722, "722")

cmd.pseudoatom(object="angle_723", pos=[25.47800064086914, 32.207000732421875, 6.7729997634887695], label=113.6)
obj_723 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 26.36400032043457, 33.45100021362305, 6.959000110626221, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 25.47800064086914, 32.207000732421875, 6.7729997634887695, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 26.7440002986363, 30.058000292096818, 6.841857160840716, 0.2,
]
cmd.load_cgo(obj_723, "723")

obj_724 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 24.59000015258789, 36.78799819946289, 7.468999862670898, 0.2,
]
cmd.load_cgo(obj_724, "724")

cmd.pseudoatom(object="angle_725", pos=[23.05299949645996, 40.762001037597656, 9.319999694824219], label=126.4)
obj_725 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 22.60300064086914, 39.310001373291016, 9.541999816894531, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 23.05299949645996, 40.762001037597656, 9.319999694824219, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 23.335999806722004, 41.090667724609375, 9.69266668955485, 0.2,
]
cmd.load_cgo(obj_725, "725")

cmd.pseudoatom(object="angle_726", pos=[18.7450008392334, 37.67900085449219, 11.833000183105469], label=126.4)
obj_726 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 19.06999969482422, 38.979000091552734, 11.050999641418457, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 18.7450008392334, 37.67900085449219, 11.833000183105469, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 18.458667119344074, 37.20199966430664, 11.62166690826416, 0.2,
]
cmd.load_cgo(obj_726, "726")

cmd.pseudoatom(object="angle_727", pos=[16.246000289916992, 42.779998779296875, 10.67199993133545], label=114.5)
obj_727 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 16.43899917602539, 41.698001861572266, 11.760000228881836, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 16.246000289916992, 42.779998779296875, 10.67199993133545, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 18.701375007629395, 43.78774929046631, 9.536625146865845, 0.2,
]
cmd.load_cgo(obj_727, "727")

cmd.pseudoatom(object="angle_728", pos=[12.734999656677246, 42.292999267578125, 14.875], label=147.8)
obj_728 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 13.045999526977539, 41.83599853515625, 13.45199966430664, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 12.734999656677246, 42.292999267578125, 14.875, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 11.330399703979491, 43.306399536132815, 16.25199966430664, 0.2,
]
cmd.load_cgo(obj_728, "728")

cmd.pseudoatom(object="angle_742", pos=[8.687000274658203, 38.5989990234375, 8.982000350952148], label=130.3)
obj_742 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 8.78600025177002, 38.35599899291992, 10.513999938964844, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 8.687000274658203, 38.5989990234375, 8.982000350952148, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 8.321333408355713, 38.92866643269857, 8.678000132242838, 0.2,
]
cmd.load_cgo(obj_742, "742")

cmd.pseudoatom(object="angle_743", pos=[13.145000457763672, 36.520999908447266, 11.607000350952148], label=180.0)
obj_743 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 12.503999710083008, 37.73500061035156, 10.909000396728516, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 13.145000457763672, 36.520999908447266, 11.607000350952148, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 13.145000457763672, 36.520999908447266, 11.607000350952148, 0.2,
]
cmd.load_cgo(obj_743, "743")

cmd.pseudoatom(object="angle_744", pos=[14.187000274658203, 40.45100021362305, 7.520999908447266], label=141.5)
obj_744 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 14.482000350952148, 38.98899841308594, 7.940999984741211, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 14.187000274658203, 40.45100021362305, 7.520999908447266, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 13.483000040054321, 41.00800037384033, 7.147249937057495, 0.2,
]
cmd.load_cgo(obj_744, "744")

cmd.pseudoatom(object="angle_745", pos=[18.29400062561035, 36.0629997253418, 7.318999767303467], label=137.6)
obj_745 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 17.972999572753906, 37.573001861572266, 7.138999938964844, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 18.29400062561035, 36.0629997253418, 7.318999767303467, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 20.258399963378906, 34.63600082397461, 8.084599876403809, 0.2,
]
cmd.load_cgo(obj_745, "745")

cmd.pseudoatom(object="angle_746", pos=[20.301000595092773, 41.152000427246094, 4.901000022888184], label=143.2)
obj_746 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 20.142000198364258, 39.63800048828125, 4.745999813079834, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 20.301000595092773, 41.152000427246094, 4.901000022888184, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 20.344800567626955, 42.841799926757815, 3.897000026702881, 0.2,
]
cmd.load_cgo(obj_746, "746")

cmd.pseudoatom(object="angle_747", pos=[22.836999893188477, 36.347999572753906, 2.436000108718872], label=123.8)
obj_747 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 23.02400016784668, 37.487998962402344, 3.450000047683716, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 22.836999893188477, 36.347999572753906, 2.436000108718872, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 21.874000072479248, 35.24324989318848, 2.610750049352646, 0.2,
]
cmd.load_cgo(obj_747, "747")

cmd.pseudoatom(object="angle_758", pos=[21.32699966430664, 29.142000198364258, -9.166000366210938], label=146.7)
obj_758 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 19.81599998474121, 29.18400001525879, -8.920999526977539, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 21.32699966430664, 29.142000198364258, -9.166000366210938, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 23.009800338745116, 28.11339988708496, -8.967799949645997, 0.2,
]
cmd.load_cgo(obj_758, "758")

cmd.pseudoatom(object="angle_759", pos=[19.533000946044922, 32.55400085449219, -5.39300012588501], label=140.8)
obj_759 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 19.233999252319336, 31.052000045776367, -5.622000217437744, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 19.533000946044922, 32.55400085449219, -5.39300012588501, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 20.21725034713745, 33.15025043487549, -4.986500024795532, 0.2,
]
cmd.load_cgo(obj_759, "759")

cmd.pseudoatom(object="angle_760", pos=[14.701000213623047, 31.506000518798828, -7.381999969482422], label=119.2)
obj_760 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 15.47700023651123, 30.871000289916992, -6.229000091552734, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 14.701000213623047, 31.506000518798828, -7.381999969482422, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 14.341000080108643, 32.92224979400635, -7.3064998388290405, 0.2,
]
cmd.load_cgo(obj_760, "760")

cmd.pseudoatom(object="angle_761", pos=[16.625, 26.500999450683594, -8.020000457763672], label=113.5)
obj_761 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 15.788000106811523, 27.125999450683594, -6.888999938964844, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 16.625, 26.500999450683594, -8.020000457763672, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 16.47350025177002, 25.058499813079834, -8.124500274658203, 0.2,
]
cmd.load_cgo(obj_761, "761")

cmd.pseudoatom(object="angle_762", pos=[19.02899932861328, 26.774999618530273, -3.25600004196167], label=139.6)
obj_762 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 17.559999465942383, 26.437999725341797, -3.5759999752044678, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 19.02899932861328, 26.774999618530273, -3.25600004196167, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 20.860999298095702, 26.15540008544922, -3.720799970626831, 0.2,
]
cmd.load_cgo(obj_762, "762")

cmd.pseudoatom(object="angle_763", pos=[15.0649995803833, 30.253000259399414, -1.8869999647140503], label=180.0)
obj_763 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 15.258000373840332, 28.767000198364258, -1.6670000553131104, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 15.0649995803833, 30.253000259399414, -1.8869999647140503, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 15.0649995803833, 30.253000259399414, -1.8869999647140503, 0.2,
]
cmd.load_cgo(obj_763, "763")

cmd.pseudoatom(object="angle_764", pos=[11.968000411987305, 26.408000946044922, -4.480999946594238], label=112.8)
obj_764 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 12.229000091552734, 26.863000869750977, -3.0439999103546143, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 11.968000411987305, 26.408000946044922, -4.480999946594238, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 9.393499910831451, 25.079750299453735, -4.796124994754791, 0.2,
]
cmd.load_cgo(obj_764, "764")

cmd.pseudoatom(object="angle_765", pos=[14.66100025177002, 22.98200035095215, -1.2489999532699585], label=128.7)
obj_765 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 13.338000297546387, 23.790000915527344, -1.1059999465942383, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 14.66100025177002, 22.98200035095215, -1.2489999532699585, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 14.735666592915853, 22.441333770751953, -1.4393332997957866, 0.2,
]
cmd.load_cgo(obj_765, "765")

cmd.pseudoatom(object="angle_766", pos=[14.560999870300293, 27.0939998626709, 2.308000087738037], label=138.9)
obj_766 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 13.557999610900879, 25.937999725341797, 2.055000066757202, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 14.560999870300293, 27.0939998626709, 2.308000087738037, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 16.562000513076782, 27.472999572753906, 2.1030000150203705, 0.2,
]
cmd.load_cgo(obj_766, "766")

cmd.pseudoatom(object="angle_767", pos=[9.586999893188477, 28.141000747680664, 0.38499999046325684], label=180.0)
obj_767 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 9.954999923706055, 27.14699935913086, 1.4609999656677246, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 9.586999893188477, 28.141000747680664, 0.38499999046325684, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 9.586999893188477, 28.141000747680664, 0.38499999046325684, 0.2,
]
cmd.load_cgo(obj_767, "767")

cmd.pseudoatom(object="angle_768", pos=[9.3149995803833, 22.841999053955078, -0.4779999852180481], label=109.1)
obj_768 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 8.673999786376953, 23.649999618530273, 0.6660000085830688, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 9.3149995803833, 22.841999053955078, -0.4779999852180481, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 9.825999736785889, 22.42449951171875, -0.2109999917447567, 0.2,
]
cmd.load_cgo(obj_768, "768")

cmd.pseudoatom(object="angle_769", pos=[10.607999801635742, 22.267000198364258, 4.96999979019165], label=127.3)
obj_769 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 9.345999717712402, 22.117000579833984, 4.111999988555908, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 10.607999801635742, 22.267000198364258, 4.96999979019165, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 11.056333223978678, 22.625, 4.877000013987224, 0.2,
]
cmd.load_cgo(obj_769, "769")

cmd.pseudoatom(object="angle_770", pos=[5.125999927520752, 20.70599937438965, 5.119999885559082], label=113.0)
obj_770 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 6.2789998054504395, 21.035999298095703, 6.0879998207092285, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 5.125999927520752, 20.70599937438965, 5.119999885559082, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 3.827750027179718, 20.74250030517578, 5.753250002861023, 0.2,
]
cmd.load_cgo(obj_770, "770")

cmd.pseudoatom(object="angle_772", pos=[5.539999961853027, 19.773000717163086, 14.045000076293945], label=85.6)
obj_772 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 6.236999988555908, 20.746000289916992, 13.09000015258789, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 5.539999961853027, 19.773000717163086, 14.045000076293945, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 5.3836666742960615, 19.066666920979817, 13.060333251953125, 0.2,
]
cmd.load_cgo(obj_772, "772")

cmd.pseudoatom(object="angle_773", pos=[10.50100040435791, 19.683000564575195, 14.866000175476074], label=111.2)
obj_773 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 9.92300033569336, 20.889999389648438, 14.095000267028809, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 10.50100040435791, 19.683000564575195, 14.866000175476074, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 8.965166886647543, 18.762000401814777, 16.18999989827474, 0.2,
]
cmd.load_cgo(obj_773, "773")

cmd.pseudoatom(object="angle_774", pos=[11.562000274658203, 21.857999801635742, 9.756999969482422], label=129.1)
obj_774 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 11.291999816894531, 22.71299934387207, 11.029000282287598, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 11.562000274658203, 21.857999801635742, 9.756999969482422, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 12.065333366394043, 21.61566670735677, 9.573333422342936, 0.2,
]
cmd.load_cgo(obj_774, "774")

cmd.pseudoatom(object="angle_775", pos=[11.836000442504883, 27.503999710083008, 10.706999778747559], label=112.8)
obj_775 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 10.866999626159668, 26.410999298095703, 10.246000289916992, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 11.836000442504883, 27.503999710083008, 10.706999778747559, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 11.808500289916992, 27.642000198364258, 11.594500064849854, 0.2,
]
cmd.load_cgo(obj_775, "775")

cmd.pseudoatom(object="angle_776", pos=[8.682000160217285, 26.631999969482422, 6.125999927520752], label=127.1)
obj_776 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 9.925999641418457, 27.32200050354004, 6.703999996185303, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 8.682000160217285, 26.631999969482422, 6.125999927520752, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 6.37671422958374, 27.9608576638358, 4.512571402958462, 0.2,
]
cmd.load_cgo(obj_776, "776")

cmd.pseudoatom(object="angle_777", pos=[11.873000144958496, 30.007999420166016, 3.2839999198913574], label=124.4)
obj_777 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 10.83899974822998, 30.155000686645508, 4.396999835968018, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 11.873000144958496, 30.007999420166016, 3.2839999198913574, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 12.84000015258789, 31.007500171661377, 2.941499948501587, 0.2,
]
cmd.load_cgo(obj_777, "777")

cmd.pseudoatom(object="angle_778", pos=[7.250999927520752, 33.354000091552734, 4.705999851226807], label=126.9)
obj_778 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 7.849999904632568, 32.40700149536133, 3.6710000038146973, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 7.250999927520752, 33.354000091552734, 4.705999851226807, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 6.47475004196167, 32.98675012588501, 5.882499933242798, 0.2,
]
cmd.load_cgo(obj_778, "778")

obj_779 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 9.482999801635742, 34.70600128173828, 1.1469999551773071, 0.2,
]
cmd.load_cgo(obj_779, "779")

cmd.pseudoatom(object="angle_780", pos=[13.270999908447266, 34.93299865722656, -1.3049999475479126], label=145.9)
obj_780 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 12.548999786376953, 36.077999114990234, -0.5370000004768372, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 13.270999908447266, 34.93299865722656, -1.3049999475479126, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 14.033999919891357, 34.6899995803833, -1.8514999747276306, 0.2,
]
cmd.load_cgo(obj_780, "780")

cmd.pseudoatom(object="angle_787", pos=[16.259000778198242, 40.04899978637695, 2.058000087738037], label=146.1)
obj_787 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 16.944000244140625, 38.69499969482422, 1.8830000162124634, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 16.259000778198242, 40.04899978637695, 2.058000087738037, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 16.50620002746582, 41.975599670410155, 2.3926000118255617, 0.2,
]
cmd.load_cgo(obj_787, "787")

cmd.pseudoatom(object="angle_788", pos=[15.88700008392334, 34.52299880981445, 3.9089999198913574], label=124.7)
obj_788 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 15.086000442504883, 35.73699951171875, 3.440999984741211, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 15.88700008392334, 34.52299880981445, 3.9089999198913574, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 16.79050040245056, 33.76974964141846, 3.080749988555908, 0.2,
]
cmd.load_cgo(obj_788, "788")

cmd.pseudoatom(object="angle_789", pos=[10.798999786376953, 37.178001403808594, 4.563000202178955], label=125.0)
obj_789 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 11.798999786376953, 36.20500183105469, 5.230999946594238, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 10.798999786376953, 37.178001403808594, 4.563000202178955, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 10.99524974822998, 37.895999908447266, 4.232000052928925, 0.2,
]
cmd.load_cgo(obj_789, "789")

cmd.pseudoatom(object="angle_790", pos=[11.411999702453613, 32.81800079345703, 8.305000305175781], label=126.8)
obj_790 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 10.342000007629395, 33.82699966430664, 7.815999984741211, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 11.411999702453613, 32.81800079345703, 8.305000305175781, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 11.4793332417806, 32.62066650390625, 8.838000297546387, 0.2,
]
cmd.load_cgo(obj_790, "790")

cmd.pseudoatom(object="angle_791", pos=[6.818999767303467, 32.678001403808594, 10.987000465393066], label=146.2)
obj_791 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 7.784999847412109, 33.80699920654297, 10.630000114440918, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 6.818999767303467, 32.678001403808594, 10.987000465393066, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 5.414399909973144, 32.02620086669922, 12.262200164794923, 0.2,
]
cmd.load_cgo(obj_791, "791")

cmd.pseudoatom(object="angle_792", pos=[8.92199993133545, 37.111000061035156, 14.074999809265137], label=123.0)
obj_792 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 9.173999786376953, 35.6609992980957, 13.656000137329102, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 8.92199993133545, 37.111000061035156, 14.074999809265137, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 9.745999813079834, 37.809000968933105, 15.055000066757202, 0.2,
]
cmd.load_cgo(obj_792, "792")

cmd.pseudoatom(object="angle_793", pos=[10.541999816894531, 32.097999572753906, 16.608999252319336], label=97.2)
obj_793 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 9.710000038146973, 33.38999938964844, 16.66699981689453, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 10.541999816894531, 32.097999572753906, 16.608999252319336, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 9.34375, 31.091750144958496, 15.557249784469604, 0.2,
]
cmd.load_cgo(obj_793, "793")

cmd.pseudoatom(object="angle_794", pos=[8.248000144958496, 36.694000244140625, 19.709999084472656], label=84.3)
obj_794 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 9.545000076293945, 35.86000061035156, 19.542999267578125, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 8.248000144958496, 36.694000244140625, 19.709999084472656, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 7.704333305358887, 35.72433344523112, 19.19066619873047, 0.2,
]
cmd.load_cgo(obj_794, "794")

cmd.pseudoatom(object="angle_795", pos=[9.84000015258789, 32.13199996948242, 22.792999267578125], label=113.0)
obj_795 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 10.526000022888184, 33.422000885009766, 22.309999465942383, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 9.84000015258789, 32.13199996948242, 22.792999267578125, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 7.467285769326346, 32.48357091631208, 23.46328571864537, 0.2,
]
cmd.load_cgo(obj_795, "795")

obj_796 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 13.979000091552734, 33.38800048828125, 20.628999710083008, 0.2,
]
cmd.load_cgo(obj_796, "796")

cmd.pseudoatom(object="angle_797", pos=[17.97800064086914, 30.81399917602539, 19.648000717163086], label=112.4)
obj_797 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 16.586000442504883, 30.652000427246094, 20.2810001373291, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 17.97800064086914, 30.81399917602539, 19.648000717163086, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 18.55049991607666, 31.234999656677246, 20.189000129699707, 0.2,
]
cmd.load_cgo(obj_797, "797")

cmd.pseudoatom(object="angle_798", pos=[16.768999099731445, 26.091999053955078, 21.945999145507812], label=121.6)
obj_798 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 16.874000549316406, 27.548999786376953, 22.44499969482422, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 16.768999099731445, 26.091999053955078, 21.945999145507812, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 16.275749921798706, 25.004499912261963, 22.822750091552734, 0.2,
]
cmd.load_cgo(obj_798, "798")

cmd.pseudoatom(object="angle_799", pos=[21.591999053955078, 28.875, 22.229999542236328], label=121.3)
obj_799 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 20.483999252319336, 28.48900032043457, 23.22599983215332, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 21.591999053955078, 28.875, 22.229999542236328, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 23.00149965286255, 29.006999492645264, 22.65974998474121, 0.2,
]
cmd.load_cgo(obj_799, "799")

cmd.pseudoatom(object="angle_800", pos=[18.844999313354492, 32.992000579833984, 24.15999984741211], label=112.8)
obj_800 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 19.374000549316406, 31.73900032043457, 24.854999542236328, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 18.844999313354492, 32.992000579833984, 24.15999984741211, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 19.842999935150146, 33.64674949645996, 23.358500480651855, 0.2,
]
cmd.load_cgo(obj_800, "800")

cmd.pseudoatom(object="angle_801", pos=[15.380999565124512, 29.167999267578125, 25.7810001373291], label=113.9)
obj_801 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 16.479999542236328, 29.941999435424805, 26.51300048828125, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 15.380999565124512, 29.167999267578125, 25.7810001373291, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 13.093374848365784, 28.52999997138977, 27.438249826431274, 0.2,
]
cmd.load_cgo(obj_801, "801")

cmd.pseudoatom(object="angle_802", pos=[19.31800079345703, 26.048999786376953, 27.707000732421875], label=129.6)
obj_802 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 18.618999481201172, 27.316999435424805, 28.274999618530273, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 19.31800079345703, 26.048999786376953, 27.707000732421875, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 19.37266667683919, 25.8733336130778, 27.16033363342285, 0.2,
]
cmd.load_cgo(obj_802, "802")

cmd.pseudoatom(object="angle_803", pos=[21.881999969482422, 30.92300033569336, 28.493000030517578], label=142.0)
obj_803 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 21.118000030517578, 29.982999801635742, 29.43000030517578, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 21.881999969482422, 30.92300033569336, 28.493000030517578, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 24.402428490774973, 30.96700014386858, 26.224857057843888, 0.2,
]
cmd.load_cgo(obj_803, "803")

cmd.pseudoatom(object="angle_804", pos=[17.298999786376953, 32.94599914550781, 29.947999954223633], label=109.4)
obj_804 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 18.270999908447266, 32.14500045776367, 30.82200050354004, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 17.298999786376953, 32.94599914550781, 29.947999954223633, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 16.971799850463867, 34.719400024414064, 30.7810001373291, 0.2,
]
cmd.load_cgo(obj_804, "804")

cmd.pseudoatom(object="angle_827", pos=[11.48799991607666, 16.2189998626709, 8.607999801635742], label=112.1)
obj_827 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 12.519000053405762, 15.20300006866455, 8.079999923706055, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 11.48799991607666, 16.2189998626709, 8.607999801635742, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 9.928750157356262, 17.380249977111816, 6.486874997615814, 0.2,
]
cmd.load_cgo(obj_827, "827")

cmd.pseudoatom(object="angle_828", pos=[16.29800033569336, 18.14900016784668, 7.0289998054504395], label=123.1)
obj_828 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 15.312999725341797, 17.165000915527344, 6.390999794006348, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 16.29800033569336, 18.14900016784668, 7.0289998054504395, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 15.951749801635742, 19.536749839782715, 7.3445000648498535, 0.2,
]
cmd.load_cgo(obj_828, "828")

cmd.pseudoatom(object="angle_829", pos=[17.687999725341797, 12.862000465393066, 6.539999961853027], label=146.7)
obj_829 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 17.03700065612793, 13.847999572753906, 5.551000118255615, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 17.687999725341797, 12.862000465393066, 6.539999961853027, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 18.809799575805663, 11.194200134277343, 6.772800064086914, 0.2,
]
cmd.load_cgo(obj_829, "829")

cmd.pseudoatom(object="angle_830", pos=[12.45300006866455, 12.404000282287598, 4.599999904632568], label=112.8)
obj_830 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 13.805000305175781, 12.642000198364258, 3.9159998893737793, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 12.45300006866455, 12.404000282287598, 4.599999904632568, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 12.434749841690063, 11.190750122070312, 5.4075000286102295, 0.2,
]
cmd.load_cgo(obj_830, "830")

cmd.pseudoatom(object="angle_831", pos=[13.442999839782715, 17.29400062561035, 2.4600000381469727], label=113.5)
obj_831 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 13.848999977111816, 15.901000022888184, 1.9220000505447388, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 13.442999839782715, 17.29400062561035, 2.4600000381469727, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 10.546571459089007, 18.11814280918666, 1.7081428595951624, 0.2,
]
cmd.load_cgo(obj_831, "831")

cmd.pseudoatom(object="angle_832", pos=[17.481000900268555, 14.595000267028809, -0.4490000009536743], label=159.4)
obj_832 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 17.523000717163086, 15.180999755859375, 0.9869999885559082, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 17.481000900268555, 14.595000267028809, -0.4490000009536743, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 17.75257137843541, 12.711714335850306, -2.568000022854124, 0.2,
]
cmd.load_cgo(obj_832, "832")

cmd.pseudoatom(object="angle_833", pos=[17.36400032043457, 19.57900047302246, 2.6570000648498535], label=125.5)
obj_833 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 18.510000228881836, 18.57699966430664, 2.4179999828338623, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 17.36400032043457, 19.57900047302246, 2.6570000648498535, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 17.52500009536743, 21.032000064849854, 2.8740000128746033, 0.2,
]
cmd.load_cgo(obj_833, "833")

cmd.pseudoatom(object="angle_834", pos=[23.1560001373291, 18.30900001525879, 3.9170000553131104], label=124.9)
obj_834 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 21.72800064086914, 18.643999099731445, 4.436999797821045, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 23.1560001373291, 18.30900001525879, 3.9170000553131104, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 23.458333333333332, 17.831000010172527, 4.057666619618733, 0.2,
]
cmd.load_cgo(obj_834, "834")

cmd.pseudoatom(object="angle_835", pos=[21.277000427246094, 21.097000122070312, 8.607000350952148], label=112.6)
obj_835 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 21.506000518798828, 21.382999420166016, 7.11299991607666, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 21.277000427246094, 21.097000122070312, 8.607000350952148, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 20.638500213623047, 22.9048334757487, 9.735833326975504, 0.2,
]
cmd.load_cgo(obj_835, "835")

cmd.pseudoatom(object="angle_836", pos=[25.60099983215332, 22.652999877929688, 5.966000080108643], label=130.9)
obj_836 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 25.22100067138672, 22.12299919128418, 7.369999885559082, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 25.60099983215332, 22.652999877929688, 5.966000080108643, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 25.061857223510742, 21.44971466064453, 3.0702857034546986, 0.2,
]
cmd.load_cgo(obj_836, "836")

cmd.pseudoatom(object="angle_837", pos=[24.66699981689453, 25.972000122070312, 9.765999794006348], label=114.4)
obj_837 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 24.81399917602539, 24.538999557495117, 10.329999923706055, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 24.66699981689453, 25.972000122070312, 9.765999794006348, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 25.096999645233154, 27.02074956893921, 10.672749757766724, 0.2,
]
cmd.load_cgo(obj_837, "837")

cmd.pseudoatom(object="angle_838", pos=[22.42799949645996, 21.288999557495117, 13.138999938964844], label=121.7)
obj_838 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 22.93600082397461, 22.738000869750977, 13.109000205993652, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 22.42799949645996, 21.288999557495117, 13.138999938964844, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 21.405749797821045, 20.865999698638916, 14.0552499294281, 0.2,
]
cmd.load_cgo(obj_838, "838")

cmd.pseudoatom(object="angle_839", pos=[24.559999465942383, 25.503000259399414, 16.172000885009766], label=180.0)
obj_839 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 23.624000549316406, 24.3439998626709, 16.48900032043457, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 24.559999465942383, 25.503000259399414, 16.172000885009766, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 24.559999465942383, 25.503000259399414, 16.172000885009766, 0.2,
]
cmd.load_cgo(obj_839, "839")

cmd.pseudoatom(object="angle_840", pos=[22.035999298095703, 25.315000534057617, 21.014999389648438], label=180.0)
obj_840 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 21.586000442504883, 25.3439998626709, 19.56800079345703, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 22.035999298095703, 25.315000534057617, 21.014999389648438, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 22.035999298095703, 25.315000534057617, 21.014999389648438, 0.2,
]
cmd.load_cgo(obj_840, "840")

cmd.pseudoatom(object="angle_841", pos=[22.763999938964844, 29.722999572753906, 17.691999435424805], label=128.3)
obj_841 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 21.56399917602539, 28.878000259399414, 18.115999221801758, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 22.763999938964844, 29.722999572753906, 17.691999435424805, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 25.189856937953405, 28.88499995640346, 15.891999925885882, 0.2,
]
cmd.load_cgo(obj_841, "841")

cmd.pseudoatom(object="angle_842", pos=[20.434999465942383, 27.010000228881836, 13.873000144958496], label=113.0)
obj_842 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 19.636999130249023, 27.48699951171875, 15.083000183105469, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 20.434999465942383, 27.010000228881836, 13.873000144958496, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 21.245749950408936, 28.060750484466553, 13.277999877929688, 0.2,
]
cmd.load_cgo(obj_842, "842")

cmd.pseudoatom(object="angle_843", pos=[17.211999893188477, 24.725000381469727, 18.139999389648438], label=127.4)
obj_843 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 16.902999877929688, 25.867000579833984, 17.159000396728516, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 17.211999893188477, 24.725000381469727, 18.139999389648438, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 17.08899974822998, 24.161000569661457, 18.0749994913737, 0.2,
]
cmd.load_cgo(obj_843, "843")

cmd.pseudoatom(object="angle_844", pos=[13.463000297546387, 29.11400032043457, 16.781999588012695], label=119.4)
obj_844 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 14.015999794006348, 28.27199935913086, 17.951000213623047, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 13.463000297546387, 29.11400032043457, 16.781999588012695, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 14.37375020980835, 30.021000385284424, 16.045249700546265, 0.2,
]
cmd.load_cgo(obj_844, "844")

cmd.pseudoatom(object="angle_845", pos=[11.706000328063965, 28.70400047302246, 22.143999099731445], label=128.2)
obj_845 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 11.40999984741211, 28.16900062561035, 20.722000122070312, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 11.706000328063965, 28.70400047302246, 22.143999099731445, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 11.78333314259847, 28.399333953857422, 22.635332743326824, 0.2,
]
cmd.load_cgo(obj_845, "845")

cmd.pseudoatom(object="angle_853", pos=[15.694999694824219, 22.700000762939453, 12.885000228881836], label=143.3)
obj_853 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 15.11400032043457, 24.013999938964844, 13.465999603271484, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 15.694999694824219, 22.700000762939453, 12.885000228881836, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 16.079750061035156, 21.89050006866455, 13.119750022888184, 0.2,
]
cmd.load_cgo(obj_853, "853")

cmd.pseudoatom(object="angle_854", pos=[15.656000137329102, 28.302000045776367, 11.704999923706055], label=127.0)
obj_854 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 16.072999954223633, 26.895999908447266, 11.178999900817871, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 15.656000137329102, 28.302000045776367, 11.704999923706055, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 15.947666803995768, 28.794666926066082, 11.644333521525065, 0.2,
]
cmd.load_cgo(obj_854, "854")

cmd.pseudoatom(object="angle_855", pos=[19.976999282836914, 28.577999114990234, 10.059000015258789], label=113.5)
obj_855 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 19.024999618530273, 28.077999114990234, 8.949000358581543, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 19.976999282836914, 28.577999114990234, 10.059000015258789, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 20.866000175476074, 29.647749423980713, 9.62374997138977, 0.2,
]
cmd.load_cgo(obj_855, "855")

cmd.pseudoatom(object="angle_856", pos=[18.298999786376953, 23.608999252319336, 7.116000175476074], label=113.9)
obj_856 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 19.295000076293945, 24.76799964904785, 6.98799991607666, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 18.298999786376953, 23.608999252319336, 7.116000175476074, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 15.917999948774066, 24.27614266531808, 6.765000002724784, 0.2,
]
cmd.load_cgo(obj_856, "856")

obj_857 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 20.858999252319336, 24.804000854492188, 3.5190000534057617, 0.2,
]
cmd.load_cgo(obj_857, "857")

cmd.pseudoatom(object="angle_858", pos=[22.222999572753906, 29.19300079345703, 4.931000232696533], label=125.1)
obj_858 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 22.374000549316406, 28.250999450683594, 3.7330000400543213, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 22.222999572753906, 29.19300079345703, 4.931000232696533, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 21.851499557495117, 30.58775043487549, 4.851000070571899, 0.2,
]
cmd.load_cgo(obj_858, "858")

cmd.pseudoatom(object="angle_859", pos=[26.545000076293945, 25.93000030517578, 4.745999813079834], label=180.0)
obj_859 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, 25.941999435424805, 26.924999237060547, 3.7669999599456787, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, 26.545000076293945, 25.93000030517578, 4.745999813079834, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, 26.545000076293945, 25.93000030517578, 4.745999813079834, 0.2,
]
cmd.load_cgo(obj_859, "859")

cmd.group("HUMAN_EGFR_3w32_chainA", "716 717 718 719 720 721 722 723 724 725 726 727 728 742 743 744 745 746 747 758 759 760 761 762 763 764 765 766 767 768 769 770 772 773 774 775 776 777 778 779 780 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 853 854 855 856 857 858 859 angle_716 angle_717 angle_718 angle_720 angle_722 angle_723 angle_725 angle_726 angle_727 angle_728 angle_742 angle_743 angle_744 angle_745 angle_746 angle_747 angle_758 angle_759 angle_760 angle_761 angle_762 angle_763 angle_764 angle_765 angle_766 angle_767 angle_768 angle_769 angle_770 angle_772 angle_773 angle_774 angle_775 angle_776 angle_777 angle_778 angle_780 angle_787 angle_788 angle_789 angle_790 angle_791 angle_792 angle_793 angle_794 angle_795 angle_797 angle_798 angle_799 angle_800 angle_801 angle_802 angle_803 angle_804 angle_827 angle_828 angle_829 angle_830 angle_831 angle_832 angle_833 angle_834 angle_835 angle_836 angle_837 angle_838 angle_839 angle_840 angle_841 angle_842 angle_843 angle_844 angle_845 angle_853 angle_854 angle_855 angle_856 angle_858 angle_859")