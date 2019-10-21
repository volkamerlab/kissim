from pymol import *
import os
from pymol.cgo import *

cmd.fetch("5l4q")
cmd.remove("solvent")
cmd.remove("5l4q and not chain A")
cmd.remove("all and not (resi 50+51+52+53+54+55+56+57+58+59+60+61+62+71+72+73+74+75+76+86+87+88+89+90+91+92+93+94+95+96+97+98+101+102+103+104+105+106+107+108+109+123+124+125+126+127+128+129+130+131+132+133+134+135+136+137+138+139+140+166+167+168+169+170+171+172+173+174+175+176+177+178+179+180+181+182+183+184+192+193+194+195+196+197+198)")

cmd.pseudoatom(object="angle_50", pos=[-25.32900047302246, 5.755000114440918, -21.389999389648438], label=121.6)
obj_50 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -26.617000579833984, 5.059999942779541, -21.839000701904297, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -25.32900047302246, 5.755000114440918, -21.389999389648438, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -24.556727496060457, 5.278909163041548, -21.218545393510297, 0.2,
]
cmd.load_cgo(obj_50, "50")

cmd.pseudoatom(object="angle_51", pos=[-26.615999221801758, 5.8470001220703125, -26.608999252319336], label=109.5)
obj_51 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -25.999000549316406, 4.874000072479248, -25.579999923706055, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -26.615999221801758, 5.8470001220703125, -26.608999252319336, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -27.009166717529297, 5.665416757265727, -26.777833302815754, 0.2,
]
cmd.load_cgo(obj_51, "51")

cmd.pseudoatom(object="angle_52", pos=[-22.176000595092773, 1.9539999961853027, -26.200000762939453], label=116.9)
obj_52 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -22.569000244140625, 3.4240000247955322, -26.343000411987305, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -22.176000595092773, 1.9539999961853027, -26.200000762939453, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -22.05786679585775, 1.48706666380167, -24.976800028483073, 0.2,
]
cmd.load_cgo(obj_52, "52")

cmd.pseudoatom(object="angle_53", pos=[-21.750999450683594, 3.131999969482422, -30.753999710083008], label=20.9)
obj_53 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -22.777999877929688, 4.01800012588501, -30.076000213623047, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -21.750999450683594, 3.131999969482422, -30.753999710083008, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -22.084499677022297, 3.33133335908254, -30.35366694132487, 0.2,
]
cmd.load_cgo(obj_53, "53")

cmd.pseudoatom(object="angle_54", pos=[-26.20199966430664, 6.3979997634887695, -32.242000579833984], label=127.8)
obj_54 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -25.45599937438965, 5.09499979019165, -32.518001556396484, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -26.20199966430664, 6.3979997634887695, -32.242000579833984, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -27.0256364995783, 6.545363642952659, -32.31827319752086, 0.2,
]
cmd.load_cgo(obj_54, "54")

obj_55 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -25.40399932861328, 3.822000026702881, -36.084999084472656, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -25.33300018310547, 3.5213332970937095, -35.65833282470703, 0.2,
]
cmd.load_cgo(obj_55, "55")

obj_56 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -27.649999618530273, 4.875999927520752, -38.952999114990234, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -27.055999755859375, 4.950000127156575, -39.024000803629555, 0.2,
]
cmd.load_cgo(obj_56, "56")

cmd.pseudoatom(object="angle_57", pos=[-29.71299934387207, 0.9200000166893005, -40.1879997253418], label=109.6)
obj_57 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -29.645999908447266, 1.6330000162124634, -38.83100128173828, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -29.71299934387207, 0.9200000166893005, -40.1879997253418, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -27.82518768310547, 0.32368749659508467, -40.74775004386902, 0.2,
]
cmd.load_cgo(obj_57, "57")

cmd.pseudoatom(object="angle_58", pos=[-27.45400047302246, -1.0509999990463257, -35.83300018310547], label=21.6)
obj_58 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -28.42300033569336, 0.08500000089406967, -35.564998626708984, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -27.45400047302246, -1.0509999990463257, -35.83300018310547, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -27.75933329264323, -0.6036666681369146, -35.93433380126953, 0.2,
]
cmd.load_cgo(obj_58, "58")

cmd.pseudoatom(object="angle_59", pos=[-29.027000427246094, 2.181999921798706, -31.415000915527344], label=143.1)
obj_59 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -27.895000457763672, 1.2979999780654907, -31.989999771118164, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -29.027000427246094, 2.181999921798706, -31.415000915527344, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -29.25206667582194, 2.5298666656017303, -30.802266438802082, 0.2,
]
cmd.load_cgo(obj_59, "59")

cmd.pseudoatom(object="angle_60", pos=[-24.297000885009766, -0.5260000228881836, -29.46500015258789], label=107.6)
obj_60 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -25.77199935913086, -0.17100000381469727, -29.191999435424805, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -24.297000885009766, -0.5260000228881836, -29.46500015258789, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -24.282333374023438, -0.9513333514332771, -29.592833201090496, 0.2,
]
cmd.load_cgo(obj_60, "60")

cmd.pseudoatom(object="angle_61", pos=[-28.292999267578125, 0.2630000114440918, -25.156999588012695], label=110.3)
obj_61 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -26.843000411987305, 0.4749999940395355, -25.600000381469727, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -28.292999267578125, 0.2630000114440918, -25.156999588012695, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -29.588500261306763, 1.1774375019595027, -26.478875041007996, 0.2,
]
cmd.load_cgo(obj_61, "61")

cmd.pseudoatom(object="angle_62", pos=[-24.180999755859375, 1.0230000019073486, -21.55900001525879], label=117.7)
obj_62 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -25.145000457763672, 0.06700000166893005, -22.266000747680664, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -24.180999755859375, 1.0230000019073486, -21.55900001525879, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -23.57560005187988, 0.6039333343505859, -20.61120007832845, 0.2,
]
cmd.load_cgo(obj_62, "62")

cmd.pseudoatom(object="angle_71", pos=[-25.180999755859375, -6.796000003814697, -21.11199951171875], label=78.6)
obj_71 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -24.101999282836914, -5.7220001220703125, -20.957000732421875, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -25.180999755859375, -6.796000003814697, -21.11199951171875, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -25.070714133126394, -6.84357145854405, -20.68400001525879, 0.2,
]
cmd.load_cgo(obj_71, "71")

cmd.pseudoatom(object="angle_72", pos=[-22.856000900268555, -4.236000061035156, -25.368999481201172], label=31.0)
obj_72 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -24.058000564575195, -4.251999855041504, -24.451000213623047, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -22.856000900268555, -4.236000061035156, -25.368999481201172, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -23.04283332824707, -4.337833325068156, -24.94266668955485, 0.2,
]
cmd.load_cgo(obj_72, "72")

cmd.pseudoatom(object="angle_73", pos=[-28.638999938964844, -4.014999866485596, -25.458999633789062], label=118.4)
obj_73 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -27.39699935913086, -4.35699987411499, -26.28499984741211, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -28.638999938964844, -4.014999866485596, -25.458999633789062, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -29.68193333943685, -4.500133323669433, -25.77459996541341, 0.2,
]
cmd.load_cgo(obj_73, "73")

cmd.pseudoatom(object="angle_74", pos=[-26.725000381469727, -4.724999904632568, -31.03499984741211], label=139.9)
obj_74 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -27.618999481201172, -3.9749999046325684, -30.047000885009766, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -26.725000381469727, -4.724999904632568, -31.03499984741211, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -25.99727768368191, -4.567944447199504, -33.13322247399224, 0.2,
]
cmd.load_cgo(obj_74, "74")

cmd.pseudoatom(object="angle_75", pos=[-31.97100067138672, -2.24399995803833, -30.60099983215332], label=146.4)
obj_75 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -31.006000518798828, -3.0309998989105225, -31.48699951171875, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -31.97100067138672, -2.24399995803833, -30.60099983215332, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -33.79700002670288, -0.7270499985665083, -30.52355012893677, 0.2,
]
cmd.load_cgo(obj_75, "75")

cmd.pseudoatom(object="angle_76", pos=[-31.05500030517578, -2.940999984741211, -36.04199981689453], label=115.9)
obj_76 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -32.1609992980957, -2.5759999752044678, -35.05099868774414, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -31.05500030517578, -2.940999984741211, -36.04199981689453, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -30.224538656381462, -4.370615390630869, -35.72423083965595, 0.2,
]
cmd.load_cgo(obj_76, "76")

cmd.pseudoatom(object="angle_86", pos=[-28.5049991607666, -8.958000183105469, -42.83000183105469], label=109.6)
obj_86 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -28.701000213623047, -10.413999557495117, -42.375, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -28.5049991607666, -8.958000183105469, -42.83000183105469, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -28.18125009536743, -8.942083398501078, -43.18566703796387, 0.2,
]
cmd.load_cgo(obj_86, "86")

cmd.pseudoatom(object="angle_87", pos=[-30.951000213623047, -9.428999900817871, -38.2130012512207], label=69.4)
obj_87 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -29.709999084472656, -10.156999588012695, -38.72200012207031, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -30.951000213623047, -9.428999900817871, -38.2130012512207, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -30.56485721043178, -9.025142805916923, -38.521428789411274, 0.2,
]
cmd.load_cgo(obj_87, "87")

cmd.pseudoatom(object="angle_88", pos=[-31.285999298095703, -14.734999656677246, -39.28900146484375], label=138.2)
obj_88 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -30.336000442504883, -13.894000053405762, -38.433998107910156, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -31.285999298095703, -14.734999656677246, -39.28900146484375, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -33.57822216881646, -15.108722156948513, -39.48127788967557, 0.2,
]
cmd.load_cgo(obj_88, "88")

cmd.pseudoatom(object="angle_89", pos=[-26.3799991607666, -14.232000350952148, -41.268001556396484], label=172.1)
obj_89 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -26.892000198364258, -14.612000465393066, -39.875999450683594, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -26.3799991607666, -14.232000350952148, -41.268001556396484, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -25.620350074768066, -13.57189998626709, -42.864599990844724, 0.2,
]
cmd.load_cgo(obj_89, "89")

cmd.pseudoatom(object="angle_90", pos=[-25.812999725341797, -10.598999977111816, -37.367000579833984], label=132.4)
obj_90 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -25.360000610351562, -12.055000305175781, -37.51900100708008, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -25.812999725341797, -10.598999977111816, -37.367000579833984, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -25.48472681912509, -9.943999940698797, -36.95936341719194, 0.2,
]
cmd.load_cgo(obj_90, "90")

cmd.pseudoatom(object="angle_91", pos=[-28.45599937438965, -13.942000389099121, -34.263999938964844], label=135.4)
obj_91 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -26.94099998474121, -13.831999778747559, -34.551998138427734, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -28.45599937438965, -13.942000389099121, -34.263999938964844, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -29.05446662902832, -13.43546657562256, -33.87726643880208, 0.2,
]
cmd.load_cgo(obj_91, "91")

cmd.pseudoatom(object="angle_92", pos=[-26.42799949645996, -17.898000717163086, -37.15700149536133], label=127.9)
obj_92 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -25.958999633789062, -17.28499984741211, -35.83599853515625, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -26.42799949645996, -17.898000717163086, -37.15700149536133, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -27.859769381009617, -18.13530782552866, -37.622692401592545, 0.2,
]
cmd.load_cgo(obj_92, "92")

cmd.pseudoatom(object="angle_93", pos=[-21.89299964904785, -14.97599983215332, -37.28300094604492], label=135.5)
obj_93 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -22.3700008392334, -16.145000457763672, -36.39400100708008, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -21.89299964904785, -14.97599983215332, -37.28300094604492, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -21.674866739908854, -14.879733276367187, -38.12926686604818, 0.2,
]
cmd.load_cgo(obj_93, "93")

cmd.pseudoatom(object="angle_94", pos=[-22.952999114990234, -13.54800033569336, -32.34199905395508], label=124.8)
obj_94 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -22.125999450683594, -14.711999893188477, -32.882999420166016, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -22.952999114990234, -13.54800033569336, -32.34199905395508, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -22.492845975435696, -12.338000004108135, -32.13038444519043, 0.2,
]
cmd.load_cgo(obj_94, "94")

cmd.pseudoatom(object="angle_95", pos=[-25.10700035095215, -18.31100082397461, -31.57200050354004], label=153.8)
obj_95 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -23.722000122070312, -17.7549991607666, -31.243000030517578, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -25.10700035095215, -18.31100082397461, -31.57200050354004, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -26.980949878692627, -19.17135009765625, -31.074399852752684, 0.2,
]
cmd.load_cgo(obj_95, "95")

cmd.pseudoatom(object="angle_96", pos=[-21.469999313354492, -20.503999710083008, -34.5], label=83.4)
obj_96 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -21.424999237060547, -20.208999633789062, -32.999000549316406, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -21.469999313354492, -20.503999710083008, -34.5, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -21.435625076293945, -20.902124881744385, -34.373374462127686, 0.2,
]
cmd.load_cgo(obj_96, "96")

cmd.pseudoatom(object="angle_97", pos=[-17.92099952697754, -17.016000747680664, -33.387001037597656], label=114.2)
obj_97 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -18.27899932861328, -18.13800048828125, -32.40999984741211, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -17.92099952697754, -17.016000747680664, -33.387001037597656, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -17.80466651916504, -17.410133361816406, -34.64779993693034, 0.2,
]
cmd.load_cgo(obj_97, "97")

cmd.pseudoatom(object="angle_98", pos=[-20.327999114990234, -16.67799949645996, -28.22800064086914], label=28.6)
obj_98 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -18.95800018310547, -16.954999923706055, -28.851999282836914, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -20.327999114990234, -16.67799949645996, -28.22800064086914, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -20.03471429007394, -16.60028580256871, -28.28299985613142, 0.2,
]
cmd.load_cgo(obj_98, "98")

cmd.pseudoatom(object="angle_101", pos=[-9.706999778747559, -15.687000274658203, -26.025999069213867], label=130.9)
obj_101 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -11.098999977111816, -15.293000221252441, -26.527999877929688, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -9.706999778747559, -15.687000274658203, -26.025999069213867, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -7.870555533303155, -16.06394444571601, -27.098944452073837, 0.2,
]
cmd.load_cgo(obj_101, "101")

cmd.pseudoatom(object="angle_102", pos=[-9.748000144958496, -13.019000053405762, -30.413999557495117], label=86.7)
obj_102 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -10.876999855041504, -12.791999816894531, -29.40999984741211, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -9.748000144958496, -13.019000053405762, -30.413999557495117, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -9.100700044631958, -12.996799945831299, -29.601099967956543, 0.2,
]
cmd.load_cgo(obj_102, "102")

cmd.pseudoatom(object="angle_103", pos=[-15.22599983215332, -13.619000434875488, -30.992000579833984], label=134.9)
obj_103 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -14.592000007629395, -12.454999923706055, -30.19300079345703, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -15.22599983215332, -13.619000434875488, -30.992000579833984, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -15.158866691589356, -13.898466555277507, -31.759733708699546, 0.2,
]
cmd.load_cgo(obj_103, "103")

cmd.pseudoatom(object="angle_104", pos=[-17.774999618530273, -9.055000305175781, -28.649999618530273], label=116.9)
obj_104 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -17.113000869750977, -10.388999938964844, -28.232999801635742, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -17.774999618530273, -9.055000305175781, -28.649999618530273, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -18.1361665725708, -9.05483341217041, -28.68600018819173, 0.2,
]
cmd.load_cgo(obj_104, "104")

obj_105 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -19.367000579833984, -12.635000228881836, -26.177000045776367, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -18.85266621907552, -12.430000305175781, -25.94466718037923, 0.2,
]
cmd.load_cgo(obj_105, "105")

cmd.pseudoatom(object="angle_106", pos=[-23.297000885009766, -14.371999740600586, -28.267000198364258], label=113.7)
obj_106 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -22.94499969482422, -13.47700023651123, -27.073999404907227, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -23.297000885009766, -14.371999740600586, -28.267000198364258, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -25.45335309645709, -14.783470602596507, -28.460235371309167, 0.2,
]
cmd.load_cgo(obj_106, "106")

cmd.pseudoatom(object="angle_107", pos=[-25.555999755859375, -12.541999816894531, -23.180999755859375], label=136.0)
obj_107 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -25.410999298095703, -13.708999633789062, -24.18000030517578, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -25.555999755859375, -12.541999816894531, -23.180999755859375, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -25.06919987996419, -12.179933420817058, -22.60126673380534, 0.2,
]
cmd.load_cgo(obj_107, "107")

cmd.pseudoatom(object="angle_108", pos=[-29.829999923706055, -13.71500015258789, -24.601999282836914], label=68.6)
obj_108 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -28.847999572753906, -13.805000305175781, -25.774999618530273, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -29.829999923706055, -13.71500015258789, -24.601999282836914, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -29.903125286102295, -14.101875066757202, -24.857374668121338, 0.2,
]
cmd.load_cgo(obj_108, "108")

cmd.pseudoatom(object="angle_109", pos=[-30.180999755859375, -12.652000427246094, -29.913000106811523], label=48.2)
obj_109 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -30.952999114990234, -12.42300033569336, -28.61199951171875, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -30.180999755859375, -12.652000427246094, -29.913000106811523, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -30.311857223510742, -13.016714368547712, -29.457714080810547, 0.2,
]
cmd.load_cgo(obj_109, "109")

cmd.pseudoatom(object="angle_123", pos=[-33.020999908447266, -7.400000095367432, -28.875], label=117.3)
obj_123 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -32.111000061035156, -7.464000225067139, -30.10300064086914, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -33.020999908447266, -7.400000095367432, -28.875, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -33.824933242797854, -6.400599988301595, -28.789066569010416, 0.2,
]
cmd.load_cgo(obj_123, "123")

cmd.pseudoatom(object="angle_124", pos=[-27.634000778198242, -8.906000137329102, -30.53700065612793], label=134.6)
obj_124 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -28.6560001373291, -8.87600040435791, -29.381000518798828, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -27.634000778198242, -8.906000137329102, -30.53700065612793, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -27.608666610717773, -9.277799987792969, -31.30573336283366, 0.2,
]
cmd.load_cgo(obj_124, "124")

cmd.pseudoatom(object="angle_125", pos=[-28.229999542236328, -9.482999801635742, -24.6299991607666], label=121.2)
obj_125 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -27.399999618530273, -8.932999610900879, -25.79199981689453, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -28.229999542236328, -9.482999801635742, -24.6299991607666, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -28.209599939982095, -8.958866691589355, -23.54533322652181, 0.2,
]
cmd.load_cgo(obj_125, "125")

cmd.pseudoatom(object="angle_126", pos=[-23.020000457763672, -8.343999862670898, -26.82900047302246], label=125.4)
obj_126 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -23.607999801635742, -9.027999877929688, -25.594999313354492, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -23.020000457763672, -8.343999862670898, -26.82900047302246, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -23.71576925424429, -7.6182307830223674, -27.725000088031475, 0.2,
]
cmd.load_cgo(obj_126, "126")

cmd.pseudoatom(object="angle_127", pos=[-19.253999710083008, -9.357999801635742, -23.481000900268555], label=86.0)
obj_127 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -20.618999481201172, -8.6899995803833, -23.29599952697754, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -19.253999710083008, -9.357999801635742, -23.481000900268555, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -19.601624727249146, -9.980125188827515, -23.388499975204468, 0.2,
]
cmd.load_cgo(obj_127, "127")

cmd.pseudoatom(object="angle_128", pos=[-20.687999725341797, -4.383999824523926, -21.10099983215332], label=110.6)
obj_128 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -19.945999145507812, -5.14300012588501, -22.197999954223633, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -20.687999725341797, -4.383999824523926, -21.10099983215332, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -19.987124919891357, -2.5929999854415655, -20.921937465667725, 0.2,
]
cmd.load_cgo(obj_128, "128")

cmd.pseudoatom(object="angle_129", pos=[-15.668999671936035, -4.395999908447266, -24.187999725341797], label=66.5)
obj_129 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -16.31599998474121, -4.224999904632568, -22.812000274658203, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -15.668999671936035, -4.395999908447266, -24.187999725341797, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -15.968571526663643, -4.8899999005453925, -23.997428349086217, 0.2,
]
cmd.load_cgo(obj_129, "129")

cmd.pseudoatom(object="angle_130", pos=[-15.107999801635742, -1.7829999923706055, -18.79199981689453], label=127.2)
obj_130 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -15.472000122070312, -1.534000039100647, -20.257999420166016, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -15.107999801635742, -1.7829999923706055, -18.79199981689453, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -15.723416884740194, -2.3911666522423425, -18.003000179926556, 0.2,
]
cmd.load_cgo(obj_130, "130")

obj_131 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -12.23900032043457, -0.7689999938011169, -22.076000213623047, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -12.244333267211914, -1.2880000124375026, -21.73033332824707, 0.2,
]
cmd.load_cgo(obj_131, "131")

obj_132 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -14.218000411987305, 0.4690000116825104, -25.065000534057617, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -14.511333465576172, 0.32866668701171875, -24.571666717529297, 0.2,
]
cmd.load_cgo(obj_132, "132")

cmd.pseudoatom(object="angle_133", pos=[-13.496999740600586, 1.1579999923706055, -29.78700065612793], label=126.2)
obj_133 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -12.864999771118164, 0.39500001072883606, -28.61400032043457, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -13.496999740600586, 1.1579999923706055, -29.78700065612793, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -14.92192319723276, 1.673538483106173, -29.856153928316555, 0.2,
]
cmd.load_cgo(obj_133, "133")

cmd.pseudoatom(object="angle_134", pos=[-8.520000457763672, -0.36000001430511475, -30.736000061035156], label=111.2)
obj_134 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -9.196000099182129, 0.4480000138282776, -29.606000900268555, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -8.520000457763672, -0.36000001430511475, -30.736000061035156, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -8.212583343187967, -0.6571666821837425, -30.5666667620341, 0.2,
]
cmd.load_cgo(obj_134, "134")

cmd.pseudoatom(object="angle_135", pos=[-11.107000350952148, 4.507999897003174, -31.577999114990234], label=109.4)
obj_135 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -9.88599967956543, 4.017000198364258, -30.76799964904785, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -11.107000350952148, 4.507999897003174, -31.577999114990234, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -10.970916668574015, 4.59674996137619, -32.03949991861979, 0.2,
]
cmd.load_cgo(obj_135, "135")

cmd.pseudoatom(object="angle_136", pos=[-11.51200008392334, 4.252999782562256, -26.200000762939453], label=86.7)
obj_136 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -10.50100040435791, 4.923999786376953, -27.128000259399414, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -11.51200008392334, 4.252999782562256, -26.200000762939453, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -12.29680004119873, 4.678700041770935, -26.850600051879884, 0.2,
]
cmd.load_cgo(obj_136, "136")

cmd.pseudoatom(object="angle_137", pos=[-6.599999904632568, 2.115000009536743, -26.44499969482422], label=116.1)
obj_137 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -7.046000003814697, 3.572999954223633, -26.312000274658203, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -6.599999904632568, 2.115000009536743, -26.44499969482422, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -5.8270666122436525, 1.7003333499034246, -25.572266896565754, 0.2,
]
cmd.load_cgo(obj_137, "137")

cmd.pseudoatom(object="angle_138", pos=[-6.01200008392334, 5.413000106811523, -30.674999237060547], label=138.5)
obj_138 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -5.566999912261963, 5.52400016784668, -29.2189998626709, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -6.01200008392334, 5.413000106811523, -30.674999237060547, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -5.70653853049645, 4.6840769694401665, -31.778615364661583, 0.2,
]
cmd.load_cgo(obj_138, "138")

cmd.pseudoatom(object="angle_139", pos=[-8.496000289916992, 9.029000282287598, -27.548999786376953], label=114.2)
obj_139 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -7.015999794006348, 8.758999824523926, -27.82900047302246, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -8.496000289916992, 9.029000282287598, -27.548999786376953, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -9.107499980926514, 8.778300046920776, -28.299099922180176, 0.2,
]
cmd.load_cgo(obj_139, "139")

cmd.pseudoatom(object="angle_140", pos=[-5.171000003814697, 6.931000232696533, -23.73200035095215], label=128.6)
obj_140 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -4.9720001220703125, 8.116000175476074, -24.680999755859375, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -5.171000003814697, 6.931000232696533, -23.73200035095215, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -6.448846156780537, 6.413692290966328, -23.134077072143555, 0.2,
]
cmd.load_cgo(obj_140, "140")

cmd.pseudoatom(object="angle_166", pos=[-10.836999893188477, -16.672000885009766, -42.94900131225586], label=105.7)
obj_166 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -12.1899995803833, -17.274999618530273, -42.58000183105469, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -10.836999893188477, -16.672000885009766, -42.94900131225586, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -11.120153940640963, -14.895153852609488, -43.09784610454853, 0.2,
]
cmd.load_cgo(obj_166, "166")

cmd.pseudoatom(object="angle_167", pos=[-10.284000396728516, -21.216999053955078, -40.82699966430664], label=126.5)
obj_167 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -11.543999671936035, -20.906999588012695, -41.641998291015625, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -10.284000396728516, -21.216999053955078, -40.82699966430664, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -8.907384652357836, -20.689153817983772, -41.080768878643326, 0.2,
]
cmd.load_cgo(obj_167, "167")

cmd.pseudoatom(object="angle_168", pos=[-16.29599952697754, -20.895000457763672, -40.479000091552734], label=90.9)
obj_168 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -15.11299991607666, -21.847000122070312, -40.69300079345703, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -16.29599952697754, -20.895000457763672, -40.479000091552734, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -16.039856910705566, -20.645286015101842, -40.11800003051758, 0.2,
]
cmd.load_cgo(obj_168, "168")

cmd.pseudoatom(object="angle_169", pos=[-17.329999923706055, -25.881000518798828, -42.382999420166016], label=135.7)
obj_169 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -16.395000457763672, -24.74799919128418, -42.80099868774414, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -17.329999923706055, -25.881000518798828, -42.382999420166016, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -18.1905001534356, -27.709222157796223, -43.37411096360948, 0.2,
]
cmd.load_cgo(obj_169, "169")

cmd.pseudoatom(object="angle_170", pos=[-20.017000198364258, -21.923999786376953, -44.340999603271484], label=89.6)
obj_170 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -18.48699951171875, -22.038000106811523, -44.433998107910156, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -20.017000198364258, -21.923999786376953, -44.340999603271484, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -20.050600242614745, -22.595299911499023, -44.15759925842285, 0.2,
]
cmd.load_cgo(obj_170, "170")

cmd.pseudoatom(object="angle_171", pos=[-15.168999671936035, -18.856000900268555, -45.762001037597656], label=84.6)
obj_171 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -16.107999801635742, -19.033000946044922, -44.5620002746582, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -15.168999671936035, -18.856000900268555, -45.762001037597656, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -15.4721999168396, -19.906200218200684, -46.01860046386719, 0.2,
]
cmd.load_cgo(obj_171, "171")

cmd.pseudoatom(object="angle_172", pos=[-17.665000915527344, -15.508000373840332, -41.805999755859375], label=136.2)
obj_172 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -17.197999954223633, -15.628000259399414, -43.270999908447266, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -17.665000915527344, -15.508000373840332, -41.805999755859375, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -18.265733273824058, -15.832933362325033, -41.34486694335938, 0.2,
]
cmd.load_cgo(obj_172, "172")

cmd.pseudoatom(object="angle_173", pos=[-16.292999267578125, -11.755999565124512, -46.08100128173828], label=135.3)
obj_173 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -16.114999771118164, -12.270000457763672, -44.6349983215332, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -16.292999267578125, -11.755999565124512, -46.08100128173828, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -16.06233336130778, -12.043733215332031, -46.8800666809082, 0.2,
]
cmd.load_cgo(obj_173, "173")

cmd.pseudoatom(object="angle_174", pos=[-15.244999885559082, -9.312999725341797, -40.86899948120117], label=109.7)
obj_174 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -16.35099983215332, -9.694000244140625, -41.85300064086914, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -15.244999885559082, -9.312999725341797, -40.86899948120117, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -15.882461547851562, -8.518076823307918, -39.61446233896109, 0.2,
]
cmd.load_cgo(obj_174, "174")

cmd.pseudoatom(object="angle_175", pos=[-17.724000930786133, -7.081999778747559, -45.09700012207031], label=163.2)
obj_175 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -16.395999908447266, -6.822999954223633, -44.37699890136719, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -17.724000930786133, -7.081999778747559, -45.09700012207031, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -19.310400199890136, -7.8782000064849855, -46.51425037384033, 0.2,
]
cmd.load_cgo(obj_175, "175")

cmd.pseudoatom(object="angle_176", pos=[-17.356000900268555, -3.5220000743865967, -41.257999420166016], label=70.3)
obj_176 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -16.01799964904785, -4.051000118255615, -41.78799819946289, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -17.356000900268555, -3.5220000743865967, -41.257999420166016, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -17.0923752784729, -3.1870000064373016, -41.35237407684326, 0.2,
]
cmd.load_cgo(obj_176, "176")

cmd.pseudoatom(object="angle_177", pos=[-11.67300033569336, -5.755000114440918, -40.507999420166016], label=117.8)
obj_177 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -12.734999656677246, -4.75600004196167, -40.05099868774414, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -11.67300033569336, -5.755000114440918, -40.507999420166016, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -11.92880007425944, -7.045466709136963, -40.34879989624024, 0.2,
]
cmd.load_cgo(obj_177, "177")

cmd.pseudoatom(object="angle_178", pos=[-12.324999809265137, -0.40700000524520874, -39.04199981689453], label=141.2)
obj_178 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -11.48799991607666, -1.5460000038146973, -38.47200012207031, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -12.324999809265137, -0.40700000524520874, -39.04199981689453, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -14.389333142174614, 0.409888896263308, -39.00544463263618, 0.2,
]
cmd.load_cgo(obj_178, "178")

cmd.pseudoatom(object="angle_179", pos=[-9.105999946594238, 1.0219999551773071, -35.10100173950195], label=105.4)
obj_179 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -10.097000122070312, -0.14900000393390656, -35.231998443603516, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -9.105999946594238, 1.0219999551773071, -35.10100173950195, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -8.942083358764648, 1.1589999937762816, -35.66933377583822, 0.2,
]
cmd.load_cgo(obj_179, "179")

cmd.pseudoatom(object="angle_180", pos=[-14.609000205993652, 1.715999960899353, -34.944000244140625], label=97.9)
obj_180 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -13.592000007629395, 0.9020000100135803, -34.137001037597656, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -14.609000205993652, 1.715999960899353, -34.944000244140625, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -13.900636239485307, 2.5728181925686924, -35.27609148892489, 0.2,
]
cmd.load_cgo(obj_180, "180")

cmd.pseudoatom(object="angle_181", pos=[-15.144000053405762, -3.3940000534057617, -35.49599838256836], label=114.6)
obj_181 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -14.654000282287598, -2.765000104904175, -34.18899917602539, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -15.144000053405762, -3.3940000534057617, -35.49599838256836, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -15.727599811553954, -2.8462999939918516, -36.002099609375, 0.2,
]
cmd.load_cgo(obj_181, "181")

cmd.pseudoatom(object="angle_182", pos=[-10.354000091552734, -3.9509999752044678, -32.362998962402344], label=96.8)
obj_182 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -11.838000297546387, -4.011000156402588, -31.958999633789062, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -10.354000091552734, -3.9509999752044678, -32.362998962402344, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -10.373333358764649, -4.678133328755696, -32.971266428629555, 0.2,
]
cmd.load_cgo(obj_182, "182")

cmd.pseudoatom(object="angle_183", pos=[-14.175999641418457, -4.3520002365112305, -27.881000518798828], label=117.9)
obj_183 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -12.762999534606934, -3.9119999408721924, -28.263999938964844, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -14.175999641418457, -4.3520002365112305, -27.881000518798828, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -15.174466705322265, -3.8565333048502604, -28.557400258382163, 0.2,
]
cmd.load_cgo(obj_183, "183")

cmd.pseudoatom(object="angle_184", pos=[-10.13599967956543, -3.1489999294281006, -24.284000396728516], label=121.2)
obj_184 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -10.800000190734863, -4.301000118255615, -25.042999267578125, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -10.13599967956543, -3.1489999294281006, -24.284000396728516, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -9.520266691843668, -2.1684666792551677, -24.926866658528645, 0.2,
]
cmd.load_cgo(obj_184, "184")

cmd.pseudoatom(object="angle_192", pos=[-13.312000274658203, -9.366000175476074, -33.891998291015625], label=123.4)
obj_192 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -13.657999992370605, -8.475000381469727, -32.698001861572266, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -13.312000274658203, -9.366000175476074, -33.891998291015625, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -13.425866762797037, -9.020933278401692, -35.013999684651694, 0.2,
]
cmd.load_cgo(obj_192, "192")

cmd.pseudoatom(object="angle_193", pos=[-17.615999221801758, -6.25, -31.677000045776367], label=80.2)
obj_193 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -17.261999130249023, -7.251999855041504, -32.777000427246094, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -17.615999221801758, -6.25, -31.677000045776367, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -17.31742831638881, -6.030714273452759, -31.88128580365862, 0.2,
]
cmd.load_cgo(obj_193, "193")

cmd.pseudoatom(object="angle_194", pos=[-19.45400047302246, -4.834000110626221, -35.763999938964844], label=68.7)
obj_194 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -19.96299934387207, -6.185999870300293, -35.257999420166016, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -19.45400047302246, -4.834000110626221, -35.763999938964844, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -19.97850012779236, -4.879750102758408, -35.83024978637695, 0.2,
]
cmd.load_cgo(obj_194, "194")

cmd.pseudoatom(object="angle_195", pos=[-21.011999130249023, -10.491999626159668, -36.81800079345703], label=107.7)
obj_195 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -21.003000259399414, -9.102999687194824, -37.452999114990234, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -21.011999130249023, -10.491999626159668, -36.81800079345703, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -19.293875098228455, -10.675250053405762, -35.679062366485596, 0.2,
]
cmd.load_cgo(obj_195, "195")

obj_196 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -23.915000915527344, -7.244999885559082, -39.040000915527344, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -23.779999415079754, -7.150333404541016, -38.47999954223633, 0.2,
]
cmd.load_cgo(obj_196, "196")

cmd.pseudoatom(object="angle_197", pos=[-20.656999588012695, -5.456999778747559, -42.073001861572266], label=70.3)
obj_197 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -21.77400016784668, -6.500999927520752, -42.09600067138672, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -20.656999588012695, -5.456999778747559, -42.073001861572266, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -20.96528570992606, -5.33157137462071, -41.844428471156526, 0.2,
]
cmd.load_cgo(obj_197, "197")

cmd.pseudoatom(object="angle_198", pos=[-20.447999954223633, -11.041000366210938, -41.25299835205078], label=35.0)
obj_198 = [
	COLOR, 1.0, 0.0, 0.0,
	SPHERE, -20.65999984741211, -10.109999656677246, -42.433998107910156, 0.2,
	COLOR, 0.0, 1.0, 1.0,
	SPHERE, -20.447999954223633, -11.041000366210938, -41.25299835205078, 0.2,
	COLOR, 0.0, 0.0, 1.0,
	SPHERE, -20.518166542053223, -10.595499833424887, -41.38416735331217, 0.2,
]
cmd.load_cgo(obj_198, "198")

cmd.group("HUMAN_AAK1_5l4q_altA_chainA", "50 51 52 53 54 55 56 57 58 59 60 61 62 71 72 73 74 75 76 86 87 88 89 90 91 92 93 94 95 96 97 98 101 102 103 104 105 106 107 108 109 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 192 193 194 195 196 197 198 angle_50 angle_51 angle_52 angle_53 angle_54 angle_57 angle_58 angle_59 angle_60 angle_61 angle_62 angle_71 angle_72 angle_73 angle_74 angle_75 angle_76 angle_86 angle_87 angle_88 angle_89 angle_90 angle_91 angle_92 angle_93 angle_94 angle_95 angle_96 angle_97 angle_98 angle_101 angle_102 angle_103 angle_104 angle_106 angle_107 angle_108 angle_109 angle_123 angle_124 angle_125 angle_126 angle_127 angle_128 angle_129 angle_130 angle_133 angle_134 angle_135 angle_136 angle_137 angle_138 angle_139 angle_140 angle_166 angle_167 angle_168 angle_169 angle_170 angle_171 angle_172 angle_173 angle_174 angle_175 angle_176 angle_177 angle_178 angle_179 angle_180 angle_181 angle_182 angle_183 angle_184 angle_192 angle_193 angle_194 angle_195 angle_197 angle_198")