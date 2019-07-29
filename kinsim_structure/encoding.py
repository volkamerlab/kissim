def exposure(mol, kette):
    """Input: MOL2 file of a kinase binding site, PDB code as string
    Calculates half sphere exposure for binding site residues.
    Output: Numpy array with exposure values for all binding site residues (dtype = float)"""
    # write residue IDs of binding site from MOL2 file into list (in the correct order, non redundant)
    res_ids = []
    for name in mol.df.subst_name:
        if name in res_ids:
            continue
        else:
            res_ids.append(name)
    # it happened, that the MOL2 data frame did not only contatin residues but also ligand information
    res_nos = []
    tag_or_lig = []
    for x in res_ids:
        # check if ID id amino acid and not ligand
        if re.match(r'^[A-Z]{3}', x) and not '_' in x:
            res_nos.append(int(x[3:]))
        else:
            tag_or_lig.append(res_ids.index(x))
    # align binding site residue IDs from MOL2 file to the key format of HSE output dictionary
    hse_keys = [(kette.id, (' ', ident,' ')) for ident in res_nos]
    # get protein residue IDs from PDB file and select the binding site residues based on the IDs from MOL2 file
    res_list = Selection.unfold_entities(kette, 'R')
    keys = [res for res in res_list if res.get_full_id()[3][1] in res_nos]
    # compute HSEa and HSEb for the chain
    RADIUS = 13.0
    hse_cb = HSExposureCB(kette, RADIUS)
    hse_ca = HSExposureCA(kette, RADIUS)

    hse_vector = []
    # Case: there is no HSE value computed for the first n residue(s)
    # solution: take value of first residue that HSE could be calculated for and assign it to the missing residues
    # no averaging possible in this case
    i = 0
    if hse_keys[0][1] not in [k[1] for k in hse_cb.keys()] and hse_keys[0][1] not in [k[1] for k in hse_ca.keys()]:
        while hse_keys[i][1] not in [k[1] for k in hse_cb.keys()] and hse_keys[i][1] not in [k[1] for k in hse_ca.keys()]:
            ident = hse_keys[i][1][1]
            rest = [res for res in keys if res.id[1] == ident][0]
            logging.warning('no hse for first residue ('+str(rest.resname)+str(ident)+') of binding site sequence, compensate by next available hse')
            i += 1

    if i!=0:
        key = hse_keys[i]
        ident = key[1][1]
        rest = [res for res in keys if res.id[1] == ident][0]
        pred = rest.resname
        pred_id = ident
        logging.warning('missing hse compensated by value of '+str(pred)+str(pred_id))
        # if there is no hse value in HSEb dict, take value from HSEa dict, else residue is missing
        if key[1] in [k[1] for k in hse_cb.keys()]:
            v = hse_cb[key][0]/float((hse_cb[key][0]+hse_cb[key][1]))
        else:
            v = hse_ca[key][0]/float((hse_ca[key][0]+hse_ca[key][1]))
        hse_vector = [v for c in range(0,i+1)]
        i += 1
        pred_value = v
    # iterate over binding site residue keys and extract HSE value from dictionary

    missing = False
    m = 0
    m_name = []
    for j in range(i, len(hse_keys)):
        # get residue key, residue id and residue entity
        key = hse_keys[j]
        ident = key[1][1]
        rest = [res for res in keys if res.id[1] == ident][0]
        # check dictionaries for residue key, if residue in keys, remember residue ID in case following residue
        # is missing in dict and a dummy needs to be computed
        if key[1] in [k[1] for k in hse_cb.keys()]:
            v = hse_cb[key][0]/float((hse_cb[key][0]+hse_cb[key][1]))
            # if there are missing residues between predecessor and current key, assign average of last and current
            # value to these
            if missing == True:
                dummy = np.average([pred_value, v])
                for n in range(m):
                    hse_vector.append(dummy)
                    n += 1
                logging.info('assigned average hse of '+pred+str(pred_id)+' and '+str(rest.resname)+str(ident)+' to missing residues '+','.join(m_name))
                m = 0
                m_name = []
                missing = False
            hse_vector.append(v)
            pred_value = v
            pred = rest.resname
            pred_id = ident
        elif key[1] in [k[1] for k in hse_ca.keys()]:
            v = hse_ca[key][0]/float((hse_ca[key][0]+hse_ca[key][1]))
            # if there are missing resiude between predecessor and current key, assign average of last and current
            # value to these
            if missing == True:
                dummy = np.average([pred_value, v])
                for n in range(m):
                    hse_vector.append(dummy)
                    n += 1
                logging.info('assigned average hse of '+pred+str(pred_id)+' and '+str(rest.resname)+str(ident)+' to missing residues '+','.join(m_name))
                m = 0
                m_name = []
                missing = False
            hse_vector.append(v)
            pred_value = v
            pred = rest.resname
            pred_id = ident
        else:
            # if residue is no dict key, raise counter
            logging.warning('no hse available for current residue '+str(rest.resname)+str(ident))
            missing = True
            m += 1
        # for missing residues at the end of the binding site sequence, append value of last available residue
    while len(hse_vector) < len(hse_keys):
        hse_vector.append(pred_value)
    for elem in tag_or_lig:
        hse_vector.insert(elem,0.0)
    return np.array(hse_vector, dtype=float)


# Copyright (C) 2010, Joao Rodrigues (anaryin@gmail.com)
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
Module with assorted geometrical functions on
macromolecules.
"""
from Bio.PDB import Entity
def center_of_mass(entity, geometric=False):
    """
    Returns gravitic [default] or geometric center of mass of an Entity.
    Geometric assumes all masses are equal (geometric=True)
    """
    # Structure, Model, Chain, Residue
    if isinstance(entity, Entity.Entity):
        atom_list = entity.get_atoms()
    # List of Atoms
    elif hasattr(entity, '__iter__') and [x for x in entity if x.level == 'A']:
        atom_list = entity
    else: # Some other weirdo object
        raise ValueError("Center of Mass can only be calculated from the following objects:\n"
                            "Structure, Model, Chain, Residue, list of Atoms.")

    masses = []
    positions = [ [], [], [] ] # [ [X1, X2, ..] , [Y1, Y2, ...] , [Z1, Z2, ...] ]

    for atom in atom_list:
        masses.append(atom.mass)

        for i, coord in enumerate(atom.coord.tolist()):
            positions[i].append(coord)

    # If there is a single atom with undefined mass complain loudly.
    if 'ukn' in set(masses) and not geometric:
        raise ValueError("Some Atoms don't have an element assigned.\n"
                         "Try adding them manually or calculate the geometrical center of mass instead.")

    if geometric:
        return [sum(coord_list)/len(masses) for coord_list in positions]
    else:
        w_pos = [ [], [], [] ]
        for atom_index, atom_mass in enumerate(masses):
            w_pos[0].append(positions[0][atom_index]*atom_mass)
            w_pos[1].append(positions[1][atom_index]*atom_mass)
            w_pos[2].append(positions[2][atom_index]*atom_mass)

        return [sum(coord_list)/sum(masses) for coord_list in w_pos]


def side_chain_orientation(mol, kette):
    """
    Input: MOL2 file, PDB code as string, chain as char/string
    Calculates side chain orientation for every binding site residue
    Output: side chain orientation values for the binding site (numpy array, dtype = float)
    """
    # write residue IDs of binding site from MOL2 file into list (in the correct order, non redundant)
    res_ids = []
    for name in mol.df.subst_name:
        if name in res_ids:
            continue
        else:
            res_ids.append(name)
    res_nos = []
    tag_or_lig = []
    for x in res_ids:
        # check if ID id amino acid and not ligand
        if re.match(r'^[A-Z]{3}', x) and not '_' in x:
            res_nos.append(int(x[3:]))
        else:
            tag_or_lig.append(res_ids.index(x))
    # get protein residue IDs from PDB file and select the binding site residues based on the IDs from MOL2 file
    res_list = Selection.unfold_entities(kette, 'R')
    keys = [res for res in res_list if res.get_full_id()[3][1] in res_nos]

    # for every binding site residue check availability of CA and CB atom in its atom list
    sco_vector = []
    for res in keys:
        atomlist = [a.fullname for a in res.get_atoms()]
        if 'CA' in atomlist:
        # CA and CB available => calculate angle between the binding site's CoM, the residue's CA and CB
            if 'CB' in atomlist:
                vector1 = res['CA'].get_vector()
                vector2 = res['CB'].get_vector()
                vector3 = Vector(center_of_mass(res,geometric=True))
                angle =  np.degrees(calc_angle(vector1, vector2, vector3))
            else:
                angle = 0.0
        else:
        # if only CB available => replace CA by midpoint between CB and binding site's CoM
            if 'CB' in atomlist:
                logging.warning('no CA in atomlist for '+str(res.resname)+str(res.id)+', side chain orientation = 0')
                angle = 0.0
            else:
                logging.warning('neither CA nor CB in atomlist for '+str(res.resname)+str(res.id)+', side chain orientation = 0')
                angle = 0.0
        sco_vector.append(round(angle,2))
    for elem in tag_or_lig:
        sco_vector.insert(elem, 0.0)
    return np.array(sco_vector, dtype=float)


def size_feature(res):
    """
    Input: A residue's three letter code as string
    Output: The residue's size value according to SiteAlign feature encoding (int)
    """
    S_res = ['ALA','CYS','GLY','PRO','SER','THR','VAL']
    M_res = ['ASN','ASP','GLN','GLU','HIS','ILE','LEU','LYS','MET']
    L_res = ['ARG','PHE','TRP','TYR']
    if res in S_res:
        return 1
    elif res in M_res:
        return 2
    elif res in L_res:
        return 3


def hbd_feature(res):
    """
    Input: A residue's three letter code as string
    Output: The residue's H-bond donor value according to SiteAlign feature encoding (int)
    """
    HBD_null = ['ALA','ASP','GLY','GLU','ILE','LEU','MET','PHE','PRO','VAL']
    HBD_one = ['ASN','CYS','GLN','HIS','LYS','SER','THR','TRP','TYR']
    HBD_three = ['ARG']
    if res in HBD_null:
        return 0
    elif res in HBD_one:
        return 1
    elif res in HBD_three:
        return 3


def hba_feature(res):
    """
    Input: A residue's three letter code as string
    Output: The residue's H-bond acceptor value according to SiteAlign feature encoding (int)
    """
    HBA_null = ['ALA','ARG','CYS','GLY','ILE','LEU','LYS','MET','PHE','PRO','TRP','VAL']
    HBA_one = ['ASN','GLN','HIS','SER','THR','TYR']
    HBA_two = ['ASP','GLU']
    if res in HBA_null:
        return 0
    elif res in HBA_one:
        return 1
    elif res in HBA_two:
        return 2


def charge_feature(res):
    """
    Input: A residue's three letter code as string
    Output: The residue's charge value according to SiteAlign feature encoding (int)
    """
    zero = ['ALA','ASN','CYS','GLY','GLN','HIS','ILE','LEU','MET','PHE','PRO','SER','TRP','TYR','VAL']
    plus = ['ARG','LYS','THR']
    minus = ['ASP','GLU']
    if res in zero:
        return 0
    elif res in plus:
        return 1
    elif res in minus:
        return -1


def aromatic_feature(res):
    """
    Input: A residue's three letter code as string
    Output: The residue's aromatic value according to SiteAlign feature encoding (int)
    """
    nix = ['VAL','THR','SER','PRO','MET','LYS','LEU','ILE','GLU','GLN','GLY','CYS', 'ASP','ASN','ARG','ALA']
    eins = ['TYR','TRP','PHE','HIS']
    if res in nix:
        return 0
    elif res in eins:
        return 1


def aliphatic_feature(res):
    """
    Input: A residue's three letter code as string
    Output: The residue's aliphatic value according to SiteAlign feature encoding (int)
    """
    null = ['ARG','ASN','ASP','GLY','GLN','GLU','HIS','LYS','PHE','SER','TRP','TYR']
    one = ['ALA','CYS','ILE','LEU','MET','PRO','THR','VAL']
    if res in null:
        return 0
    elif res in one:
        return 1