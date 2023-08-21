from copy import deepcopy
import re

from rdkit import Chem
from rdkit.Chem import ChemicalForceFields
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem import rdFMCS

from parker.atom_mapping.atom_mapping import AtomMapping
from parker.molecule.parser import ParkerFileParser

def get_mol_without_indices(mol_input,
                            remove_indices=[],
                            keep_properties=[],
                            keep_mol_properties=[]):

    mol_property_dict = {}
    for mol_property_name in keep_mol_properties:
        mol_property_dict[mol_property_name] = mol_input.GetProp(mol_property_name)

    atom_list, bond_list, idx_map = [], [], {}  # idx_map: {old: new}

    for atom in mol_input.GetAtoms():

        props = {}
        for property_name in keep_properties:
            if property_name in atom.GetPropsAsDict():
                props[property_name] = atom.GetPropsAsDict()[property_name]

        symbol = atom.GetSymbol()

        if symbol.startswith('*'):
            atom_symbol = '*'
            props['molAtomMapNumber'] = atom.GetAtomMapNum()

        elif symbol.startswith('R'):
            atom_symbol = '*'
            if len(symbol) > 1:
                atom_map_num = int(symbol[1:])
            else:
                atom_map_num = atom.GetAtomMapNum()
            props['molAtomMapNumber'] = atom_map_num
            props['dummyLabel'] = 'R' + str(atom_map_num)
            props['_MolFileRLabel'] = str(atom_map_num)

        else:
            atom_symbol = symbol

        atom_list.append(
            (
                atom_symbol,
                atom.GetChiralTag(),
                atom.GetFormalCharge(),
                atom.GetNumExplicitHs(),
                props
            )
        )

    for bond in mol_input.GetBonds():
        bond_list.append(
            (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBondType()
            )
        )

    mol = Chem.RWMol(Chem.Mol())

    new_idx = 0
    for atom_index, atom_info in enumerate(atom_list):
        if atom_index not in remove_indices:
            atom = Chem.Atom(atom_info[0])
            atom.SetChiralTag(atom_info[1])
            atom.SetFormalCharge(atom_info[2])
            atom.SetNumExplicitHs(atom_info[3])

            for property_name in atom_info[4]:
                if isinstance(atom_info[4][property_name], str):
                    atom.SetProp(property_name, atom_info[4][property_name])
                elif isinstance(atom_info[4][property_name], int):
                    atom.SetIntProp(property_name, atom_info[4][property_name])

            mol.AddAtom(atom)
            idx_map[atom_index] = new_idx
            new_idx += 1

    for bond_info in bond_list:
        if (
            bond_info[0] not in remove_indices
            and bond_info[1] not in remove_indices
        ):
            mol.AddBond(
                idx_map[bond_info[0]],
                idx_map[bond_info[1]],
                bond_info[2]
            )

        else:
            one_in = False
            if (
                (bond_info[0] in remove_indices)
                and (bond_info[1] not in remove_indices)
            ):
                keep_index = bond_info[1]
                remove_index = bond_info[0]
                one_in = True
            elif (
                (bond_info[1] in remove_indices)
                and (bond_info[0] not in remove_indices)
            ):
                keep_index = bond_info[0]
                remove_index = bond_info[1]
                one_in = True
            if one_in:
                if atom_list[keep_index][0] in ['N', 'P']:
                    old_num_explicit_Hs = mol.GetAtomWithIdx(
                        idx_map[keep_index]
                    ).GetNumExplicitHs()

                    mol.GetAtomWithIdx(idx_map[keep_index]).SetNumExplicitHs(
                        old_num_explicit_Hs + 1
                    )

    mol = Chem.Mol(mol)

    for mol_property_name in mol_property_dict:
        mol.SetProp(mol_property_name, mol_property_dict[mol_property_name])

    Chem.GetSymmSSSR(mol)
    mol.UpdatePropertyCache(strict=False)
    return mol

def prepare_covalent_ligand_mol(mol):
    covalent_atom_idx_string = mol.GetProp('covalent_atom_indices')
    covalent_atom_idx_string_list = covalent_atom_idx_string.split(',')
    covalent_atom_idx_list = [int(covalent_atom_idx_string) for covalent_atom_idx_string in covalent_atom_idx_string_list]

    covalent_atom_name_string = mol.GetProp('covalent_atom_names')
    covalent_atom_name_list = covalent_atom_name_string.split(',')

    covalent_residue_name_string = mol.GetProp('covalent_residue_names')
    covalent_residue_name_list = covalent_residue_name_string.split(',')

    covalent_residue_idx_string = mol.GetProp('covalent_residue_indices')
    covalent_residue_idx_string_list = covalent_residue_idx_string.split(',')
    covalent_residue_idx_list = [int(covalent_residue_idx_string) for covalent_residue_idx_string in covalent_residue_idx_string_list]

    covalent_chain_idx_string = mol.GetProp('covalent_chain_indices')
    covalent_chain_idx_list = covalent_chain_idx_string.split(',')

    covalent_anchor_atom_info = (covalent_chain_idx_list[0], covalent_residue_name_list[0], covalent_residue_idx_list[0], covalent_atom_name_list[0])

    removed_atom_idx_list = []

    num_atoms = mol.GetNumAtoms()
    internal_atom_idx = 0
    atom_coords_dict = {}
    conformer = mol.GetConformer()
    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() == 0:
            atom.SetProp('atom_name', 'None')
            removed_atom_idx_list.append(atom_idx)
        else:
            if atom_idx in covalent_atom_idx_list:
                atom_name = covalent_atom_name_list[covalent_atom_idx_list.index(atom_idx)]
                residue_name = covalent_residue_name_list[covalent_atom_idx_list.index(atom_idx)]
                residue_idx = covalent_residue_idx_list[covalent_atom_idx_list.index(atom_idx)]
                chain_idx = covalent_chain_idx_list[covalent_atom_idx_list.index(atom_idx)]
                atom.SetProp('atom_name', atom_name)
                atom.SetProp('residue_name', residue_name)
                atom.SetIntProp('residue_idx', residue_idx)
                atom.SetProp('chain_idx', chain_idx)
            else:
                atom_element = atom.GetSymbol()
                atom_name = atom_element + str(internal_atom_idx+1)
                atom.SetProp('atom_name', atom_name)
                atom.SetProp('residue_name', 'MOL')
                atom.SetIntProp('residue_idx', 1)
                atom.SetProp('chain_idx', 'A')
                internal_atom_idx += 1

            atom_coords_point_3D = deepcopy(conformer.GetAtomPosition(atom_idx))
            atom_coords_dict[atom_name] = atom_coords_point_3D

    covalent_mol = get_mol_without_indices(mol, remove_indices=removed_atom_idx_list, keep_properties=['atom_name', 'residue_name', 'residue_idx', 'chain_idx'])
    num_covalent_atoms = covalent_mol.GetNumAtoms()
    covalent_conformer = Chem.Conformer(num_covalent_atoms)

    for atom_idx in range(num_covalent_atoms):
        atom = covalent_mol.GetAtomWithIdx(atom_idx)
        atom_name = atom.GetProp('atom_name')
        atom_coords_point_3D = atom_coords_dict[atom_name]
        covalent_conformer.SetAtomPosition(atom_idx, atom_coords_point_3D)

    _ = covalent_mol.AddConformer(covalent_conformer)
    _ = Chem.SanitizeMol(covalent_mol)

    return covalent_mol, covalent_anchor_atom_info

def get_template_docking_atom_mapping_parker(reference_mol, query_mol, atom_mapping_scheme='all'):
    parker_file_parser = ParkerFileParser()
    reference_parker_mol = parker_file_parser.from_rdmol(reference_mol)
    query_parker_mol = parker_file_parser.from_rdmol(query_mol)

    if atom_mapping_scheme == 'all':
        parker_atom_mapping_rgroup = AtomMapping(reference_parker_mol,
                                                 query_parker_mol,
                                                 pre_match_ring=False,
                                                 detect_chirality=False,
                                                 scheme=0)

        parker_atom_mapping_core_hopping_1 = AtomMapping(reference_parker_mol,
                                                         query_parker_mol,
                                                         pre_match_ring=False,
                                                         detect_chirality=False,
                                                         scheme=1)

        parker_atom_mapping_core_hopping_2 = AtomMapping(reference_parker_mol,
                                                         query_parker_mol,
                                                         pre_match_ring=False,
                                                         detect_chirality=False,
                                                         scheme=2)

        rgroup_atom_mapping_list = parker_atom_mapping_rgroup.find()
        core_hopping_1_atom_mapping_list = parker_atom_mapping_core_hopping_1.find()
        core_hopping_2_atom_mapping_list = parker_atom_mapping_core_hopping_2.find()

        num_matched_rgroup_atoms = len(rgroup_atom_mapping_list)
        num_matched_core_hopping_1_atoms = len(core_hopping_1_atom_mapping_list)
        num_matched_core_hopping_2_atoms = len(core_hopping_2_atom_mapping_list)

        if num_matched_core_hopping_2_atoms > num_matched_core_hopping_1_atoms and num_matched_core_hopping_2_atoms > num_matched_rgroup_atoms:
            atom_mapping_list = core_hopping_2_atom_mapping_list
        elif num_matched_core_hopping_1_atoms >= num_matched_core_hopping_2_atoms and num_matched_core_hopping_1_atoms > num_matched_rgroup_atoms:
            atom_mapping_list = core_hopping_1_atom_mapping_list
        else:
            atom_mapping_list = rgroup_atom_mapping_list

    else:
        parker_atom_mapping = AtomMapping(reference_parker_mol,
                                          query_parker_mol,
                                          pre_match_ring=False,
                                          detect_chirality=False,
                                          scheme=atom_mapping_scheme)

        atom_mapping_list = parker_atom_mapping.find()

    # to get flip_mapping for some symmetry case
    core_atom_mapping_dict = dict(atom_mapping_list)

    return core_atom_mapping_dict

def get_template_docking_atom_mapping(reference_mol, query_mol):
    mcs = rdFMCS.FindMCS([query_mol, reference_mol],
                     ringMatchesRingOnly=True,
                     completeRingsOnly=True,
                     atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
                     bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                     timeout=60)

    mcs_string = mcs.smartsString.replace('#0', '*')
    generic_core_smarts_string = re.sub('\[\*.*?\]', '[*]', mcs_string)

    generic_core_mol = Chem.MolFromSmarts(generic_core_smarts_string)

    reference_atom_mapping = reference_mol.GetSubstructMatches(generic_core_mol)[0]
    query_atom_mapping = query_mol.GetSubstructMatches(generic_core_mol)[0]

    core_atom_mapping_dict = {reference_atom_idx: query_atom_idx for reference_atom_idx, query_atom_idx in zip(reference_atom_mapping, query_atom_mapping)}

    return core_atom_mapping_dict

def get_core_alignment_for_template_docking(reference_mol, query_mol):
    core_atom_mapping_dict = get_template_docking_atom_mapping(reference_mol, query_mol)

    core_atom_mapping_dict = {query_atom_idx: reference_atom_idx for reference_atom_idx, query_atom_idx in core_atom_mapping_dict.items()}
    # the initial position of query_mol is random, so align to the reference_mol firstly
    _ = AlignMol(query_mol, reference_mol, atomMap=list(core_atom_mapping_dict.items()))

    # assign template positions from reference mol to query mol
    core_fixed_query_conformer = Chem.Conformer(query_mol.GetNumAtoms())
    reference_conformer = reference_mol.GetConformer()
    query_conformer = query_mol.GetConformer()

    for query_atom_idx in range(query_mol.GetNumAtoms()):
        if query_atom_idx in core_atom_mapping_dict:
            reference_atom_idx = core_atom_mapping_dict[query_atom_idx]
            atom_position = reference_conformer.GetAtomPosition(reference_atom_idx)
            core_fixed_query_conformer.SetAtomPosition(query_atom_idx, atom_position)
        else:
            atom_position = query_conformer.GetAtomPosition(query_atom_idx)
            core_fixed_query_conformer.SetAtomPosition(query_atom_idx, atom_position)

    query_mol.RemoveAllConformers()
    query_mol.AddConformer(core_fixed_query_conformer)

    # optimize conformer using chemical forcefield
    ff_property = ChemicalForceFields.MMFFGetMoleculeProperties(query_mol, 'MMFF94s')
    ff = ChemicalForceFields.MMFFGetMoleculeForceField(query_mol, ff_property, confId=0)

    for query_atom_idx in core_atom_mapping_dict.keys():
        reference_atom_idx = core_atom_mapping_dict[query_atom_idx]
        core_atom_position = reference_conformer.GetAtomPosition(reference_atom_idx)
        virtual_site_atom_idx = ff.AddExtraPoint(core_atom_position.x, core_atom_position.y, core_atom_position.z, fixed=True) - 1
        ff.AddDistanceConstraint(virtual_site_atom_idx, query_atom_idx, 0, 0, 100.0)

    ff.Initialize()

    max_minimize_iteration = 5
    for _ in range(max_minimize_iteration):
        minimize_seed = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
        if minimize_seed == 0:
            break

    query_mol.SetProp('aligned_conformer_energy', str(ff.CalcEnergy()))

    core_atom_idx_list = list(core_atom_mapping_dict.keys())

    return core_atom_idx_list

def assign_atom_properties(mol):
    atom_positions = mol.GetConformer().GetPositions()
    num_atoms = mol.GetNumAtoms()

    internal_atom_idx = 0
    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom.SetIntProp('sdf_atom_idx', atom_idx+1)
        if not atom.HasProp('atom_name'):
            atom_element = atom.GetSymbol()
            atom_name = atom_element + str(internal_atom_idx+1)
            atom.SetProp('atom_name', atom_name)
            atom.SetProp('residue_name', 'MOL')
            atom.SetIntProp('residue_idx', 1)
            atom.SetProp('chain_idx', 'A')
            internal_atom_idx += 1

        atom.SetDoubleProp('charge', atom.GetDoubleProp('_GasteigerCharge'))
        atom.SetDoubleProp('x', atom_positions[atom_idx, 0])
        atom.SetDoubleProp('y', atom_positions[atom_idx, 1])
        atom.SetDoubleProp('z', atom_positions[atom_idx, 2])
