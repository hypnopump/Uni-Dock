import os
from copy import deepcopy

import multiprocess as mp
from multiprocess.pool import Pool

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D
from rdkit.Chem import ChemicalForceFields
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

from xdatools.modules.docking_engines.autodock_topology import utils

def parse_pdbqt_file(pdbqt_file_name):
    with open(pdbqt_file_name, 'r') as pdbqt_file:
        pdbqt_lines = pdbqt_file.read().split('\n')

    ## Parse headers
    #####################################################################################
    docked_conf_starting_line_idx_list = []
    docked_conf_ending_line_idx_list = []
    for pdbqt_line_idx, pdbqt_line in enumerate(pdbqt_lines):
        if 'MODEL' in pdbqt_line:
            docked_conf_starting_line_idx_list.append(pdbqt_line_idx)
        elif 'ENDMDL' in pdbqt_line:
            docked_conf_ending_line_idx_list.append(pdbqt_line_idx)
    #####################################################################################

    ## Parse docked conformations
    #####################################################################################
    num_docked_conformations = len(docked_conf_starting_line_idx_list)
    docked_conf_line_block_list = [None] * num_docked_conformations
    for docked_conf_idx in range(num_docked_conformations):
        current_docked_conf_starting_line_idx = docked_conf_starting_line_idx_list[docked_conf_idx]
        current_docked_conf_ending_line_idx = docked_conf_ending_line_idx_list[docked_conf_idx]
        docked_conf_line_block_list[docked_conf_idx] = pdbqt_lines[current_docked_conf_starting_line_idx:current_docked_conf_ending_line_idx+1]

    ligand_atom_info_tuple_list = []
    first_docked_conf_line_block = docked_conf_line_block_list[0]
    for current_docked_conf_line in first_docked_conf_line_block:
        current_docked_conf_line_splited_list = current_docked_conf_line.split()
        if current_docked_conf_line_splited_list[0] == 'ATOM':
            current_ligand_atom_name = current_docked_conf_line_splited_list[2]

            ########################################################################################################################################################
            #FIX ME: this is for working around with the ADFR C++ pdbqt atom name writing bug!
            #if current_raw_ligand_atom_name[0].isnumeric():
            #    current_ligand_atom_name = current_raw_ligand_atom_name[1:] + current_raw_ligand_atom_name[0]
            #else:
            #    current_ligand_atom_name = current_raw_ligand_atom_name
            ########################################################################################################################################################

            current_ligand_residue_name = current_docked_conf_line_splited_list[3]
            current_ligand_residue_idx = int(current_docked_conf_line_splited_list[5])
            current_ligand_chain_idx = current_docked_conf_line_splited_list[4]
            
            current_ligand_atom_info_tuple = (current_ligand_chain_idx, current_ligand_residue_name, current_ligand_residue_idx, current_ligand_atom_name)
            ligand_atom_info_tuple_list.append(current_ligand_atom_info_tuple)

    ligand_atom_info_array = np.array(ligand_atom_info_tuple_list, dtype=object)

    num_atoms = ligand_atom_info_array.shape[0]
    docked_conf_positions_list = []
    docked_conf_score_list = []

    for docked_conf_idx in range(num_docked_conformations):
        current_docked_conf_line_block = docked_conf_line_block_list[docked_conf_idx]
        current_docked_conf_atom_info_nested_list = []
        for current_docked_conf_line in current_docked_conf_line_block:
            current_docked_conf_line_splited_list = current_docked_conf_line.split()
            if current_docked_conf_line_splited_list[0] == 'ATOM':
                current_docked_conf_atom_info_nested_list.append(current_docked_conf_line_splited_list)
            elif current_docked_conf_line_splited_list[0] == 'REMARK':
                if current_docked_conf_line_splited_list[1] == 'WATVINA' and current_docked_conf_line_splited_list[2] == 'RESULT:':
                    docked_conf_score_list.append(np.float32(current_docked_conf_line_splited_list[3]))

        if len(current_docked_conf_atom_info_nested_list) != num_atoms:
            raise ValueError('Some bugs in the code or problematic PDBQT file!')

        docked_conf_positions = np.zeros((num_atoms, 3), dtype=np.float32)
        for atom_idx in range(num_atoms):
            current_docked_conf_atom_info_list = current_docked_conf_atom_info_nested_list[atom_idx]
            current_atom_coord_x = np.float32(current_docked_conf_atom_info_list[6])
            current_atom_coord_y = np.float32(current_docked_conf_atom_info_list[7])
            current_atom_coord_z = np.float32(current_docked_conf_atom_info_list[8])
            current_atom_positions = np.array([current_atom_coord_x, current_atom_coord_y, current_atom_coord_z], dtype=np.float32)
            docked_conf_positions[atom_idx, :] = current_atom_positions

        docked_conf_positions_list.append(docked_conf_positions)

    docked_conf_positions_array = np.array(docked_conf_positions_list, dtype=np.float32)
    docked_conf_score_array = np.array(docked_conf_score_list, dtype=np.float32)
    #####################################################################################

    parsed_docking_info_dict = {}
    parsed_docking_info_dict['atom_info'] = ligand_atom_info_array
    parsed_docking_info_dict['docked_positions'] = docked_conf_positions_array
    parsed_docking_info_dict['docked_score'] = docked_conf_score_array

    return parsed_docking_info_dict

def generate_docking_pose_sdf_files(working_dir_name,
                                    ligand_reference_mol,
                                    ligand_atom_info_array,
                                    docked_conf_positions_array,
                                    docked_conf_binding_free_energy_array,
                                    docked_sdf_file_name_prefix):

    num_docked_poses = docked_conf_positions_array.shape[0]
    num_docked_pose_atoms = docked_conf_positions_array.shape[1]
    ComputeGasteigerCharges(ligand_reference_mol)
    utils.assign_atom_properties(ligand_reference_mol)
    num_reference_mol_atoms = ligand_reference_mol.GetNumAtoms()
    num_reference_mol_heavy_atoms = ligand_reference_mol.GetNumHeavyAtoms()
    docked_conf_ligand_efficiency_array = docked_conf_binding_free_energy_array / num_reference_mol_heavy_atoms
    docked_sdf_file_name_list = [None] * num_docked_poses
    docked_conf_energy_list = [None] * num_docked_poses

    for pose_idx in range(num_docked_poses):
        docked_mol = deepcopy(ligand_reference_mol)
        docked_conf_binding_free_energy = docked_conf_binding_free_energy_array[pose_idx]
        docked_conf_ligand_efficiency = docked_conf_ligand_efficiency_array[pose_idx]
        docked_mol.SetProp('watvina_binding_free_energy', str(docked_conf_binding_free_energy))
        docked_mol.SetProp('watvina_ligand_efficiency', str(docked_conf_ligand_efficiency))
        docked_mol_conformer = docked_mol.GetConformer()

        for atom_idx in range(num_docked_pose_atoms):
            current_docked_pose_atom_coord = docked_conf_positions_array[pose_idx, atom_idx, :].astype(np.float64)
            current_docked_pose_atom_coord_point_3D = Point3D(current_docked_pose_atom_coord[0], current_docked_pose_atom_coord[1], current_docked_pose_atom_coord[2])
            current_docked_pose_atom_info_tuple = tuple(ligand_atom_info_array[atom_idx])

            for reference_atom_idx in range(num_reference_mol_atoms):
                reference_atom = docked_mol.GetAtomWithIdx(reference_atom_idx)
                reference_chain_idx = reference_atom.GetProp('chain_idx')
                reference_residue_name = reference_atom.GetProp('residue_name')
                reference_residue_idx = reference_atom.GetIntProp('residue_idx')
                reference_atom_name = reference_atom.GetProp('atom_name')
                reference_atom_info_tuple = (reference_chain_idx, reference_residue_name, reference_residue_idx, reference_atom_name)
                if reference_atom_info_tuple == current_docked_pose_atom_info_tuple:
                    docked_mol_conformer.SetAtomPosition(reference_atom_idx, current_docked_pose_atom_coord_point_3D)
                    break

        ff_property = ChemicalForceFields.MMFFGetMoleculeProperties(docked_mol, 'MMFF94s')
        ff = ChemicalForceFields.MMFFGetMoleculeForceField(docked_mol, ff_property)

        if ff is not None:
            ff.Initialize()
            docked_conf_energy = ff.CalcEnergy()
            docked_mol.SetProp('conformer_energy', str(docked_conf_energy))
        else:
            docked_conf_energy = 1000.0

        docked_sdf_file_name = os.path.join(working_dir_name, docked_sdf_file_name_prefix + '_pose_' + str(pose_idx) + '.sdf')
        sdf_writer = Chem.SDWriter(docked_sdf_file_name)
        sdf_writer.write(docked_mol)
        sdf_writer.flush()
        sdf_writer.close()

        docked_sdf_file_name_list[pose_idx] = docked_sdf_file_name
        docked_conf_energy_list[pose_idx] = docked_conf_energy

    docking_pose_summary_info_dict = {}
    docking_pose_summary_info_dict['ligand_docked_sdf_file_name'] = np.array(docked_sdf_file_name_list, dtype='U')
    docking_pose_summary_info_dict['conformer_energy'] = np.array(docked_conf_energy_list, dtype=np.float32)
    docking_pose_summary_info_dict['binding_free_energy'] = docked_conf_binding_free_energy_array
    docking_pose_summary_info_dict['ligand_efficiency'] = docked_conf_ligand_efficiency_array

    return docking_pose_summary_info_dict

def watvina_parsing_process(reference_sdf_file_name,
                            pdbqt_file_name,
                            working_dir_name,
                            docking_pose_summary_info_df_proxy_list,
                            docked_file_idx):

    reference_mol = Chem.SDMolSupplier(reference_sdf_file_name, removeHs=False)[0]

    if reference_mol.HasProp('smiles_string'):
        ligand_smiles_string = reference_mol.GetProp('smiles_string')
    else:
        reference_mol_no_h = Chem.RemoveHs(reference_mol)
        ligand_smiles_string = Chem.MolToSmiles(reference_mol_no_h)

    docked_sdf_file_name_prefix = os.path.basename(pdbqt_file_name).split('.')[0].replace('_out', '')
    parsed_docking_pdbqt_info_dict = parse_pdbqt_file(pdbqt_file_name)

    ligand_atom_info_array = parsed_docking_pdbqt_info_dict['atom_info']
    docked_conf_positions_array = parsed_docking_pdbqt_info_dict['docked_positions']
    docked_conf_binding_free_energy_array = parsed_docking_pdbqt_info_dict['docked_score']

    docking_pose_summary_info_dict = generate_docking_pose_sdf_files(working_dir_name,
                                                                     reference_mol,
                                                                     ligand_atom_info_array,
                                                                     docked_conf_positions_array,
                                                                     docked_conf_binding_free_energy_array,
                                                                     docked_sdf_file_name_prefix)

    num_docked_conformations = docking_pose_summary_info_dict['ligand_docked_sdf_file_name'].shape[0]
    ligand_smiles_string_array = np.array([ligand_smiles_string] * num_docked_conformations, dtype='U')
    original_sdf_file_name_array = np.array([reference_sdf_file_name] * num_docked_conformations, dtype='U')

    docking_pose_summary_info_dict['ligand_smiles_string'] = ligand_smiles_string_array
    docking_pose_summary_info_dict['ligand_original_sdf_file_name'] = original_sdf_file_name_array

    docking_pose_summary_info_df = pd.DataFrame(docking_pose_summary_info_dict)
    docking_pose_summary_info_df_proxy_list[docked_file_idx] = docking_pose_summary_info_df

    return True

class WatVinaParsing(object):
    def __init__(self,
                 ligand_original_sdf_file_name_list,
                 ligand_docked_pdbqt_file_name_list,
                 n_cpu=16,
                 working_dir_name='.'):

        self.ligand_original_sdf_file_name_list = ligand_original_sdf_file_name_list
        self.ligand_docked_pdbqt_file_name_list = ligand_docked_pdbqt_file_name_list
        self.n_cpu = n_cpu
        self.num_docked_files = len(self.ligand_docked_pdbqt_file_name_list)
        self.working_dir_name = os.path.abspath(working_dir_name)

    def run_watvina_parsing(self):
        manager = mp.Manager()
        docking_pose_summary_info_df_proxy_list = manager.list()
        docking_pose_summary_info_df_proxy_list.extend([None] * self.num_docked_files)
        autodock_parsing_results_list = [None] * self.num_docked_files
        autodock_parsing_pool = Pool(processes=self.n_cpu)

        for docked_file_idx in range(self.num_docked_files):
            reference_sdf_file_name = self.ligand_original_sdf_file_name_list[docked_file_idx]
            pdbqt_file_name = self.ligand_docked_pdbqt_file_name_list[docked_file_idx]
            autodock_parsing_results = autodock_parsing_pool.apply_async(watvina_parsing_process,
                                                                         args=(reference_sdf_file_name,
                                                                               pdbqt_file_name,
                                                                               self.working_dir_name,
                                                                               docking_pose_summary_info_df_proxy_list,
                                                                               docked_file_idx))

            autodock_parsing_results_list[docked_file_idx] = autodock_parsing_results

        autodock_parsing_pool.close()
        autodock_parsing_pool.join()

        autodock_parsing_results_list = [autodock_parsing_results.get() for autodock_parsing_results in autodock_parsing_results_list]
        docking_pose_summary_info_df_list = list(docking_pose_summary_info_df_proxy_list)

        docking_pose_summary_info_merged_df = pd.concat(docking_pose_summary_info_df_list)
        docking_pose_summary_info_merged_df.reset_index(drop=True, inplace=True)
        self.docking_pose_summary_info_df = docking_pose_summary_info_merged_df

        docking_pose_summary_info_csv_file_name = os.path.join(self.working_dir_name, 'docking_pose_summary.csv')
        self.docking_pose_summary_info_df.to_csv(docking_pose_summary_info_csv_file_name, index=False)
