import os
import re
from copy import deepcopy

import multiprocess as mp
from multiprocess.pool import Pool

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D
from rdkit.Chem import ChemicalForceFields

def parse_dlg_file(dlg_file_name):
    with open(dlg_file_name, 'r') as dlg_file:
        dlg_lines = dlg_file.read().split('\n')

    ## Parse headers
    #####################################################################################
    docked_conf_starting_line_idx_list = []
    docked_conf_ending_line_idx_list = []
    for dlg_line_idx, dlg_line in enumerate(dlg_lines):
        if 'DOCKED: MODEL' in dlg_line:
            docked_conf_starting_line_idx_list.append(dlg_line_idx)
        elif 'DOCKED: ENDMDL' in dlg_line:
            docked_conf_ending_line_idx_list.append(dlg_line_idx)
        elif 'CLUSTERING HISTOGRAM' in dlg_line:
            docked_cluster_starting_line_idx = dlg_line_idx
        elif 'RMSD TABLE' in dlg_line:
            docked_cluster_ending_line_idx = dlg_line_idx
    #####################################################################################

    ## Parse docked clusters
    #####################################################################################
    docked_cluster_info_nested_list = []
    docked_cluster_line_block = dlg_lines[docked_cluster_starting_line_idx:docked_cluster_ending_line_idx+1]
    for docked_cluster_line in docked_cluster_line_block:
        docked_cluster_line_split_list = docked_cluster_line.strip().split()
        if len(docked_cluster_line_split_list) > 0:
            if docked_cluster_line_split_list[0].isnumeric():
                docked_cluster_info_nested_list.append(docked_cluster_line_split_list)

    num_docked_clusters = len(docked_cluster_info_nested_list)

    cluster_representative_pose_idx_array = np.zeros((num_docked_clusters), dtype=np.int32)
    cluster_size_array = np.zeros((num_docked_clusters), dtype=np.int32)
    for cluster_conf_idx in range(num_docked_clusters):
        docked_cluster_line_split_list = docked_cluster_info_nested_list[cluster_conf_idx]
        cluster_representative_pose_idx = np.int32(docked_cluster_line_split_list[4]) - 1
        cluster_representative_pose_idx_array[cluster_conf_idx] = cluster_representative_pose_idx
        cluster_size_array[cluster_conf_idx] = np.int32(docked_cluster_line_split_list[8])
    #####################################################################################

    ## Parse docked conformations
    #####################################################################################
    docked_conf_line_block_list = [None] * num_docked_clusters
    for cluster_conf_idx in range(num_docked_clusters):
        docked_conf_idx = cluster_representative_pose_idx_array[cluster_conf_idx]
        current_docked_conf_starting_line_idx = docked_conf_starting_line_idx_list[docked_conf_idx]
        current_docked_conf_ending_line_idx = docked_conf_ending_line_idx_list[docked_conf_idx]
        docked_conf_line_block_list[cluster_conf_idx] = dlg_lines[current_docked_conf_starting_line_idx:current_docked_conf_ending_line_idx+1]

    ligand_atom_name_list = []
    first_docked_conf_line_block = docked_conf_line_block_list[0]
    for current_docked_conf_line in first_docked_conf_line_block:
        if 'DOCKED: ATOM' in current_docked_conf_line:
            current_docked_conf_line_splited_list = current_docked_conf_line.split()
            ligand_atom_name_list.append(current_docked_conf_line_splited_list[3])

    ligand_atom_name_array = np.array(ligand_atom_name_list, dtype='U')

    num_atoms = ligand_atom_name_array.shape[0]
    binding_free_energy_list = []
    docked_conf_positions_list = []

    for cluster_conf_idx in range(num_docked_clusters):
        current_docked_conf_line_block = docked_conf_line_block_list[cluster_conf_idx]
        current_docked_conf_atom_line_block = []
        for current_docked_conf_line in current_docked_conf_line_block:
            if 'Estimated Free Energy of Binding' in current_docked_conf_line:
                binding_free_energy = np.float32(re.findall('[-+]?\d*\.\d+|\d+', current_docked_conf_line)[0])
                binding_free_energy_list.append(binding_free_energy)

            elif 'DOCKED: ATOM' in current_docked_conf_line:
                current_docked_conf_atom_line_block.append(current_docked_conf_line)

        if len(current_docked_conf_atom_line_block) != num_atoms:
            raise ValueError('Some bugs in the code or problematic DLG file!')

        docked_conf_positions = np.zeros((num_atoms, 3), dtype=np.float32)
        for atom_idx in range(num_atoms):
            current_docked_conf_atom_line = current_docked_conf_atom_line_block[atom_idx]
            current_docked_conf_atom_line_splited_list = re.findall('[-+]?\d*\.\d+|\d+', current_docked_conf_atom_line)
            current_atom_coord_x = np.float32(current_docked_conf_atom_line_splited_list[3])
            current_atom_coord_y = np.float32(current_docked_conf_atom_line_splited_list[4])
            current_atom_coord_z = np.float32(current_docked_conf_atom_line_splited_list[5])
            current_atom_positions = np.array([current_atom_coord_x, current_atom_coord_y, current_atom_coord_z], dtype=np.float32)
            docked_conf_positions[atom_idx, :] = current_atom_positions

        docked_conf_positions_list.append(docked_conf_positions)

    binding_free_energy_array = np.array(binding_free_energy_list, dtype=np.float32)
    docked_conf_positions_array = np.array(docked_conf_positions_list, dtype=np.float32)
    #####################################################################################

    parsed_docking_info_dict = {}
    parsed_docking_info_dict['atom_names'] = ligand_atom_name_array
    parsed_docking_info_dict['docked_positions'] = docked_conf_positions_array
    parsed_docking_info_dict['binding_free_energy'] = binding_free_energy_array
    parsed_docking_info_dict['cluster_size'] = cluster_size_array

    return parsed_docking_info_dict

def generate_docking_pose_sdf_files(working_dir_name,
                                    ligand_reference_mol,
                                    ligand_atom_name_array,
                                    docked_conf_positions_array,
                                    docked_sdf_file_name_prefix):

    num_docked_poses = docked_conf_positions_array.shape[0]
    num_docked_pose_atoms = docked_conf_positions_array.shape[1]
    docked_sdf_file_name_list = [None] * num_docked_poses
    docked_conf_energy_list = [None] * num_docked_poses

    for pose_idx in range(num_docked_poses):
        docked_mol = deepcopy(ligand_reference_mol)
        docked_mol_conformer = docked_mol.GetConformer()
        for atom_idx in range(num_docked_pose_atoms):
            current_docked_pose_atom_coord = docked_conf_positions_array[pose_idx, atom_idx, :].astype(np.float64)
            current_docked_pose_atom_coord_point_3D = Point3D(current_docked_pose_atom_coord[0], current_docked_pose_atom_coord[1], current_docked_pose_atom_coord[2])
            current_docked_pose_atom_name = ligand_atom_name_array[atom_idx]
            current_docked_pose_atom_idx = int(re.findall(r'\d+', current_docked_pose_atom_name)[0]) - 1
            docked_mol_conformer.SetAtomPosition(current_docked_pose_atom_idx, current_docked_pose_atom_coord_point_3D)

#        docked_mol_no_h = Chem.RemoveHs(docked_mol)
#        docked_mol_added_h = Chem.AddHs(docked_mol_no_h, addCoords=True)

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

    return docked_sdf_file_name_list, docked_conf_energy_list

def autodock_parsing_process(reference_sdf_file_name,
                             dlg_file_name,
                             working_dir_name,
                             docking_pose_summary_info_df_proxy_list,
                             docked_file_idx):

    reference_mol = Chem.SDMolSupplier(reference_sdf_file_name, removeHs=False)[0]
    mol_num_heavy_atoms = reference_mol.GetNumHeavyAtoms()

    if reference_mol.HasProp('smiles_string'):
        ligand_smiles_string = reference_mol.GetProp('smiles_string')
    else:
        reference_mol_no_h = Chem.RemoveHs(reference_mol)
        ligand_smiles_string = Chem.MolToSmiles(reference_mol_no_h)

    docked_sdf_file_name_prefix = os.path.basename(dlg_file_name).split('.')[0]

    parsed_docking_info_dict = parse_dlg_file(dlg_file_name)
    ligand_atom_name_array = parsed_docking_info_dict['atom_names']
    docked_conf_positions_array = parsed_docking_info_dict['docked_positions']
    binding_free_energy_array = parsed_docking_info_dict['binding_free_energy']
    cluster_size_array = parsed_docking_info_dict['cluster_size']

    ligand_efficiency_array = binding_free_energy_array / mol_num_heavy_atoms
    docked_sdf_file_name_list, docked_conf_energy_list = generate_docking_pose_sdf_files(working_dir_name,
                                                                                         reference_mol,
                                                                                         ligand_atom_name_array,
                                                                                         docked_conf_positions_array,
                                                                                         docked_sdf_file_name_prefix)

    docked_sdf_file_name_array = np.array(docked_sdf_file_name_list, dtype='U')
    docked_conf_energy_array = np.array(docked_conf_energy_list, dtype=np.float32)
    num_docked_conformations = docked_sdf_file_name_array.shape[0]
    ligand_smiles_string_array = np.array([ligand_smiles_string] * num_docked_conformations, dtype='U')
    original_sdf_file_name_array = np.array([reference_sdf_file_name] * num_docked_conformations, dtype='U')

    docking_pose_summary_info_dict = {}
    docking_pose_summary_info_dict['ligand_smiles_string'] = ligand_smiles_string_array
    docking_pose_summary_info_dict['ligand_original_sdf_file_name'] = original_sdf_file_name_array
    docking_pose_summary_info_dict['ligand_docked_sdf_file_name'] = docked_sdf_file_name_array
    docking_pose_summary_info_dict['binding_free_energy'] = binding_free_energy_array
    docking_pose_summary_info_dict['ligand_efficiency'] = ligand_efficiency_array
    docking_pose_summary_info_dict['conformer_energy'] = docked_conf_energy_array
    docking_pose_summary_info_dict['cluster_size'] = cluster_size_array

    docking_pose_summary_info_df = pd.DataFrame(docking_pose_summary_info_dict)
    docking_pose_summary_info_df_proxy_list[docked_file_idx] = docking_pose_summary_info_df

    return True

class AutoDockGPUParsing(object):
    def __init__(self,
                 ligand_original_sdf_file_name_list,
                 ligand_docked_dlg_file_name_list,
                 n_cpu=16,
                 working_dir_name='.'):

        self.ligand_original_sdf_file_name_list = ligand_original_sdf_file_name_list
        self.ligand_docked_dlg_file_name_list = ligand_docked_dlg_file_name_list
        self.n_cpu = n_cpu
        self.num_docked_files = len(self.ligand_docked_dlg_file_name_list)
        self.working_dir_name = os.path.abspath(working_dir_name)

    def run_autodock_parsing(self):
        manager = mp.Manager()
        docking_pose_summary_info_df_proxy_list = manager.list()
        docking_pose_summary_info_df_proxy_list.extend([None] * self.num_docked_files)
        autodock_parsing_results_list = [None] * self.num_docked_files
        autodock_parsing_pool = Pool(processes=self.n_cpu)

        for docked_file_idx in range(self.num_docked_files):
            reference_sdf_file_name = self.ligand_original_sdf_file_name_list[docked_file_idx]
            dlg_file_name = self.ligand_docked_dlg_file_name_list[docked_file_idx]
            autodock_parsing_results = autodock_parsing_pool.apply_async(autodock_parsing_process,
                                                                         args=(reference_sdf_file_name,
                                                                               dlg_file_name,
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
