import os
from shutil import rmtree

from unidock_tools.docking_engines.autogrid_runner import AutoGridRunner
from unidock_tools.docking_engines.ligand_conformation_preprocessor import LigandConformationPreprocessor
from unidock_tools.docking_engines.torsion_tree_generator import TorsionTreeGenerator
from unidock_tools.docking_engines.autodock_gpu_engine import AutoDockGPUEngine
from unidock_tools.docking_engines.autodock_fr_engine import AutoDockFREngine
from unidock_tools.docking_engines.unidock_engine import UniDockEngine
from unidock_tools.docking_engines.watvina_engine import WatvinaEngine
from unidock_tools.docking_engines.autodock_gpu_parsing import AutoDockGPUParsing
from unidock_tools.docking_engines.autodock_fr_parsing import AutoDockFRParsing
from unidock_tools.docking_engines.unidock_parsing import UniDockParsing
from unidock_tools.docking_engines.unidock_parsing_sdf import UniDockParsingSDF
from unidock_tools.docking_engines.watvina_parsing import WatVinaParsing

class UniDockRunner(object):
    def __init__(self,
                 ligand_sdf_file_name_list,
                 protein_pdb_file_name_list,
                 protein_conf_name_list=None,
                 kept_ligand_resname_nested_list=None,
                 target_center_list=None,
                 box_size=(22.5, 22.5, 22.5),
                 covalent_residue_atom_info_nested_list=None,
                 docking_engine='unidock',
                 docking_method='free_docking',
                 reference_sdf_file_name=None,
                 generate_torsion_tree_sdf=False,
                 n_cpu=16,
                 cuda_device_idx_list=None,
                 num_docking_runs=10,
                 preprocess_ligand_sdf_files=False,
                 refine_amide_torsions=False,
                 working_dir_name='.'):

        self.ligand_sdf_file_name_list = ligand_sdf_file_name_list
        self.protein_pdb_file_name_list = protein_pdb_file_name_list
        self.num_protein_conformations = len(self.protein_pdb_file_name_list)

        if protein_conf_name_list is None:
            self.protein_conf_name_list = [None] * self.num_protein_conformations
            for protein_conf_idx in range(self.num_protein_conformations):
                self.protein_conf_name_list[protein_conf_idx] = 'protein_conf_' + str(protein_conf_idx)
        else:
            self.protein_conf_name_list = protein_conf_name_list

        self.kept_ligand_resname_nested_list = kept_ligand_resname_nested_list
        self.target_center_list = target_center_list
        self.box_size = box_size
        self.covalent_residue_atom_info_nested_list = covalent_residue_atom_info_nested_list

        if docking_engine not in ['ad-gpu', 'adfr', 'unidock', 'watvina']:
            raise ValueError('Specified docking enigne not supported.')
        else:
            self.docking_engine = docking_engine

        if docking_method == 'free_docking':
            self.covalent_ligand = False
            self.template_docking = False
            self.reference_sdf_file_name = None
        elif docking_method == 'covalent_docking':
            self.covalent_ligand = True
            self.template_docking = False
            self.reference_sdf_file_name = None
        elif docking_method == 'template_docking':
            self.covalent_ligand = False
            self.template_docking = True

            if reference_sdf_file_name:
                self.reference_sdf_file_name = reference_sdf_file_name
            else:
                raise ValueError('Template docking method specified but reference sdf file is missing.')
        else:
            raise ValueError('Unknown docking type specified.')

        self.docking_method = docking_method

        self.generate_torsion_tree_sdf = generate_torsion_tree_sdf

        self.n_cpu = n_cpu
        self.cuda_device_idx_list = cuda_device_idx_list
        self.num_docking_runs = num_docking_runs

        self.preprocess_ligand_sdf_files = preprocess_ligand_sdf_files
        self.refine_amide_torsions = refine_amide_torsions

        if self.refine_amide_torsions:
            self.preprocess_ligand_sdf_files = True

        self.protein_pdb_file_name_list = protein_pdb_file_name_list

        self.root_working_dir_name = os.path.abspath(working_dir_name)
        self.receptor_grids_working_dir_name = os.path.join(self.root_working_dir_name, 'receptor_grids')
        self.conformation_refinement_working_dir_name = os.path.join(self.root_working_dir_name, 'ligand_conformation_preprocess')
        self.torsion_tree_working_dir_name = os.path.join(self.root_working_dir_name, 'ligand_torsion_tree')
        self.batch_docking_working_dir_name = os.path.join(self.root_working_dir_name, 'batch_docking')
        self.docking_pose_parsing_working_dir_name = os.path.join(self.root_working_dir_name, 'docking_pose_sdf')
        self.pose_penalty_scoring_working_dir_name = os.path.join(self.root_working_dir_name, 'pose_penalty_scoring')

        if os.path.isdir(self.receptor_grids_working_dir_name):
            rmtree(self.receptor_grids_working_dir_name, ignore_errors=True)
            os.mkdir(self.receptor_grids_working_dir_name)
        else:
            os.mkdir(self.receptor_grids_working_dir_name)

        if os.path.isdir(self.conformation_refinement_working_dir_name):
            rmtree(self.conformation_refinement_working_dir_name, ignore_errors=True)
            os.mkdir(self.conformation_refinement_working_dir_name)
        else:
            os.mkdir(self.conformation_refinement_working_dir_name)

        if os.path.isdir(self.torsion_tree_working_dir_name):
            rmtree(self.torsion_tree_working_dir_name, ignore_errors=True)
            os.mkdir(self.torsion_tree_working_dir_name)
        else:
            os.mkdir(self.torsion_tree_working_dir_name)

        if os.path.isdir(self.batch_docking_working_dir_name):
            rmtree(self.batch_docking_working_dir_name, ignore_errors=True)
            os.mkdir(self.batch_docking_working_dir_name)
        else:
            os.mkdir(self.batch_docking_working_dir_name)

        if os.path.isdir(self.docking_pose_parsing_working_dir_name):
            rmtree(self.docking_pose_parsing_working_dir_name, ignore_errors=True)
            os.mkdir(self.docking_pose_parsing_working_dir_name)
        else:
            os.mkdir(self.docking_pose_parsing_working_dir_name)

        if os.path.isdir(self.pose_penalty_scoring_working_dir_name):
            rmtree(self.pose_penalty_scoring_working_dir_name, ignore_errors=True)
            os.mkdir(self.pose_penalty_scoring_working_dir_name)
        else:
            os.mkdir(self.pose_penalty_scoring_working_dir_name)

        self.nested_extended_ligand_dlg_file_name_list = [None] * self.num_protein_conformations
        self.nested_extended_ligand_torsion_tree_file_name_list = [None] * self.num_protein_conformations
        self.docking_pose_summary_info_df_list = [None] * self.num_protein_conformations
        self.scored_docking_pose_summary_info_df_list = [None] * self.num_protein_conformations

    def run(self):
        if self.docking_engine in ['adfr', 'ad-gpu']:
            generate_ad4_grids = True
        else:
            generate_ad4_grids = False

        autogrid_runner = AutoGridRunner(protein_pdb_file_name_list=self.protein_pdb_file_name_list,
                                         protein_conf_name_list=self.protein_conf_name_list,
                                         kept_ligand_resname_nested_list=self.kept_ligand_resname_nested_list,
                                         target_center_list=self.target_center_list,
                                         box_size=self.box_size,
                                         covalent_residue_atom_info_nested_list=self.covalent_residue_atom_info_nested_list,
                                         generate_ad4_grids=generate_ad4_grids,
                                         working_dir_name=self.receptor_grids_working_dir_name)

        autogrid_runner.run()

        self.protein_pdbqt_file_name_list = autogrid_runner.protein_info_df.loc[:, 'protein_pdbqt_file_name'].values.tolist()
        self.protein_grid_maps_fld_file_name_list = autogrid_runner.protein_info_df.loc[:, 'protein_grid_maps_fld_file_name'].values.tolist()

        if self.preprocess_ligand_sdf_files:
            ligand_conformation_refinement = LigandConformationPreprocessor(self.ligand_sdf_file_name_list,
                                                                            refine_amide_torsions=self.refine_amide_torsions,
                                                                            covalent_ligand=self.covalent_ligand,
                                                                            n_cpu=self.n_cpu,
                                                                            working_dir_name=self.conformation_refinement_working_dir_name)

            ligand_conformation_refinement.generate_refined_ligand_sdf_files()
            self.extended_ligand_sdf_file_name_list = ligand_conformation_refinement.extended_ligand_sdf_file_name_list
        else:
            self.extended_ligand_sdf_file_name_list = self.ligand_sdf_file_name_list

        torsion_tree_generator = TorsionTreeGenerator(self.extended_ligand_sdf_file_name_list,
                                                      covalent_ligand=self.covalent_ligand,
                                                      template_docking=self.template_docking,
                                                      reference_sdf_file_name=self.reference_sdf_file_name,
                                                      generate_torsion_tree_sdf=self.generate_torsion_tree_sdf,
                                                      n_cpu=self.n_cpu,
                                                      working_dir_name=self.torsion_tree_working_dir_name)

        torsion_tree_generator.generate_ligand_torsion_tree_files()

        for protein_conf_idx in range(self.num_protein_conformations):
            current_protein_pdbqt_file_name = self.protein_pdbqt_file_name_list[protein_conf_idx]
            current_protein_docking_grid_file_name = self.protein_grid_maps_fld_file_name_list[protein_conf_idx]
            current_target_center = self.target_center_list[protein_conf_idx]
            current_protein_conf_name = self.protein_conf_name_list[protein_conf_idx]
            current_batch_docking_working_dir_name = os.path.join(self.batch_docking_working_dir_name, current_protein_conf_name)
            os.mkdir(current_batch_docking_working_dir_name)

            if self.covalent_ligand:
                if self.docking_engine not in ['adfr']:
                    raise ValueError('Specified docking enigne not supported for covalent docking.')

                autodock_engine = AutoDockFREngine(current_protein_docking_grid_file_name,
                                                   torsion_tree_generator.ligand_pdbqt_file_name_list,
                                                   n_cpu=self.n_cpu,
                                                   num_docking_runs=self.num_docking_runs,
                                                   working_dir_name=current_batch_docking_working_dir_name)

                autodock_engine.run_batch_docking()
                self.nested_extended_ligand_dlg_file_name_list[protein_conf_idx] = autodock_engine.ligand_docked_dlg_file_name_list
                self.nested_extended_ligand_torsion_tree_file_name_list[protein_conf_idx] = autodock_engine.ligand_docked_pdbqt_file_name_list

            elif self.template_docking:
                if self.docking_engine not in ['unidock', 'watvina']:
                    raise ValueError('Specified docking enigne not supported for template docking.')

                if self.docking_engine == 'watvina':
                    autodock_engine = WatvinaEngine(current_protein_pdbqt_file_name,
                                                    torsion_tree_generator.ligand_pdbqt_file_name_list,
                                                    current_target_center,
                                                    box_size=self.box_size,
                                                    template_docking=self.template_docking,
                                                    n_cpu=self.n_cpu,
                                                    num_modes=self.num_docking_runs,
                                                    working_dir_name=current_batch_docking_working_dir_name)

                    autodock_engine.run_batch_docking()
                    self.nested_extended_ligand_torsion_tree_file_name_list[protein_conf_idx] = autodock_engine.ligand_docked_torsion_tree_file_name_list

                else:
                    if self.generate_torsion_tree_sdf:
                        current_torsion_tree_file_name_list = torsion_tree_generator.ligand_torsion_tree_sdf_file_name_list
                    else:
                        current_torsion_tree_file_name_list = torsion_tree_generator.ligand_pdbqt_file_name_list

                    autodock_engine = UniDockEngine(current_protein_pdbqt_file_name,
                                                    current_torsion_tree_file_name_list,
                                                    current_target_center,
                                                    box_size=self.box_size,
                                                    template_docking=self.template_docking,
                                                    working_dir_name=current_batch_docking_working_dir_name)

                    autodock_engine.run_batch_docking()
                    self.nested_extended_ligand_torsion_tree_file_name_list[protein_conf_idx] = autodock_engine.ligand_docked_torsion_tree_file_name_list

            else:
                if self.docking_engine not in ['ad-gpu', 'watvina', 'unidock']:
                    raise ValueError('Specified docking enigne not supported for free docking.')

                if self.docking_engine == 'ad-gpu':
                    autodock_engine = AutoDockGPUEngine(current_protein_docking_grid_file_name,
                                                        torsion_tree_generator.ligand_pdbqt_file_name_list,
                                                        n_cpu=self.n_cpu,
                                                        cuda_device_idx_list=self.cuda_device_idx_list,
                                                        num_docking_runs=self.num_docking_runs,
                                                        working_dir_name=current_batch_docking_working_dir_name)

                    autodock_engine.run_batch_docking()
                    self.nested_extended_ligand_dlg_file_name_list[protein_conf_idx] = autodock_engine.ligand_dlg_file_name_list

                elif self.docking_engine == 'watvina':
                    autodock_engine = WatvinaEngine(current_protein_pdbqt_file_name,
                                                    torsion_tree_generator.ligand_pdbqt_file_name_list,
                                                    current_target_center,
                                                    box_size=self.box_size,
                                                    template_docking=self.template_docking,
                                                    n_cpu=self.n_cpu,
                                                    num_modes=self.num_docking_runs,
                                                    working_dir_name=current_batch_docking_working_dir_name)

                    autodock_engine.run_batch_docking()
                    self.nested_extended_ligand_torsion_tree_file_name_list[protein_conf_idx] = autodock_engine.ligand_docked_torsion_tree_file_name_list

                else:
                    if self.generate_torsion_tree_sdf:
                        current_torsion_tree_file_name_list = torsion_tree_generator.ligand_torsion_tree_sdf_file_name_list
                    else:
                        current_torsion_tree_file_name_list = torsion_tree_generator.ligand_pdbqt_file_name_list

                    autodock_engine = UniDockEngine(current_protein_pdbqt_file_name,
                                                    current_torsion_tree_file_name_list,
                                                    current_target_center,
                                                    box_size=self.box_size,
                                                    template_docking=self.template_docking,
                                                    working_dir_name=current_batch_docking_working_dir_name)

                    autodock_engine.run_batch_docking()
                    self.nested_extended_ligand_torsion_tree_file_name_list[protein_conf_idx] = autodock_engine.ligand_docked_torsion_tree_file_name_list

        for protein_conf_idx in range(self.num_protein_conformations):
            current_protein_conf_name = self.protein_conf_name_list[protein_conf_idx]
            current_docking_pose_parsing_working_dir_name = os.path.join(self.docking_pose_parsing_working_dir_name, current_protein_conf_name)
            os.mkdir(current_docking_pose_parsing_working_dir_name)
            current_extended_ligand_dlg_file_name_list = self.nested_extended_ligand_dlg_file_name_list[protein_conf_idx]
            current_extended_ligand_torsion_tree_file_name_list = self.nested_extended_ligand_torsion_tree_file_name_list[protein_conf_idx]

            if self.covalent_ligand:
                autodock_parsing = AutoDockFRParsing(self.extended_ligand_sdf_file_name_list,
                                                     current_extended_ligand_dlg_file_name_list,
                                                     current_extended_ligand_torsion_tree_file_name_list,
                                                     n_cpu=self.n_cpu,
                                                     working_dir_name=current_docking_pose_parsing_working_dir_name)

                autodock_parsing.run_autodock_parsing()
                self.docking_pose_summary_info_df_list[protein_conf_idx] = autodock_parsing.docking_pose_summary_info_df

            elif self.template_docking:
                if self.docking_engine == 'watvina':
                    autodock_parsing = WatVinaParsing(self.extended_ligand_sdf_file_name_list,
                                                     current_extended_ligand_torsion_tree_file_name_list,
                                                     n_cpu=self.n_cpu,
                                                     working_dir_name=current_docking_pose_parsing_working_dir_name)

                    autodock_parsing.run_watvina_parsing()
                    self.docking_pose_summary_info_df_list[protein_conf_idx] = autodock_parsing.docking_pose_summary_info_df

                else:
                    if self.generate_torsion_tree_sdf:
                        unified_unidock_parsing_class = UniDockParsingSDF
                    else:
                        unified_unidock_parsing_class = UniDockParsing

                    autodock_parsing = unified_unidock_parsing_class(self.extended_ligand_sdf_file_name_list,
                                                                    current_extended_ligand_torsion_tree_file_name_list,
                                                                    n_cpu=self.n_cpu,
                                                                    working_dir_name=current_docking_pose_parsing_working_dir_name)

                    autodock_parsing.run_unidock_parsing()
                    self.docking_pose_summary_info_df_list[protein_conf_idx] = autodock_parsing.docking_pose_summary_info_df

            else:
                if self.docking_engine == 'ad-gpu':
                    autodock_parsing = AutoDockGPUParsing(self.extended_ligand_sdf_file_name_list,
                                                          current_extended_ligand_dlg_file_name_list,
                                                          n_cpu=self.n_cpu,
                                                          working_dir_name=current_docking_pose_parsing_working_dir_name)

                    autodock_parsing.run_autodock_parsing()
                    self.docking_pose_summary_info_df_list[protein_conf_idx] = autodock_parsing.docking_pose_summary_info_df

                elif self.docking_engine == 'watvina':
                    autodock_parsing = WatVinaParsing(self.extended_ligand_sdf_file_name_list,
                                                     current_extended_ligand_torsion_tree_file_name_list,
                                                     n_cpu=self.n_cpu,
                                                     working_dir_name=current_docking_pose_parsing_working_dir_name)

                    autodock_parsing.run_watvina_parsing()
                    self.docking_pose_summary_info_df_list[protein_conf_idx] = autodock_parsing.docking_pose_summary_info_df

                else:
                    if self.generate_torsion_tree_sdf:
                        unified_unidock_parsing_class = UniDockParsingSDF
                    else:
                        unified_unidock_parsing_class = UniDockParsing

                    autodock_parsing = unified_unidock_parsing_class(self.extended_ligand_sdf_file_name_list,
                                                                     current_extended_ligand_torsion_tree_file_name_list,
                                                                     n_cpu=self.n_cpu,
                                                                     working_dir_name=current_docking_pose_parsing_working_dir_name)

                    autodock_parsing.run_unidock_parsing()
                    self.docking_pose_summary_info_df_list[protein_conf_idx] = autodock_parsing.docking_pose_summary_info_df

        return self.docking_pose_summary_info_df_list

def main():
    parser = argparse.ArgumentParser(description='DP unified docking runner')
    pass
