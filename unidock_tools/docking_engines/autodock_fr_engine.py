import os

class AutoDockFREngine(object):
    def __init__(self,
                 protein_grid_maps_fld_file_name,
                 ligand_pdbqt_file_name_list,
                 n_cpu=16,
                 num_docking_runs=50,
                 working_dir_name='.'):

        self.protein_grid_maps_fld_file_name = protein_grid_maps_fld_file_name
        self.ligand_pdbqt_file_name_list = ligand_pdbqt_file_name_list
        self.num_conformations = len(self.ligand_pdbqt_file_name_list)

        self.n_cpu = n_cpu
        os.environ['OMP_NUM_THREADS'] = str(self.n_cpu)
        self.num_docking_runs = num_docking_runs

        self.working_dir_name = os.path.abspath(working_dir_name)

        ###############################################################################################################
        ## Prepare autogrid zip file
#        self.autogrid_dir_name = os.path.dirname(protein_grid_maps_fld_file_name)
#        self.autogrid_dir_base_name = os.path.basename(self.autogrid_dir_name)
#        self.autogrid_parent_dir_name = os.path.dirname(self.autogrid_dir_name)
#        self.autogrid_zip_file_name = os.path.join(self.working_dir_name, os.path.basename(self.autogrid_dir_base_name) + '.zip')
#        protein_grid_zip_command = 'cd {0}; zip -r {1} {2}; cd -'.format(self.autogrid_parent_dir_name,
#                                                                         self.autogrid_zip_file_name,
#                                                                         self.autogrid_dir_base_name)
#
#        os.system(protein_grid_zip_command)
        ###############################################################################################################

        self.ligand_docked_dlg_file_name_list = [None] * self.num_conformations
        self.ligand_docked_pdbqt_file_name_list = [None] * self.num_conformations
        for conf_idx in range(self.num_conformations):
            ligand_pdbqt_file_name = self.ligand_pdbqt_file_name_list[conf_idx]
            ligand_pdbqt_base_file_name = os.path.basename(ligand_pdbqt_file_name)
            ligand_output_file_name_base_prefix = ligand_pdbqt_base_file_name.split('.')[0]
            ligand_docked_dlg_file_name = os.path.join(self.working_dir_name, ligand_output_file_name_base_prefix + '_adfr_summary.dlg')
            ligand_docked_pdbqt_file_name = os.path.join(self.working_dir_name, ligand_output_file_name_base_prefix + '_adfr_out.pdbqt')
            self.ligand_docked_dlg_file_name_list[conf_idx] = ligand_docked_dlg_file_name
            self.ligand_docked_pdbqt_file_name_list[conf_idx] = ligand_docked_pdbqt_file_name

    def __perform_docking__(self,
                            protein_grid_maps_fld_file_name,
                            ligand_pdbqt_file_name_list,
                            n_cpu,
                            num_docking_runs,
                            working_dir_name):

        for ligand_pdbqt_file_name in ligand_pdbqt_file_name_list:
            adfr_command = 'cd {0}; adfr -l {1} -t {2} -J adfr -c {3} -n {4} -C 1 2 3 -f -O; cd -'.format(working_dir_name,
                                                                                                          ligand_pdbqt_file_name,
                                                                                                          protein_grid_maps_fld_file_name,
                                                                                                          n_cpu,
                                                                                                          num_docking_runs)

            os.system(adfr_command)

    def run_batch_docking(self):
        self.__perform_docking__(self.protein_grid_maps_fld_file_name,
                                 self.ligand_pdbqt_file_name_list,
                                 self.n_cpu,
                                 self.num_docking_runs,
                                 self.working_dir_name)
