import os

class AutoDockGPUEngine(object):
    def __init__(self,
                 protein_grid_maps_fld_file_name,
                 ligand_pdbqt_file_name_list,
                 n_cpu=16,
                 cuda_device_idx_list=None,
                 num_docking_runs=10,
                 working_dir_name='.'):

        self.protein_grid_maps_fld_file_name = os.path.basename(protein_grid_maps_fld_file_name)
        self.ligand_pdbqt_file_name_list = ligand_pdbqt_file_name_list
        self.num_conformations = len(self.ligand_pdbqt_file_name_list)

        self.working_dir_name = os.path.abspath(working_dir_name)

        self.n_cpu = n_cpu
        os.environ['OMP_NUM_THREADS'] = str(self.n_cpu)

        if cuda_device_idx_list is None:
            self.cuda_device_idx_string = 'all'
        else:
            cuda_device_idx_string = ''
            for cuda_device_idx in cuda_device_idx_list:
                cuda_device_idx_string = cuda_device_idx_string + str(cuda_device_idx + 1) + ','

            self.cuda_device_idx_string = cuda_device_idx_string[:-1]

        ###############################################################################################################
        # FIX ME: this is an workaround for AutoDock-GPU C++ bug in reading file lines with fixed maximum string length.
        self.ligand_pdbqt_base_file_name_list = [None] * self.num_conformations
        self.ligand_docking_output_prefix_list = [None] * self.num_conformations
        self.ligand_dlg_file_name_list = [None] * self.num_conformations

        for conf_idx in range(self.num_conformations):
            current_ligand_pdbqt_file_name = self.ligand_pdbqt_file_name_list[conf_idx]
            current_ligand_pdbqt_base_file_name = os.path.basename(current_ligand_pdbqt_file_name)
            current_ligand_pdbqt_source_file_name = os.path.abspath(current_ligand_pdbqt_file_name)
            current_ligand_pdbqt_destination_file_name = os.path.join(self.working_dir_name, current_ligand_pdbqt_base_file_name)
            current_ligand_docking_output_prefix = current_ligand_pdbqt_base_file_name.split('.')[0]
            current_ligand_dlg_file_name = os.path.join(self.working_dir_name, current_ligand_docking_output_prefix + '.dlg')

            current_ligand_pdbqt_source_dir_name = os.path.dirname(current_ligand_pdbqt_source_file_name)
            if self.working_dir_name != current_ligand_pdbqt_source_dir_name:
                os.symlink(current_ligand_pdbqt_source_file_name, current_ligand_pdbqt_destination_file_name)

            self.ligand_pdbqt_base_file_name_list[conf_idx] = current_ligand_pdbqt_base_file_name
            self.ligand_docking_output_prefix_list[conf_idx] = current_ligand_docking_output_prefix
            self.ligand_dlg_file_name_list[conf_idx] = current_ligand_dlg_file_name

        self.ligand_batch_file_name = os.path.join(self.working_dir_name, 'ligand_conf_batch.dat')
        self.__prepare_ligand_batch_file__(self.protein_grid_maps_fld_file_name,
                                           self.ligand_pdbqt_base_file_name_list,
                                           self.ligand_docking_output_prefix_list,
                                           self.ligand_batch_file_name)
        ###############################################################################################################

        self.num_docking_runs = num_docking_runs

        protein_grid_maps_source_dir_name = os.path.dirname(os.path.abspath(protein_grid_maps_fld_file_name))
        if self.working_dir_name != protein_grid_maps_source_dir_name:
            protein_grid_maps_prefix = self.protein_grid_maps_fld_file_name.split('.')[0]
            protein_grid_maps_file_names = protein_grid_maps_source_dir_name + '/' + protein_grid_maps_prefix + '*map*'
            protein_grid_soft_link_command = 'cd {0}; ln -s {1} .; cd -'.format(self.working_dir_name, protein_grid_maps_file_names)
            os.system(protein_grid_soft_link_command)

    def __prepare_ligand_batch_file__(self, protein_grid_maps_fld_file_name, conf_pdbqt_file_name_list, output_prefix_list, ligand_batch_file_name):
        num_conformations = len(output_prefix_list)
        with open(ligand_batch_file_name, 'w') as ligand_batch_file:
            ligand_batch_file.write(protein_grid_maps_fld_file_name + '\n')
            for conf_idx in range(num_conformations):
                ligand_batch_file.write(conf_pdbqt_file_name_list[conf_idx] + '\n')
                ligand_batch_file.write(output_prefix_list[conf_idx] + '\n')

    def __perform_docking__(self, working_dir_name, ligand_batch_file_name, num_docking_runs, cuda_device_idx_string):
        ########################################################################################################################################################
        # FIX ME: the basename usage (should be abspath later) is an workaround for AutoDock-GPU C++ bug in reading file lines with fixed maximum string length.
        docking_batch_command = 'cd {0}; autodock_gpu_128wi -B {1} -n {2} -D {3}; cd -'.format(working_dir_name, os.path.basename(ligand_batch_file_name), num_docking_runs, cuda_device_idx_string)
        ########################################################################################################################################################
        os.system(docking_batch_command)

    def __check_docking_results__(self, ligand_dlg_file_name_list):
        for ligand_dlg_file_name in ligand_dlg_file_name_list:
            if not os.path.exists(ligand_dlg_file_name):
                raise Exception('Error in ' + ligand_dlg_file_name)
            else:
                continue

    def run_batch_docking(self):
        self.__perform_docking__(self.working_dir_name, self.ligand_batch_file_name, self.num_docking_runs, self.cuda_device_idx_string)
        self.__check_docking_results__(self.ligand_dlg_file_name_list)
