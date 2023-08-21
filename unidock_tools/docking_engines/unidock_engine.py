import os

class UniDockEngine(object):
    def __init__(self,
                 protein_pdbqt_file_name,
                 ligand_torsion_tree_file_name_list,
                 target_center,
                 box_size=(22.5, 22.5, 22.5),
                 template_docking=False,
                 working_dir_name='.'):

        self.protein_pdbqt_file_name = protein_pdbqt_file_name
        self.ligand_torsion_tree_file_name_list = ligand_torsion_tree_file_name_list
        self.num_conformations = len(self.ligand_torsion_tree_file_name_list)

        self.target_center = target_center
        self.box_size = box_size
        self.template_docking = template_docking
        self.working_dir_name = os.path.abspath(working_dir_name)
        self.ligand_batch_list_file_name = os.path.join(self.working_dir_name, 'ligand_batch_list.dat')
        self.__prepare_ligand_batch_file__(self.ligand_torsion_tree_file_name_list, self.ligand_batch_list_file_name)

        self.ligand_docked_torsion_tree_file_name_list = [None] * self.num_conformations
        for conf_idx in range(self.num_conformations):
            ligand_torsion_tree_file_name = self.ligand_torsion_tree_file_name_list[conf_idx]
            ligand_torsion_tree_base_file_name = os.path.basename(ligand_torsion_tree_file_name)
            ligand_torsion_tree_base_file_name_split_list = ligand_torsion_tree_base_file_name.split('.')
            ligand_output_file_name_base_prefix = ligand_torsion_tree_base_file_name_split_list[0]

            if ligand_torsion_tree_base_file_name_split_list[1] == 'pdbqt':
                ligand_docked_torsion_tree_file_name = os.path.join(self.working_dir_name, ligand_output_file_name_base_prefix + '_out.pdbqt')
            elif ligand_torsion_tree_base_file_name_split_list[1] == 'sdf':
                ligand_docked_torsion_tree_file_name = os.path.join(self.working_dir_name, ligand_output_file_name_base_prefix + '_out.sdf')

            self.ligand_docked_torsion_tree_file_name_list[conf_idx] = ligand_docked_torsion_tree_file_name

    def __prepare_ligand_batch_file__(self, ligand_torsion_tree_file_name_list, ligand_batch_list_file_name):
        with open(ligand_batch_list_file_name, 'w') as f:
            for ligand_torsion_tree_file_name in ligand_torsion_tree_file_name_list:
                f.write(os.path.abspath(ligand_torsion_tree_file_name))
                f.write('\n')

    def __perform_docking__(self,
                            protein_pdbqt_file_name,
                            ligand_batch_list_file_name,
                            target_center,
                            box_size,
                            template_docking,
                            working_dir_name):

        unidock_command = f'unidock --receptor {protein_pdbqt_file_name} --ligand_index {ligand_batch_list_file_name} --dir {working_dir_name} --search_mode fast --center_x {target_center[0]} --center_y {target_center[1]} --center_z {target_center[2]} --size_x {box_size[0]} --size_y {box_size[1]} --size_z {box_size[2]} --keep_nonpolar_H'

        if template_docking:
            unidock_command += ' --multi_bias'

        os.system(unidock_command)

    def run_batch_docking(self):
        self.__perform_docking__(self.protein_pdbqt_file_name,
                                 self.ligand_batch_list_file_name,
                                 self.target_center,
                                 self.box_size,
                                 self.template_docking,
                                 self.working_dir_name)
