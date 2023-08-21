import os

class WatvinaEngine(object):
    def __init__(self,
                 protein_pdbqt_file_name,
                 ligand_torsion_tree_file_name_list,
                 target_center,
                 box_size=(22.5, 22.5, 22.5),
                 template_docking=False,
                 n_cpu=12,
                 num_modes=10,
                 working_dir_name='.'):

        self.protein_pdbqt_file_name = protein_pdbqt_file_name
        self.ligand_torsion_tree_file_name_list = ligand_torsion_tree_file_name_list
        self.num_conformations = len(self.ligand_torsion_tree_file_name_list)

        self.target_center = target_center
        self.box_size = box_size
        self.template_docking = template_docking
        self.n_cpu = n_cpu
        self.num_modes = num_modes
        self.working_dir_name = os.path.abspath(working_dir_name)

        self.ligand_docked_torsion_tree_file_name_list = [None] * self.num_conformations
        for conf_idx in range(self.num_conformations):
            ligand_torsion_tree_file_name = self.ligand_torsion_tree_file_name_list[conf_idx]
            ligand_torsion_tree_base_file_name = os.path.basename(ligand_torsion_tree_file_name)
            ligand_torsion_tree_base_file_name_split_list = ligand_torsion_tree_base_file_name.split('.')
            ligand_output_file_name_base_prefix = ligand_torsion_tree_base_file_name_split_list[0]
            ligand_docked_torsion_tree_file_name = os.path.join(self.working_dir_name, ligand_output_file_name_base_prefix + '_out.pdbqt')

            self.ligand_docked_torsion_tree_file_name_list[conf_idx] = ligand_docked_torsion_tree_file_name

    def __perform_docking__(self,
                            protein_pdbqt_file_name,
                            ligand_torsion_tree_file_name_list,
                            ligand_docked_torsion_tree_file_name_list,
                            target_center,
                            box_size,
                            template_docking,
                            n_cpu,
                            num_modes):

        num_conformations = len(ligand_torsion_tree_file_name_list)
        for conf_idx in range(num_conformations):
            ligand_torsion_tree_file_name = ligand_torsion_tree_file_name_list[conf_idx]
            ligand_docked_torsion_tree_file_name = ligand_docked_torsion_tree_file_name_list[conf_idx]
            watvina_command = f'watvina --receptor {protein_pdbqt_file_name} --ligand {ligand_torsion_tree_file_name} --out {ligand_docked_torsion_tree_file_name} --cpu {n_cpu} --num_modes {num_modes} --exhaustiveness {n_cpu} --population 8 --ga_search 0 --center_x {target_center[0]} --center_y {target_center[1]} --center_z {target_center[2]} --size_x {box_size[0]} --size_y {box_size[1]} --size_z {box_size[2]}'

            if template_docking:
                watvina_command += ' --tramplitude 0'

            os.system(watvina_command)

    def run_batch_docking(self):
        self.__perform_docking__(self.protein_pdbqt_file_name,
                                 self.ligand_torsion_tree_file_name_list,
                                 self.ligand_docked_torsion_tree_file_name_list,
                                 self.target_center,
                                 self.box_size,
                                 self.template_docking,
                                 self.n_cpu,
                                 self.num_modes)
