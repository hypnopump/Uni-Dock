import os
import sys

sys.path.append(os.environ['WATVINA'])
from watvina_wrapper import WATVina

class PyWatvinaEngine(object):
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

        watvina = WATVina(cpu=n_cpu)

        with open(protein_pdbqt_file_name, 'r') as protein_pdbqt_file:
            receptor_lines = protein_pdbqt_file.read()

        watvina.set_receptor_from_string(receptor_lines)
        watvina.set_watvina_weights()
        watvina.set_extra_constraints()
        watvina.set_grid_dims(center_x=target_center[0], center_y=target_center[1], center_z=target_center[2], size_x=box_size[0], size_y=box_size[1], size_z=box_size[2])
        watvina.set_precalculate_sf(prec_full_atomtypes=True)
        watvina.compute_watvina_maps()

        if template_docking:
            tramplitude = 0.0
        else:
            tramplitude = 1.0

        num_conformations = len(ligand_torsion_tree_file_name_list)
        for conf_idx in range(num_conformations):
            ligand_torsion_tree_file_name = ligand_torsion_tree_file_name_list[conf_idx]
            ligand_docked_torsion_tree_file_name = ligand_docked_torsion_tree_file_name_list[conf_idx]

            with open(ligand_torsion_tree_file_name, 'r') as ligand_torsion_tree_file:
                ligand_lines = ligand_torsion_tree_file.read()

            watvina.set_ligand_from_string(ligand_lines)
            watvina.global_search(exhaustiveness=n_cpu,
                                  n_poses=num_modes,
                                  population_size=8,
                                  ga_searching=0,
                                  tramplitude=tramplitude)

            watvina.write_poses(ligand_docked_torsion_tree_file_name, num_modes)

    def run_batch_docking(self):
        self.__perform_docking__(self.protein_pdbqt_file_name,
                                 self.ligand_torsion_tree_file_name_list,
                                 self.ligand_docked_torsion_tree_file_name_list,
                                 self.target_center,
                                 self.box_size,
                                 self.template_docking,
                                 self.n_cpu,
                                 self.num_modes)
