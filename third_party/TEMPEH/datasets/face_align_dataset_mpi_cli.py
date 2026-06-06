# python datasets/face_align_dataset_mpi_cli.py
from datasets.face_align_dataset_mpi import FaceAlignDatasetMPI       
from utils.mesh_sampling import MeshSampler
from psbody.mesh import Mesh
import pudb
from visualizations import data_visualizations
import matplotlib.pyplot as plt
template_fname = "./data/template/sampling_template.obj"
template_mesh = Mesh(filename=template_fname)
mesh_sampler = MeshSampler(template_mesh, [5023], keep_boundary_adjacent=True) 
dataset_train = FaceAlignDatasetMPI(   
                                    data_list_fname="/is/cluster/mprinzler/projects/gintern/TEMPEH/data/training_data/one_subj__all_seq_frames_per_seq_40_head_rot_120_train.json",
                                    dataset_root_dir='./data', 
                                    image_dir="./data/training_data/downsampled_images_4",
                                    calibration_dir="./data/training_data/calibrations",
                                    scan_dir="./data/training_data/sampled_scan_points",
                                    registration_root_dir="./data/training_data/registrations",
                                    image_resize_factor=8,
                                    mesh_sampler=mesh_sampler,            
                                    scan_vertex_count=20_000,
                                    brightness_sigma=0.33,
                                    load_stereo_images=True,
                                    load_color_images=False,
                                    image_file_ext='png')

sample = dataset_train[0]
fig = data_visualizations.vis_projections(sample)
plt.savefig('demos/mpi_data_reprojections.jpg')

fig = data_visualizations.vis_scene_geometry(sample)
fig.write_html('demos/mpi_data_scene_geometry.html')

