import numpy as np


def save_point_cloud(xyz, color, file_path):
    assert xyz.shape[0] == color.shape[0]
    assert xyz.shape[1] == color.shape[1] == 3
    ply_file = open(file_path, 'w')
    ply_file.write('ply\n')
    ply_file.write('format ascii 1.0\n')
    ply_file.write('element vertex {}\n'.format(xyz.shape[0]))
    ply_file.write('property float x\n')
    ply_file.write('property float y\n')
    ply_file.write('property float z\n')
    ply_file.write('property uchar red\n')
    ply_file.write('property uchar green\n')
    ply_file.write('property uchar blue\n')
    ply_file.write('end_header\n')
    for i in range(xyz.shape[0]):
        ply_file.write('{:.3f} {:.3f} {:.3f} {} {} {}\n'.format(
                                xyz[i,0], xyz[i,1], xyz[i,2],
                                color[i,0], color[i,1], color[i,2]
                            )
        )