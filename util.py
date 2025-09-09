import numpy as np
from collections import namedtuple

XYZTuple = namedtuple('XYZTuple', 'x, y, z')
IRCTuple = namedtuple('IRCTuple', 'index, row, col')

def irc_to_xyz(coords_irc, origin_xyz, voxel_size_xyz, direction_a):
    cri = np.array(coords_irc)[::-1] # Convert from IRC to CRI format (Index is like Depth) for voxels
    origin = np.array(origin_xyz) # Origin of voxel (0, 0, 0) in XYZ coordinates
    voxel_size = np.array(voxel_size_xyz)
    
    coords_xyz = origin + (direction_a @ (cri * voxel_size))

    return XYZTuple(coords_xyz[0], coords_xyz[1], coords_xyz[2])


def xyz_to_irc(coords_xyz, origin_xyz, voxel_size_xyz, direction_a):
    coords = np.array(coords_xyz)
    origin = np.array(origin_xyz) # Origin of voxel (0, 0, 0) in XYZ coordinates
    voxel_size = np.array(voxel_size_xyz)
    
    # Just the reverse of the IRC to XYZ calculation
    cri_coords = np.round(((coords - origin) @ np.linalg.inv(direction_a)) / voxel_size) # Round since voxel CT Scan self.ct_HU_voxels needs an integer

    return IRCTuple(int(cri_coords[2]), int(cri_coords[1]), int(cri_coords[0]))



