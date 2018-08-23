import os
import sys
import h5py
import argparse
import numpy as np
import car_models

from scale_off_no_align import Mesh
import scipy.io
from scipy.io import loadmat

skimage = None
mcubes = None

def read_hdf5(file, key = 'tensor'):
    """
    Read a tensor, i.e. numpy array, from HDF5.

    :param file: path to file to read
    :type file: str
    :param key: key to read
    :type key: str
    :return: tensor
    :rtype: numpy.ndarray
    """

    assert os.path.exists(file), 'file %s not found' % file

    h5f = h5py.File(file, 'r')
    tensor = h5f[key][()]
    h5f.close()

    return tensor

def write_off(file, vertices, faces):
    """
    Writes the given vertices and faces to OFF.

    :param vertices: vertices as tuples of (x, y, z) coordinates
    :type vertices: [(float)]
    :param faces: faces as tuples of (num_vertices, vertex_id_1, vertex_id_2, ...)
    :type faces: [(int)]
    """

    num_vertices = len(vertices)
    num_faces = len(faces)

    assert num_vertices > 0
    assert num_faces > 0

    with open(file, 'w') as fp:
        fp.write('OFF\n')
        fp.write(str(num_vertices) + ' ' + str(num_faces) + ' 0\n')

        for vertex in vertices:
            assert len(vertex) == 3, 'invalid vertex with %d dimensions found (%s)' % (len(vertex), file)
            fp.write(str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + '\n')

        for face in faces:
            assert len(face) == 3, 'only triangular faces supported (%s)' % (file)
            fp.write('3 ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + '\n')

        # add empty line to be sure
        fp.write('\n')

try:
    from skimage import measure

    def marching_cubes(tensor):
        """
        Perform marching cubes using mcubes.

        :param tensor: input volume
        :type tensor: numpy.ndarray
        :return: vertices, faces
        :rtype: numpy.ndarray, numpy.ndarray
        """

        vertices, faces, normals, values = measure.marching_cubes_lewiner(tensor.transpose(1, 0, 2), 0)
        return vertices, faces

    print('Using skimage\'s marching cubes implementation.')
except ImportError:
    print('==== Could not find skimage, import skimage.measure failed.')
    print('If you use skimage, make sure to call voxelize with -mode=corner.')

    try:
        sys.path[0] = '/home/rz1/.local/lib/python2.7/site-packages/'
        import mcubes
        print '==== mcubes imported.'
        def marching_cubes(tensor):
            """
            Perform marching cubes using mcubes.

            :param tensor: input volume
            :type tensor: numpy.ndarray
            :return: vertices, faces
            :rtype: numpy.ndarray, numpy.ndarray
            """

            return mcubes.marching_cubes(-tensor.transpose(1, 0, 2), 0)

        print('Using PyMCubes\'s marching cubes implementation.')
    except ImportError:
        print('Could not find PyMCubes, import mcubes failed.')
        print('You can use the version at https://github.com/davidstutz/PyMCubes.')
        print('If you use the voxel_centers branch, you can use -mode=center, otherwise use -mode=corner.')

if mcubes == None and measure == None:
    print('Could not find any marching cubes implementation; aborting.')
    exit(1);

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Peform marching cubes.')
    parser.add_argument('input', type=str, help='The input HDF5 file.')
    parser.add_argument('output', type=str, help='Output directory for OFF files.')

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print('Input file does not exist.')
        exit(1)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print('Created output directory.')
    else:
        print('Output directory exists; potentially overwriting contents.')

    tensor = read_hdf5(args.input)
    # car_scales = loadmat('car_scales.mat')
    # scales = car_scales['scales'].tolist()
    # scales = (1./scales[0][0], 1./scales[0][1], 1./scales[0][2])
    # scale = float(car_scales['scale'][0][0])

    if len(tensor.shape) < 4:
        tensor = np.expand_dims(tensor, axis=0)

    for n in range(tensor.shape[0]):
        print('Minimum and maximum value: %f and %f. ' % (np.min(tensor[n]), np.max(tensor[n])))
        vertices, faces = marching_cubes(tensor[n])
        off_file = '%s/%d.off' % (args.output, n)
        write_off(off_file, vertices, faces)
        print('Wrote %s.' % off_file)

        ## read original .off files for restoring
        mesh = Mesh.from_off(off_file)
        s_t = scipy.io.loadmat(off_file.replace('mc', 's_t').replace('.off', '.mat'))

        sizes_ori = (s_t['sizes'][0][0], s_t['sizes'][0][1], s_t['sizes'][0][2])
        scale = s_t['scale'][0][0]
        print scale

        mesh.scale((1./scale, 1./scale, 1./scale))
        mesh.translate((-0.5, -0.5, -0.5))

        min, max = mesh.extents()
        total_min = np.min(np.array(min))
        total_max = np.max(np.array(max))

        # Set the center (although this should usually be the origin already).
        centers = (
            (min[0] + max[0]) / 2,
            (min[1] + max[1]) / 2,
            (min[2] + max[2]) / 2
        )
        # Scales all dimensions equally.
        sizes = (
            total_max - total_min,
            total_max - total_min,
            total_max - total_min
        )
        translation = (
            -centers[0],
            -centers[1],
            -centers[2]
        )
        mesh.translate(translation)
        mesh.scale((sizes_ori[0]/sizes[0], sizes_ori[1]/sizes[1], sizes_ori[2]/sizes[2]))
        # mesh.scale(scales)

        # mesh.to_off(os.path.join(args.output+'_ori_scale', '%s.off' % car_models.models[n].name))
        mesh.to_off(off_file.replace('%d.off'%n, '%s_ori_scale.off'%car_models.models[n].name))


    print('Use MeshLab to visualize the created OFF files.')
