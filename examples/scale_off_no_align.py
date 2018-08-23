import os
import sys
import math
import argparse
import numpy as np
from file_utils import *
import car_models
from scipy.io import savemat, loadmat

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
            assert face[0] == 3, 'only triangular faces supported (%s)' % file
            assert len(face) == 4, 'faces need to have 3 vertices, but found %d (%s)' % (len(face), file)

            for i in range(len(face)):
                assert face[i] >= 0 and face[i] < num_vertices, 'invalid vertex index %d (of %d vertices) (%s)' % (face[i], num_vertices, file)

                fp.write(str(face[i]))
                if i < len(face) - 1:
                    fp.write(' ')

            fp.write('\n')

        # add empty line to be sure
        fp.write('\n')

def read_off(file):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        # Fix for ModelNet bug were 'OFF' and the number of vertices and faces are
        # all in the first line.
        if len(lines[0]) > 3:
            assert lines[0][:3] == 'OFF' or lines[0][:3] == 'off', 'invalid OFF file %s' % file

            parts = lines[0][3:].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 1
        # This is the regular case!
        else:
            assert lines[0] == 'OFF' or lines[0] == 'off', 'invalid OFF file %s' % file

            parts = lines[1].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 2

        vertices = []
        for i in range(num_vertices):
            vertex = lines[start_index + i].split(' ')
            vertex = [float(point.strip()) for point in vertex if point != '']
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        for i in range(num_faces):
            # print len(lines), start_index, num_vertices, num_faces, i
            try:
                face = lines[start_index + num_vertices + i].split(' ')
            except IndexError:
                print(toRed('Error in reading faces. Aborted.'))
                return None, None
            # print face
            face = [index.strip() for index in face if index != '']

            # check to be sure
            for index in face:
                assert index != '', 'found empty vertex index: %s (%s)' % (lines[start_index + num_vertices + i], file)

            face = [int(index) for index in face]

            assert face[0] == len(face) - 1, 'face should have %d vertices but as %d (%s)' % (face[0], len(face) - 1, file)
            assert face[0] == 3, 'only triangular meshes supported (%s)' % file
            for index in face:
                assert index >= 0 and index < num_vertices, 'vertex %d (of %d vertices) does not exist (%s)' % (index, num_vertices, file)

            assert len(face) > 1

            faces.append(face)

        return vertices, faces

class Mesh:
    """
    Represents a mesh.
    """

    def __init__(self, vertices = [[]], faces = [[]]):
        """
        Construct a mesh from vertices and faces.

        :param vertices: list of vertices, or numpy array
        :type vertices: [[float]] or numpy.ndarray
        :param faces: list of faces or numpy array, i.e. the indices of the corresponding vertices per triangular face
        :type faces: [[int]] fo rnumpy.ndarray
        """

        self.vertices = np.array(vertices, dtype = float)
        """ (numpy.ndarray) Vertices. """

        self.faces = np.array(faces, dtype = int)
        """ (numpy.ndarray) Faces. """

        assert self.vertices.shape[1] == 3
        assert self.faces.shape[1] == 3

    def extents(self):
        """
        Get the extents.

        :return: (min_x, min_y, min_z), (max_x, max_y, max_z)
        :rtype: (float, float, float), (float, float, float)
        """

        min = [0]*3
        max = [0]*3

        for i in range(3):
            min[i] = np.min(self.vertices[:, i])
            max[i] = np.max(self.vertices[:, i])

        return tuple(min), tuple(max)

    def scale(self, scales):
        """
        Scale the mesh in all dimensions.

        :param scales: tuple of length 3 with scale for (x, y, z)
        :type scales: (float, float, float)
        """

        assert len(scales) == 3

        for i in range(3):
            self.vertices[:, i] *= scales[i]

    def translate(self, translation):
        """
        Translate the mesh.

        :param translation: translation as (x, y, z)
        :type translation: (float, float, float)
        """

        assert len(translation) == 3

        for i in range(3):
            self.vertices[:, i] += translation[i]

    @staticmethod
    def from_off(filepath):
        """
        Read a mesh from OFF.

        :param filepath: path to OFF file
        :type filepath: str
        :return: mesh
        :rtype: Mesh
        """

        vertices, faces = read_off(filepath)
        if vertices==None or faces==None:
            return None
        real_faces = []
        for face in faces:
            assert len(face) == 4
            real_faces.append([face[1], face[2], face[3]])

        return Mesh(vertices, real_faces)

    def to_off(self, filepath):
        """
        Write mesh to OFF.

        :param filepath: path to write file to
        :type filepath: str
        """

        faces = np.ones((self.faces.shape[0], 4), dtype = int)*3
        faces[:, 1:4] = self.faces[:, :]

        write_off(filepath, self.vertices.tolist(), faces.tolist())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert OFF to OBJ.')
    parser.add_argument('input', type=str, help='The input directory containing OFF files.')
    parser.add_argument('output', type=str, help='The output directory for OBJ files.')
    parser.add_argument('--padding', type=float, default=0.1, help='Padding on each side.')
    parser.add_argument('--height', type=int, default=32, help='Height to scale to.')
    parser.add_argument('--width', type=int, default=32, help='Width to scale to.')
    parser.add_argument('--depth', type=int, default=32, help='Depth to scale to.')
    bin_path = '/home/rz1/Documents/new_proj/mesh-voxelization/bin/'
    sys.path.append(bin_path)

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print('Input directory does not exist.')
        exit(1)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print('Created output directory.')
    else:
        print('Output directory exists; potentially overwriting contents.')

    # n = 0
    scale = max(args.height, args.width, args.depth)
    max_sizes = (0., 0., 0.)
    # translations = []

    # mat_path = 'car_scales.mat'

    # if os.path.exists(mat_path):
    #     car_scales = loadmat(mat_path)
    #     scales = car_scales['scales'].tolist()
    #     scales = (scales[0][0], scales[0][1], scales[0][2])
    #     scale = float(car_scales['scale'][0][0])
    # else:
    #     for model in car_models.models[:10]:
    #         filename = model.name + '.off'
    #         filepath = os.path.join(args.input, filename)
    #         if '.off' in filepath:
    #             print(toBlue('Reading .off file: ' + filepath))
    #             mesh = Mesh.from_off(filepath)
    #             if mesh==None:
    #                 print(toRed('Aborting file:' + filepath))
    #                 continue
    #         else:
    #             continue

    #         # Get extents of model.
    #         min_extent, max_extent = mesh.extents()
    #         total_min = np.min(np.array(min_extent))
    #         total_max = np.max(np.array(max_extent))

    #         # Scales all dimensions equally.
    #         sizes = (
    #             total_max - total_min,
    #             total_max - total_min,
    #             total_max - total_min
    #         )

    #         centers = (
    #             (min_extent[0] + max_extent[0]) / 2,
    #             (min_extent[1] + max_extent[1]) / 2,
    #             (min_extent[2] + max_extent[2]) / 2
    #         )
    #         translation = (
    #             -centers[0],
    #             -centers[1],
    #             -centers[2]
    #         )

    #         # translations.append(translation)


    #         max_sizes = (max(max_sizes[0], sizes[0]), max(max_sizes[1], sizes[1]), max(max_sizes[2], sizes[2]))

    #         print 'max_sizes: ', max_sizes

    #     scales = (
    #             1 / (max_sizes[0] + 2 * args.padding * max_sizes[0]),
    #             1 / (max_sizes[1] + 2 * args.padding * max_sizes[1]),
    #             1 / (max_sizes[2] + 2 * args.padding * max_sizes[2])
    #         )

    #     savemat('car_scales.mat', {'scales': scales, 'scale': scale})

    # print scales, scale

    # for filename in os.listdir(args.input):
    for idx, model in enumerate(car_models.models):
        filename = model.name + '.off'
        filepath = os.path.join(args.input, filename)
        if '.off' in filepath:
            print(toBlue('Reading .off file: ' + filepath))
            mesh = Mesh.from_off(filepath)
            if mesh==None:
                print(toRed('Aborting file:' + filepath))
                continue
        else:
            continue

        # Get extents of model.
        min_extent, max_extent = mesh.extents()
        total_min = np.min(np.array(min_extent))
        total_max = np.max(np.array(max_extent))

        centers = (
            (min_extent[0] + max_extent[0]) / 2,
            (min_extent[1] + max_extent[1]) / 2,
            (min_extent[2] + max_extent[2]) / 2
        )
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
        scales = (
            1 / (sizes[0] + 2 * args.padding * sizes[0]),
            1 / (sizes[1] + 2 * args.padding * sizes[1]),
            1 / (sizes[2] + 2 * args.padding * sizes[2])
        )

        # print 'centers, sizes, translation, scales: ', centers, sizes, translation, scales
        mesh.translate(translation)
        # scales = (scales[0]*0.8, scales[1]*0.8, scales[2]*0.8)
        mesh.scale(scales)

        mesh.translate((0.5, 0.5, 0.5))

        # scale = scale * 0.8
        mesh.scale((scale, scale, scale))

        after_min, after_max = mesh.extents()
        after_min_list = [float(after_mini) for after_mini in after_min]
        after_max_list = [float(after_maxi) for after_maxi in after_max]
        print(toCyan('%s extents after %.2f - %.2f, %.2f - %.2f, %.2f - %.2f.' % (os.path.basename(filepath), after_min[0], after_max[0], after_min[1], after_max[1], after_min[2], after_max[2])))
        if min(after_min_list)<=0 or max(after_max_list)>=args.height:
            print(toRed('Scale out of lims!'))

        savemat(os.path.join('s_t', '%d.mat' % model.id), {'translation':translation, 'scales':scales, 'sizes':sizes, 'scale':scale})


        mesh.to_off(os.path.join(args.output, '%d.off' % model.id))
    #     # n += 1
