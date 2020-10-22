import numpy as np
from tqdm import tqdm


def augment_patch(patch):
    if len(patch.shape[:-1]) == 2:
        patch = np.rot90(patch, k=np.random.randint(4), axes=(0, 1))
    elif len(patch.shape[:-1]) == 3:
        patch = np.rot90(patch, k=np.random.randint(4), axes=(1, 2))

    patch = np.flip(patch, axis=-2) if np.random.randint(2) else patch
    return patch


# Below implementation of stratified sampling inherited from Noise2Void: https://github.com/juglab/n2v
# Noise2void: learning denoising from single noisy images. Krull, Alexander, Tim-Oliver Buchholz, and Florian Jug. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.

def get_stratified_coords2D(coord_gen, box_size, shape):
    box_count_y = int(np.ceil(shape[0] / box_size))
    box_count_x = int(np.ceil(shape[1] / box_size))
    x_coords = []
    y_coords = []
    for i in range(box_count_y):
        for j in range(box_count_x):
            y, x = next(coord_gen)
            y = int(i * box_size + y)
            x = int(j * box_size + x)
            if (y < shape[0] and x < shape[1]):
                y_coords.append(y)
                x_coords.append(x)
    return (y_coords, x_coords)


def get_stratified_coords3D(coord_gen, box_size, shape):
    box_count_z = int(np.ceil(shape[0] / box_size))
    box_count_y = int(np.ceil(shape[1] / box_size))
    box_count_x = int(np.ceil(shape[2] / box_size))
    x_coords = []
    y_coords = []
    z_coords = []
    for i in range(box_count_z):
        for j in range(box_count_y):
            for k in range(box_count_x):
                z, y, x = next(coord_gen)
                z = int(i * box_size + z)
                y = int(j * box_size + y)
                x = int(k * box_size + x)
                if (z < shape[0] and y < shape[1] and x < shape[2]):
                    z_coords.append(z)
                    y_coords.append(y)
                    x_coords.append(x)
    return (z_coords, y_coords, x_coords)

def rand_float_coords2D(boxsize):
    while True:
        yield (np.random.rand() * boxsize, np.random.rand() * boxsize)
        
def rand_float_coords3D(boxsize):
    while True:
        yield (np.random.rand() * boxsize, np.random.rand() * boxsize, np.random.rand() * boxsize)