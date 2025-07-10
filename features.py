from collections import defaultdict
from contextlib import closing
from itertools import chain, product
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr
from skimage import feature
from skimage.color import rgb2lab, deltaE_ciede2000
from typing import Literal, Union

import cv2
import multiprocessing as mp
import numpy as np


from components import TemporalContour
from utilities import DisjointSet, gaussian_kernel

BlockNorm = Literal["L1", "L1-sqrt", "L2", "L2-Hys"]


def cluster_by_color(rgb_video: np.ndarray, masks: np.ndarray, delta=20.0, kL: int = 1, kC: int = 1, kH: int = 1):
    assert len(masks.shape) == 3, "expected array with shape (frames x height x width)"
    assert masks.dtype == np.bool, "expected binary array"

    _, height, width = masks.shape
    cielab_video = np.array([rgb2lab(f) for f in rgb_video])
    labels = np.zeros_like(masks, dtype=np.int32)
    label_set = DisjointSet()
    next_label = 1
    foreground = np.array(np.where(masks)).T

    coords = [
        [0, 0, -1], [0, -1, -1], [0, -1, 0], [0, -1, 1], 
        [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 1, -1 ],
        [-1, 0, 0], [-2, 0, 0], [-3, 0, 0], [-4, 0, 0]
    ]

    for f, r, c in foreground:
        candidates = [
            (t, y, x)
            for t, y, x in [(f + dz, r + dy, c + dx) for dz, dy, dx in coords]
            if t >= 0 and 0 <= y < height and 0 <= x < width and labels[t, y, x] > 0
        ]

        current_color = cielab_video[f, r, c]

        if candidates:
            cielab_colors = np.array([cielab_video[cz, cy, cx] for cz, cy, cx in candidates])
            diffs = deltaE_ciede2000(np.full_like(cielab_colors, current_color), cielab_colors, kL, kC, kH)
            min_diff_idx = np.argmin(diffs)

            if diffs[min_diff_idx] <= delta:
                contour_label = labels[*candidates[min_diff_idx]]
            else:
                contour_label = next_label
                label_set.add(next_label)
                next_label += 1
        
            labels[f, r, c] = contour_label

            for i, (cz, cy, cx) in enumerate(candidates):
                if diffs[i] <= delta:
                    label_set.union(contour_label, labels[cz, cy, cx])
        else:
            contour_label = next_label
            label_set.add(next_label)
            next_label += 1
        
            labels[f, r, c] = contour_label
        
    components = defaultdict(list)

    for f, r, c in foreground:
        label = label_set.find(labels[f, r, c])
        components[label].append((f, r, c))

    return [TemporalContour(p) for p in components.values()]


def cluster_by_color_parallel(rgb_video: np.ndarray, masks: np.ndarray, delta=20.0, kL: int = 1, kC: int = 1, kH: int = 1, cpus=4):
    assert len(masks.shape) == 3, "expected array with shape (frames x height x width)"
    assert masks.dtype == np.bool, "expected binary array"

    _, height, width = masks.shape
    cielab_video = np.array([rgb2lab(f) for f in rgb_video])
    foreground = np.array(np.where(masks)).T

    coords = [
        [0, 0, -1], [0, -1, -1], [0, -1, 0], [0, -1, 1], 
        [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 1, -1 ],
        [-1, 0, 0], [-2, 0, 0], [-3, 0, 0], [-4, 0, 0]
    ]

    params = []

    for f, r, c in foreground:
        candidates = [
            (f, r, c)
            for f, r, c in [(z + dz, y + dy, x + dx) for dz, dy, dx in coords]
            if f >= 0 and 0 <= r < height and 0 <= c < width
        ]

        current_color = cielab_video[z, y, x]
        cielab_colors = np.array([cielab_video[cz, cy, cx] for cz, cy, cx in candidates])

        params.append([z, y, x, delta, candidates, current_color, cielab_colors])
    

    with closing(mp.Pool(processes=cpus)) as pool:
        groups = pool.starmap_async(_cluster_parallel, params)

    pool.join()

    labels = np.zeros_like(masks, dtype=np.int32)
    label_set = DisjointSet()
    next_label = 1

    for group in groups.get():
        incr_label = 0
        for z, y, x in group:
            if labels[z, y, x] == 0:
                labels[z, y, x] = next_label
                incr_label += 1

        if incr_label > 0:
            label_set.add(next_label)
            next_label += 1

        first_label = labels[*group[0]]

        for i in range(1, len(group)):
            label_set.union(first_label, labels[*group[i]])
        
    components = defaultdict(list)

    for f, r, c in foreground:
        label = label_set.find(labels[f, r, c])
        components[label].append((f, r, c))

    return [TemporalContour(p) for p in components.values()]


def histogram_of_oriented_gradients(
        image: np.ndarray, 
        orientations: int = 9, 
        pixels_per_cell: tuple[int, int] = (8, 8),
        cels_per_block: tuple[int, int] = (3, 3),
        block_norm: BlockNorm = "L2-Hys",
        transform_sqrt: bool = False,
        channel_axis: Union[int, None] = None) -> np.ndarray:
    
    return feature.hog(image, orientations, pixels_per_cell, cels_per_block, block_norm, transform_sqrt, channel_axis)


def histogram_of_oriented_gradients_video(
        video: np.ndarray, 
        orientations: int = 9, 
        pixels_per_cell: tuple[int, int] = (8, 8),
        cels_per_block: tuple[int, int] = (3, 3),
        block_norm: BlockNorm = "L2-Hys",
        transform_sqrt: bool = False,
        channel_axis: Union[int, None] = None) -> np.ndarray:
    
    hogs = [
        feature.hog(frame, orientations, pixels_per_cell, cels_per_block, block_norm, transform_sqrt, channel_axis)
        for frame in video
    ]

    hogs = np.array(list(chain(*hogs)))
    nbins = int(np.max(hogs) + 1)
    hist, _ = np.histogram(hogs, bins=nbins, range=(0, nbins)).astype("float")
    
    return hist / hist.sum()


def local_binary_pattern(
        image: np.ndarray, 
        points: int, 
        radius: float, 
        method: str ='default') -> tuple[np.ndarray, np.ndarray]:
    lbp = feature.local_binary_pattern(image, points, radius, method=method)
    nbins = int(np.max(lbp) + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=nbins, range=(0, nbins)).astype("float")
    
    return lbp, hist / hist.sum()


def local_binary_pattern_video(
        video: np.ndarray, 
        points: int, 
        radius: float, 
        method: str ='default') -> tuple[np.ndarray, np.ndarray]:
    lbps = [feature.local_binary_pattern(frame, points, radius, method=method).ravel() for frame in video]
    lbps = np.array(list(chain(*lbps)))
    nbins = int(np.max(lbps) + 1)
    hist, _ = np.histogram(lbps, bins=nbins, range=(0, nbins)).astype("float")
    
    return lbps, hist / hist.sum()


def optical_flow(grayscale_video: np.ndarray):
    prvs = grayscale_video[0]
    hsv = np.zeros((grayscale_video.shape[0], grayscale_video.shape[1], grayscale_video.shape[2], 3))
    hsv[..., 1] = 255

    for f, frame in enumerate(grayscale_video[1:]):
        flow = cv2.calcOpticalFlowFarneback(prvs, frame, None, 0.5, 5, 15, 10, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[f, :, :, 0] = ang*180/np.pi/2
        hsv[f, :, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        prvs = frame

    return hsv.astype(np.uint8)


def regress_to_background(video: np.ndarray, background: np.ndarray) -> np.ndarray:
    frames, height, width = video.shape[:3]
    pixels_per_frame = height * width * 4
    pixels = frames * pixels_per_frame

    def v(f, r, c, p):
        return f * pixels_per_frame + (r * width + c) * 4 + p
    
    def equations():
        for i, j, k, l in product(range(frames), range(height), range(width), range(3)):
            yield [(1, v(i, j, k, l)), (video[i, j, k, l], v(i, j, k, 3)), background[i, j, k, l]]

        for i, j, k, l in product(range(frames), range(height), range(width - 1), [3, 0, 1, 2]):
            yield [(1, v(i, j, k, l)), (-1, v(i, j, k + 1, l)), 0]

        for i, j, k, l in product(range(frames), range(height - 1), range(width), [3, 0, 1, 2]):
            yield [(1, v(i, j, k, l)), (-1, v(i, j + 1, k, l)), 0]

        for i, j, k, l in product(range(frames - 1), range(height), range(width), [3, 0, 1, 2]):
            yield [(1, v(i, j, k, l)), (-1, v(i + 1, j, k, l)), 0]

    solution = _solve_sparse_equations(equations())

    for pixel in range(0, pixels, 4):
        solution[pixel + 3] = 1 - solution[pixel]

    return solution.reshape(frames, height, width, 4)


def regress_to_background_blocked(src: np.ndarray, background: np.ndarray, block_size: int, overlap: int) -> np.ndarray:
    assert block_size >= overlap, "expected overlap to be less than or equal to the block size"

    frames, height, width = src.shape[:3]

    assert block_size <= height and block_size <= width and block_size <= frames, "expected block to fit in dimensions of video"

    block_stride = block_size - overlap
    len_x, len_y, len_z = width - block_size + 1, height - block_size + 1, frames - block_size + 1
    block_weight = gaussian_kernel(block_size, sigma=block_size / 6.0)
    linspace_x = np.linspace(0, len_x - 1, max(1, int(np.ceil(len_x / block_stride))), dtype=int)
    linspace_y = np.linspace(0, len_y - 1, max(1, int(np.ceil(len_y / block_stride))), dtype=int)
    linspace_z = np.linspace(0, len_z - 1, max(1, int(np.ceil(len_z / block_stride))), dtype=int)
    
    result, weights = np.zeros((frames, height, width, 4)), np.zeros((frames, height, width, 1))
    i = 0

    for (z, y, x) in product(linspace_z, linspace_y, linspace_x):
        x_end, y_end, z_end = x + block_size, y + block_size, z + block_size
        if i < 20:
            print(f"Block_{i} ==> ({x}, {x_end}); ({y}, {y_end}); ({z}, {z_end})")
            i += 1

        vid_block, background_block = src[z:z_end, y:y_end, x:x_end], background[z:z_end, y:y_end, x:x_end]
        solution = regress_to_background(vid_block, background_block)
        result[z:z_end, y:y_end, x:x_end] += solution * block_weight[:, :, :, np.newaxis]
        weights[z:z_end, y:y_end, x:x_end] += block_weight[:, :, :, np.newaxis]

    return result / weights


def _cluster_parallel(f, r, c, p, candidates, current_color, cielab_colors):
    diffs = deltaE_ciede2000(np.full_like(cielab_colors, current_color), cielab_colors)
    group = [(f, r, c)]

    for i in range(len(diffs)):
        if diffs[i] <= p:
            group.append(candidates[i])

    return group


def _solve_sparse_equations(equations) -> np.ndarray:
    """Solves the linear system A*x = b, where `A` and `b` are determined by the input equations
    
    Parameters
    ----------
    equations - a collection of equations formatted as (A_i, column, b_i)
    bounds - lower and upper bounds on the variables
    """
    rows, cols, data, b = [], [], [], []

    for i, eq in enumerate(equations):
        for coeff, var in eq[:-1]:
            rows.append(i)
            cols.append(var)
            data.append(coeff)
        b.append(eq[-1])

    nvars = max(cols) + 1
    A = csr_matrix((np.array(data), (rows, cols)), shape=(len(b), nvars))
    b = np.array(b)
    
    return lsmr(A, b)[0]
