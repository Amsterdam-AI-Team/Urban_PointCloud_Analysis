import numpy as np
import logging
from numba import jit
from sklearn.decomposition import PCA
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from upcp.region_growing import LabelConnectedComp
from upcp.labels import Labels
from upcp.utils.interpolation import FastGridInterpolator
from upcp.utils import clip_utils
from upcp.utils import math_utils

logger = logging.getLogger(__name__)

DEBUG_INFO = {0: 'No errors',
              1: 'AHN fallback',
              2: 'Ground undetermined',
              3: 'Slope undetermined',
              4: 'Pole not detected'}


@jit(nopython=True)
def _get_xystd(points, z, margin):
    """Returns the mean and std.dev. of `points` within `margin` of `z`."""
    clip_mask = (points[:, 2] >= z - margin) & (points[:, 2] < z + margin)
    if np.count_nonzero(clip_mask) > 0:
        x_mean = np.mean(points[clip_mask, 0])
        y_mean = np.mean(points[clip_mask, 1])
        xy_std = np.max(np.array([np.std(points[clip_mask, 0]),
                                  np.std(points[clip_mask, 1])]))
        return x_mean, y_mean, z, xy_std
    else:
        return np.nan, np.nan, z, np.nan


def _extract_pole(points, ground_est=None, step=0.1, percentile=25):
    """
    Extract pole features from a set of points. The supplied points are assumed
    to contain one pole-shaped object. If a `ground_est` is provided this is
    assumed to be the true ground elevation at that location.

    Returns a tuple (x, y, z, x2, y2, z2, height, angle), where (x, y, z) is
    the location of the bottom of the pole, and (x2, y2, z2) is the top.
    """
    debug = 0
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    if ground_est is None:
        ground_est = z_min
    elif np.isnan(ground_est):
        ground_est = z_min

    # Collect [x_mean, y_mean, z, std_dev] statistics for horizontal slices of
    # the point cloud.
    xyzstd = np.array([[*_get_xystd(points, z, step)]
                       for z in np.arange(z_min + step, z_max, 2*step)])
    if len(xyzstd) > 0:
        # Keep only those slices of which the std_dev is < 25th percentile.
        # This makes sure we only focus on the pole itself, not any extensions.
        valid_mask = xyzstd[:, 3] <= np.nanpercentile(xyzstd[:, 3], percentile)
    else:
        valid_mask = np.zeros((len(points),), dtype=bool)
    if np.count_nonzero(valid_mask) == 0:
        logger.debug('Not enough data to extract pole.')
        debug = 4
        origin = np.mean(points, axis=0)
        direction_vector = np.array([0., 0., 1.])
    elif np.count_nonzero(valid_mask) == 1:
        logger.debug('Not enough data to determine slope.')
        debug = 3
        origin = xyzstd[valid_mask, 0:3][0]
        direction_vector = np.array([0., 0., 1.])
    else:
        # We have enough data to try a PCA fit to determine the exact pole.
        pca = PCA(n_components=1).fit(xyzstd[valid_mask, 0:3])
        origin = pca.mean_
        direction_vector = pca.components_[0]
        if direction_vector[2] < 0:
            direction_vector *= -1
    # Compute the pole dimensions from the fitted data.
    extent = (origin[2] - ground_est, z_max - origin[2])
    multiplier = np.sum(np.linalg.norm(direction_vector, 2))
    x, y, z = origin - direction_vector * extent[0] * multiplier
    x2, y2, z2 = origin + direction_vector * extent[1] * multiplier
    height = np.sum(extent) * multiplier
    angle = math_utils.vector_angle(direction_vector)
    return (x, y, z, x2, y2, z2, height, angle), debug


def get_pole_locations(
        points, labels, probabilities, target_label, ground_labels,
        ahn_reader=None, tilecode=None, building_polygons=None,
        min_component_size=100, octree_grid_size=0.2):
    """
    Returns a list of locations and dimensions of pole-like objects
    corresponding to the target_label in a given point cloud.

    Parameters
    ----------
    points : array with shape (n_points, 3)
        The point cloud.
    labels : array of shape (n_points,)
        The corresponding labels.
    probabilities : array of shape (n_points,)
        The corresponding probabilities.
    target_label : int
        The label of the target class.
    ground_labels : list of int
        The label(s) of the ground class(es), for height determination.
    ahn_reader : AHNReader, optional
        To extract the ground elevation, if this cannot be determined from the
        labelled point cloud itself. Fall-back method.
    tilecode : str, optional
        Only needed when ahn_reader is provided.
    building_polygons : list of Shapely Polygons, optional
        Used to flag whether extracted pole is inside a building.
    min_component_size : int (default: 100)
        Minimum size of a component to be considered.
    octree_grid_size : float (default: 0.2)
        Octree level for the LabelConnectedComp algorithm.

    Returns
    -------
    A list of tuples, one for each pole-like object:
    (x, y, z, x2, y2, z2, height, angle, proba, n_points, in_building, debug),
    where (x, y, z) is the location of the bottom of the pole, and (x2, y2, z2)
    is the top.
    """
    pole_locations = []
    mask_ids = np.where(labels == target_label)[0]

    if len(mask_ids) > 0:
        noise_components = (LabelConnectedComp(
                                grid_size=octree_grid_size,
                                min_component_size=10)
                            .get_components(points[mask_ids]))
        noise_filter = noise_components != -1
        if np.count_nonzero(noise_filter) < min_component_size:
            return pole_locations
        point_components = (LabelConnectedComp(
                                        grid_size=octree_grid_size,
                                        min_component_size=min_component_size)
                            .get_components(points[mask_ids[noise_filter],
                                                   0:2]))

        cc_labels, counts = np.unique(point_components, return_counts=True)
        cc_labels = cc_labels[(cc_labels != -1)
                              & (counts >= min_component_size)]

        logger.info(f'{len(cc_labels)} objects of class ' +
                    f'[{Labels.get_str(target_label)}] found.')

        ground_mask = np.zeros_like(labels, dtype=bool)
        for gnd_lab in ground_labels:
            ground_mask = ground_mask | (labels == gnd_lab)

        if building_polygons is not None:
            polys = [Polygon(bld) for bld in building_polygons]
        else:
            polys = []

        for cc in cc_labels:
            ground_debug = 0
            cc_mask = (point_components == cc)
            n_points = np.count_nonzero(cc_mask)
            logger.debug(f'Cluster {cc}: {n_points} points.')
            cluster_center = np.mean(
                        points[mask_ids[noise_filter]][cc_mask, 0:2], axis=0)
            ground_clip = clip_utils.circle_clip(
                                    points[ground_mask], cluster_center, 1.)
            if np.count_nonzero(ground_clip) > 0:
                ground_est = np.mean(points[ground_mask, 2][ground_clip])
            elif ahn_reader is None:
                ground_clip = clip_utils.circle_clip(
                                    points[ground_mask], cluster_center, 2.)
                if np.count_nonzero(ground_clip) > 0:
                    ground_est = np.mean(points[ground_mask, 2][ground_clip])
                else:
                    ground_est = None
            else:
                logger.debug('Falling back to AHN data.')
                ground_debug = 1
                ahn_tile = ahn_reader.filter_tile(tilecode)
                fast_z = FastGridInterpolator(ahn_tile['x'], ahn_tile['y'],
                                              ahn_tile['ground_surface'])
                ground_est = fast_z(np.array([cluster_center]))[0]
                if np.isnan(ground_est):
                    z_vals = fast_z(points[mask_ids[noise_filter]][cc_mask])
                    if np.isnan(z_vals).all():
                        logger.warn(
                            f'Missing AHN data for point ({cluster_center}).')
                        ground_debug = 2
                    else:
                        ground_est = np.nanmean(z_vals)
            pole, pole_debug = _extract_pole(
                        points[mask_ids[noise_filter]][cc_mask], ground_est)
            dims = tuple(round(x, 2) for x in pole)
            proba = np.mean(probabilities[mask_ids[noise_filter]][cc_mask])
            debug = f'{ground_debug}_{pole_debug}'
            in_building = 0
            point = Point(dims[0], dims[1])
            for poly in polys:
                if poly.contains(point):
                    in_building = 1
                    break
            dims = (*dims, round(proba, 2), n_points, in_building, debug)
            pole_locations.append(dims)
    return pole_locations
