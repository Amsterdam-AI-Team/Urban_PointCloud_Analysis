import numpy as np
import logging
import matplotlib.pyplot as plt

from upcp.labels import Labels
from upcp.region_growing import LabelConnectedComp
from upcp.utils import math_utils
from upcp.utils import clip_utils

logger = logging.getLogger(__name__)


CLASS_COLORS = {'Unlabelled': 'lightgrey',
                'Ground': 'peru',
                'Road': 'sandybrown',
                'Building': 'lightblue',
                'Tree': 'green',
                'Street light': 'orange',
                'Traffic sign': 'crimson',
                'Traffic light': 'red',
                'City bench': 'darkviolet',
                'Rubbish bin': 'pink',
                'Car': 'grey',
                'Noise': 'whitesmoke'}


def get_mask_for_obj(points, labels, target_label, obj_loc, obj_top_z,
                     obj_angle=0, min_component_size=100,
                     octree_grid_size=0.4, noise_filter=True):
    target_idx = np.where(labels == target_label)[0]

    # Filter noise.
    if noise_filter:
        noise_components = (LabelConnectedComp(
                                grid_size=0.2,
                                min_component_size=10)
                            .get_components(points[target_idx]))
        noise_mask = noise_components != -1
    else:
        noise_mask = np.ones_like(labels, dtypo=bool)

    # Cluster points of target class.
    point_components = (LabelConnectedComp(
                            grid_size=octree_grid_size,
                            min_component_size=min_component_size)
                        .get_components(points[target_idx[noise_mask], 0:2]))

    cc_labels = np.unique(point_components)
    cc_labels = set(cc_labels).difference((-1,))

    off_eps = 0.05
    obj_idx = -1
    for cc in cc_labels:
        cc_mask = point_components == cc
        min_x, min_y, max_x, max_y = math_utils.compute_bounding_box(
                                points[target_idx[noise_mask]][cc_mask])
        off_h = (np.min(points[target_idx[noise_mask]][cc_mask, 2])
                 - obj_loc[2])
        offset = (np.sqrt((off_h / np.sin(np.deg2rad(90 - obj_angle)))**2
                          - off_h**2)
                  + off_eps)
        min_x -= offset
        min_y -= offset
        max_x += offset
        max_y += offset
        if min_x <= obj_loc[0] <= max_x and min_y <= obj_loc[1] <= max_y:
            obj_idx = int(cc)
            break
    if obj_idx == -1:
        logger.debug("No matching object found.")
        pad = 1
        box = (obj_loc[0]-pad, obj_loc[1]-pad, obj_loc[0]+pad, obj_loc[1]+pad)
    else:
        pad = 0.5
        box = (min_x-pad, min_y-pad, max_x+pad, max_y+pad)

    bg_mask = clip_utils.box_clip(
                        points, box, bottom=obj_loc[2]-pad, top=obj_top_z+2)
    return bg_mask


def plot_object(points, labels, colors=None, estimate=None):
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection='3d')

    label_set = np.unique(labels)

    for label in label_set:
        if label == Labels.NOISE:
            continue
        label_mask = labels == label
        label_str = Labels.get_str(label)

        ax1.scatter(xs[label_mask], zs[label_mask],
                    c=CLASS_COLORS[label_str], marker='.', edgecolors='none',
                    label=label_str)
        ax2.scatter(ys[label_mask], zs[label_mask],
                    c=CLASS_COLORS[label_str], marker='.', edgecolors='none')

        if colors is not None:
            c3 = colors[label_mask]
        else:
            c3 = CLASS_COLORS[label_str]
        ax3.scatter(xs[label_mask], ys[label_mask], zs[label_mask],
                    c=c3, marker='.', edgecolors='none', alpha=0.05)

        if estimate is not None:
            ax1.plot(estimate[:, 0], estimate[:, 2],
                     c='red', linewidth=3, alpha=0.7, label='Estimate')
            ax2.plot(estimate[:, 1], estimate[:, 2],
                     c='red', linewidth=3, alpha=0.7)
            ax3.plot(estimate[:, 0], estimate[:, 1], estimate[:, 2],
                     c='red', linewidth=3, alpha=1)

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    ax3.xaxis.set_ticklabels([])
    ax3.yaxis.set_ticklabels([])
    ax3.dist = 8

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc='upper center', bbox_to_anchor=(0.5, 1),
               ncol=int(len(by_label) / 2 + 0.5))

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
