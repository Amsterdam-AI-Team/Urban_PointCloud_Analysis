import numpy as np
import logging
import matplotlib.pyplot as plt

#from upcp.labels import Labels   # TODO change back with the below
from labels import Labels
from upcp.utils import math_utils
from upcp.utils import clip_utils

from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

logger = logging.getLogger(__name__)

CLASS_COLORS = {'Unknown': 'lightgrey',
                'Road': 'sandybrown',
                'Sidewalk': 'peachpuff',
                'Other ground': 'peru',
                'Building': 'lightblue',
                'Wall': 'lightblue',
                'Fence': 'black', 
                'Houseboat': 'lightblue',
                'Bridge': 'linen',
                'Bus/tram shelter': 'chocolate',
                'Advertising column': 'chocolate',
                'Kiosk': 'chocolate',
                'Other structure': 'chocolate',
                'Tree': 'green',
                'Potted plant': 'palegreen',
                'Other vegetation ': 'seagreen',
                'Car': 'grey',
                'Truck': 'grey',
                'Bus': 'darkgrey',
                'Tram': 'darkgrey',
                'Bicycle': 'lightgrey',                
                'Scooter/Motorcycle': 'lightgrey',                
                'Other vehicle': 'grey',
                'Person': 'sienna',
                'Person sitting': 'sienna',
                'Cyclist': 'sienna',
                'Other Person': 'sienna',
                'Streetlight': 'orange',
                'Traffic light': 'red',
                'Traffic sign': 'crimson',
                'Signpost': 'crimson',
                'Flagpole': 'coral',
                'Bollard': 'red',
                'Parasol': 'coral',
                'Complex pole': 'salmon',
                'Other pole': 'coral',
                'Tram cable': 'darkgrey',
                'Other cable': 'silver',
                'City bench': 'darkviolet',
                'Rubbish bin': 'pink',
                'Small container': 'rosybrown', 
                'Large container': 'rosybrown', 
                'Letter box': 'navy',
                'Parking meter': 'royalblue',
                'EV charging station': 'cyan', 
                'Fire hydrant': 'aqua',
                'Bicycle rack': 'deepskyblue', 
                'Advertising sign': 'steelblue', 
                'Hanging streetlight': 'orangered',
                'Terrace': 'plum',
                'Playground': 'fuchsia',
                'Electrical box': 'purple',
                'Concrete block': 'thistle',
                'Construction sign': 'tomato',
                'Other object': 'teal',
                'Noise': 'whitesmoke'}


def get_mask_for_obj(points, labels, target_label, obj_loc, obj_top_z,
                     obj_angle=0, noise_filter=True, 
                     eps_noise=0.6, min_samples_noise=10,
                     eps=0.6, min_samples=100):
    target_idx = np.where(labels == target_label)[0]

    # Filter noise (label -1)
    if noise_filter:
#        noise_components = (DBSCAN(
        noise_components = (OPTICS(
                                eps=eps_noise,
                                min_samples=min_samples_noise)
                            .fit_predict(points[target_idx]))
        noise_mask = noise_components != -1
    else:
        noise_mask = np.ones_like(labels, dtypo=bool)

    # Cluster points of target class
#    point_components = (DBSCAN(
    point_components = (OPTICS(
                            eps=eps,
                            min_samples=min_samples)
                        .fit_predict(points[target_idx[noise_mask], 0:2]))

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


def plot_object(points, labels, colors=None, estimate=None, output_path=None):
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
    
    if output_path is not None:
        plt.savefig(output_path, transparent=False, facecolor='white')
    
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
