import numpy as np
import logging
import matplotlib.pyplot as plt

from labels import Labels  #from upcp.labels import Labels 
from upcp.utils import math_utils
from upcp.utils import clip_utils
import laspy
import multiprocessing

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


def get_mask_for_obj(points, obj_loc, obj_top_z):
    pad = 2.5
    box = (obj_loc[0]-pad, obj_loc[1]-pad, obj_loc[0]+pad, obj_loc[1]+pad)

    bg_mask = clip_utils.box_clip(
                        points, box, bottom=obj_loc[2]-0.5, top=obj_top_z+5)
    return bg_mask


def generate_png_all_axes(identifier, points, labels, write_path, colors=None, estimate=None, show_image=False):
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection='3d')

    label_set = list(np.unique(labels))
    if 60 in labels:
        label_set.remove(60)
        label_set.append(60)

    for label in label_set:
        if label == Labels.NOISE:
            continue
        if label == 60:
            size = 5
        else:
            size = 1
        label_mask = labels == label
        label_str = Labels.get_str(label)

        # Use original (rgb) colors or color by class
        if colors is not None:
            my_colors = colors[label_mask]
        else:
            my_colors = CLASS_COLORS[label_str]

        # Plot point cloud data
        ax1.scatter(xs[label_mask], zs[label_mask]-np.min(zs),
                    c=my_colors, marker='.', edgecolors='none',
                    label=label_str, s=size)
        ax2.scatter(ys[label_mask], zs[label_mask]-np.min(zs),
                    c=my_colors, marker='.', edgecolors='none', s=size)
        ax3.scatter(xs[label_mask], ys[label_mask], zs[label_mask],
                    c=my_colors, marker='.', edgecolors='none', alpha=0.1)

        # Plot pole fit
        if estimate is not None:
            ax1.plot(estimate[:, 0], estimate[:, 2]-[np.min(zs), np.min(zs)],
                     c='red', linewidth=1, alpha=0.7, label='Estimate')
            ax2.plot(estimate[:, 1], estimate[:, 2]-[np.min(zs), np.min(zs)],
                     c='red', linewidth=1, alpha=0.7)

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    ax3.xaxis.set_ticklabels([])
    ax3.yaxis.set_ticklabels([])
    ax3.dist = 8

    # Add legend if coloring by class
    if colors is None:
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(),
                loc='upper center', bbox_to_anchor=(0.5, 1),
                ncol=int(len(by_label) / 2 + 0.5), markerscale=8)

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig('{}/{}.png'.format(write_path, identifier))
    if show_image:
        plt.show()
    plt.close()


def generate_png_single_axis(identifier, points, labels, write_path, colors=None, plot_axis='x'):
    if plot_axis == 'x':
        axis_hor = points[:, 0]
    elif plot_axis == 'y':
        axis_hor = points[:, 1]
    axis_ver = points[:, 2]

    label_set = list(np.unique(labels))
    if 60 in labels:
        label_set.remove(60)
        label_set.append(60)
    
    for label in label_set:
        if label == Labels.NOISE:
            continue

        label_mask = labels == label
        label_str = Labels.get_str(label)

        if label == 60:
            size = 5
        else:
            size = 1

        # Use original (rgb) colors or color by class
        if colors is not None:
            my_colors = colors[label_mask]
        else:
            my_colors = CLASS_COLORS[label_str]
        
        # Plot point cloud data
        plt.scatter(axis_hor[label_mask], axis_ver[label_mask],
                        c=my_colors, marker='.', edgecolors='none',
                        label=label_str, s=size)

    ax = plt.gca()
    ax.set_aspect('equal')
    
    pad = 1
    plt.xlim(min(axis_hor)-pad, max(axis_hor)+pad)
    plt.ylim(min(axis_ver)-pad, max(axis_ver)+pad)
    plt.axis('off')

    file_name = '{}/{}/{}_{}_{}_{}_{}.png'.format(write_path, plot_axis, identifier, 
                                                  min(axis_hor)-pad, min(axis_ver)-pad, 
                                        max(axis_hor)+pad, max(axis_ver)+pad)      
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def create_images_for_poles(poles_df, dataset_folder, pred_folder, img_out_folder, prefix, prefix_pred):
    # Save png of object x, y and 3d axis
    open_tile = []
    for idx, obj in poles_df.iterrows():
        if idx % 1000 == 0:
            print(idx)
            
        # Get object location and top (per pole)
        identifier = obj.identifier
        obj_location = (obj.rd_x, obj.rd_y, obj.z)
        obj_top = (obj.tx, obj.ty, obj.tz)

        if obj.tilecode != open_tile:
            # Get the point cloud data (per tile)
            cloud = laspy.read(f'{dataset_folder}{prefix}{obj.tilecode}.laz')
            points = np.vstack((cloud.x, cloud.y, cloud.z)).T
            npz_file = np.load(pred_folder + prefix_pred + obj.tilecode + '.npz')
            labels = npz_file['label']
            colors = np.vstack((cloud.red, cloud.green, cloud.blue)).T / (2**16 - 1)
            open_tile = obj.tilecode  # tile_code that is currently open

        # Get a mask for the point cloud around the object's location (per pole)
        obj_mask = get_mask_for_obj(points, obj_location, obj_top[2])
        if sum(obj_mask) > 0:
            # Save the object for all axes
            write_path = img_out_folder + 'object_all_axes' 
            p = multiprocessing.Process(target=generate_png_all_axes, 
            args=(identifier, points[obj_mask], labels[obj_mask], write_path, 
            colors[obj_mask], np.vstack((obj_location, obj_top)), False))
            p.start()                  
            # Save the objects per axis
            write_path = img_out_folder + 'object_per_axis'
            p = multiprocessing.Process(target=generate_png_single_axis, 
            args=(identifier, points[obj_mask], labels[obj_mask], write_path, colors[obj_mask], 'x'))
            p.start() 
            p = multiprocessing.Process(target=generate_png_single_axis, 
            args=(identifier, points[obj_mask], labels[obj_mask], write_path, colors[obj_mask], 'y'))
            p.start() 