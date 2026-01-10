"""
Utility functions for loading NABirds dataset metadata.
Python 3 compatible version of core loading functions.
"""

import os


def load_class_names(dataset_path=''):
    """Load class names from classes.txt."""
    names = {}
    
    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])
    
    return names


def load_image_labels(dataset_path=''):
    """Load image class labels from image_class_labels.txt."""
    labels = {}
    
    with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            class_id = pieces[1]
            labels[image_id] = class_id
    
    return labels


def load_image_paths(dataset_path='', path_prefix=''):
    """Load image paths from images.txt."""
    paths = {}
    
    with open(os.path.join(dataset_path, 'images.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            path = os.path.join(path_prefix, pieces[1])
            paths[image_id] = path
    
    return paths


def load_bounding_box_annotations(dataset_path=''):
    """Load bounding box annotations from bounding_boxes.txt."""
    bboxes = {}
    
    with open(os.path.join(dataset_path, 'bounding_boxes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            bbox = tuple(map(int, pieces[1:]))
            bboxes[image_id] = bbox
    
    return bboxes


def load_train_test_split(dataset_path=''):
    """Load train/test split from train_test_split.txt."""
    train_images = []
    test_images = []
    
    with open(os.path.join(dataset_path, 'train_test_split.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            is_train = int(pieces[1])
            if is_train:
                train_images.append(image_id)
            else:
                test_images.append(image_id)
    
    return train_images, test_images
