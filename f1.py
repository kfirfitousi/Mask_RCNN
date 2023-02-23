"""
Mask R-CNN
Configurations and data loading code for the F1 dataset
https://universe.roboflow.com/dev-drone-bhowmik/f1-2021-cars/

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import time
import numpy as np
import imgaug

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# Path to trained COCO weights file
# Downloaded from https://github.com/matterport/Mask_RCNN/releases/tag/v2.0
COCO_MODEL_PATH = "mask_rcnn_coco.h5"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "logs"

############################################################
#  Configurations
############################################################


class F1Config(Config):
    """Configuration for training on the F1 dataset.
    Derives from the base Config class and overrides values specific
    to the F1 dataset.
    """
    # Give the configuration a recognizable name
    NAME = "f1"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 10  # F1 has 10 classes

    def __init__(self, use_mini_mask=True):
        super().__init__()
        self.USE_MINI_MASK = use_mini_mask


############################################################
#  Dataset
############################################################

class F1Dataset(utils.Dataset):
    anns = []
    class_names = [
        "Alfa Romeo",
        "Alpha Tauri",
        "Alpine",
        "Aston Martin",
        "Ferarri",
        "Haas",
        "McLaren",
        "Mercedes",
        "Red Bull",
        "Williams"
    ]

    def load_f1(self, dataset_dir, subset, return_f1=False):
        """Load a subset of the F1 dataset.
        dataset_dir: The root directory of the F1 dataset.
        subset: What to load (train, val)
        return_coco: If True, returns the F1 object.
        """
        f1 = COCO("{}/{}/_annotations.coco.json".format(dataset_dir, subset))
        self.anns = f1.anns

        # Add classes
        for i, name in enumerate(self.class_names):
            self.add_class("f1", i+1, name)

        # Add images
        image_dir = "{}/{}".format(dataset_dir, subset)
        image_ids = list(f1.imgs.keys())

        for i in image_ids:
            self.add_image(
                "f1", image_id=i,
                path=os.path.join(image_dir, f1.imgs[i]['file_name']),
                width=f1.imgs[i]["width"],
                height=f1.imgs[i]["height"],
                annotations=self.loadAnns(f1.getAnnIds(
                    imgIds=[i], 
                    catIds=[i for i in range(1, 11)], 
                    iscrowd=None)))

        if return_f1:
            return f1

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a F1 image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "f1":
            return super(F1Dataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "f1.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                #
                # probably unnecessary?
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(F1Dataset, self).load_mask(image_id)

    # taken from pycocotools/coco.py and modified to calculate segmentations from bbox
    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids                   : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if type(ids) == list:
            anns = [self.anns[i] for i in ids]
        elif type(ids) == int:
            anns = [self.anns[ids]]
        else:
            anns = []

        # bbox is a 1x4 vector [y,x,height,width]
        # segmentation is a Mx2N vector, where:
        #   M = number of objects (if only one object is in bbox then M=1)
        #   N = number of points
        # We convert bbox from [y,x,h,w] format to [y0,x0,y1,x1] format,
        # and then create segmentation vector as
        #   [[y0, x0, y1, x0, y1, x1, y0, x1, y0, x0]]
        # for each annotation.
        # We also set the area to (bbox_width * bbox_height).

        for ann in anns:
            bbox = ann['bbox']
            bbox[2] = bbox[0] + bbox[2]  # y1 = y0 + h
            bbox[3] = bbox[1] + bbox[3]  # x1 = x0 + w

            # bbox[0] = y0, bbox[1] = x0, bbox[2] = y1, bbox[3] = x1
            ann["segmentation"] = [[bbox[0], bbox[1], bbox[2], bbox[1],
                                    bbox[2], bbox[3], bbox[0], bbox[3], bbox[0], bbox[1]]]

            ann["area"] = (ann['bbox'][2] - ann['bbox'][0]) * \
                (ann['bbox'][3] - ann['bbox'][1])

        return anns

    def image_reference(self, image_id):
        """Return the image path"""
        info = self.image_info[image_id]
        if info["source"] == "f1":
            return info["path"]
        else:
            super(F1Dataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "f1"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training and Inference
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on the F1 dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'infer' on the F1 dataset")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/coco/",
                        help='Directory of the F1 dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--video', required=False,
                        metavar="/path/to/video",
                        help='Path to the input video')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    if args.command == "infer":
        from infer import VideoInference
        VideoInference(args.model, args.video, args.logs).infer()

    else:
        if args.dataset is None:
            print("must provide dataset path")
            exit(1)

        config = F1Config()
        config.display()

        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)

        # Select weights file to load
        if args.model.lower() == "coco":
            model_path = COCO_MODEL_PATH
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = model.get_imagenet_weights()
        else:
            model_path = args.model

        # Load weights
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])  # maybe don't exclude if continuing previous training?

        # Training dataset
        dataset_train = F1Dataset()
        dataset_train.load_f1(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = F1Dataset()
        dataset_val.load_f1(args.dataset, "val")
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=30,
                    layers='heads',
                    augmentation=augmentation)

        # # Training - Stage 2
        # # Finetune layers from ResNet stage 4 and up
        # print("Fine tune Resnet stage 4 and up")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=20,
        #             layers='4+',
        #             augmentation=augmentation)
        #
        # # Training - Stage 3
        # # Fine tune all layers
        # print("Fine tune all layers")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 10,
        #             epochs=30,
        #             layers='all',
        #             augmentation=augmentation)







