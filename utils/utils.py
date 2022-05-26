import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms


def decodebox(regression, anchors, img):
    dtype = regression.dtype
    anchors = anchors.to(dtype)
    #--------------------------------------#
    #   Calculate the center of prior boxes
    #--------------------------------------#
    y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
    x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2

    #--------------------------------------#
    #   Calculate the width and height of prior boxes
    #--------------------------------------#
    ha = anchors[..., 2] - anchors[..., 0]
    wa = anchors[..., 3] - anchors[..., 1]

    #--------------------------------------#
    #   Calculate the width and height of adjusted prior boxes
    #--------------------------------------#
    w = regression[..., 3].exp() * wa
    h = regression[..., 2].exp() * ha

    #--------------------------------------#
    #   Calculate the center of adjusted prior boxes
    #--------------------------------------#
    y_centers = regression[..., 0] * ha + y_centers_a
    x_centers = regression[..., 1] * wa + x_centers_a

    #--------------------------------------#
    #   Calculate the upper left and lower right corner of the prediction box
    #--------------------------------------#
    ymin = y_centers - h / 2.
    xmin = x_centers - w / 2.
    ymax = y_centers + h / 2.
    xmax = x_centers + w / 2.

    #--------------------------------------#
    #   Stack the results of the prediction box
    #--------------------------------------#
    boxes = torch.stack([xmin, ymin, xmax, ymax], dim=2)

    _, _, height, width = np.shape(img)

    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)
    

    return boxes
    
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def retinanet_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)/input_shape
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)/input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        Calculate iou
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
                 
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Get the category and its confidence
        class_conf, class_pred = torch.max(image_pred[:, 4:], 1, keepdim=True)

        # Confidence was used for the first round of screening
        conf_mask = (class_conf >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        class_conf, class_pred = class_conf[conf_mask], class_pred[conf_mask]

        if not image_pred.size(0):
            continue


        detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float()), 1)

        # Get the category
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        for c in unique_labels:
            # Obtain all the predicted results of a preliminary screening
            detections_class = detections[detections[:, -1] == c]
            
            #------------------------------------------#
            # It's faster to use the official non-max suppression!
            #------------------------------------------#
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4],
                nms_thres
            )
            max_detections = detections_class[keep]

            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output
