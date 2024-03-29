import os
import matplotlib.pyplot as plt
import scipy.signal
import torch
import torch.nn as nn


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU

def get_target(anchor, bbox_annotation, classification, cuda):
    #------------------------------------------------------#
    #   Calculate the iou between real and prior anchor box
    #   anchor              num_anchors, 4
    #   bbox_annotation     num_true_boxes, 5
    #   Iou                 num_anchors, num_true_boxes
    #------------------------------------------------------#
    IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])
    
    #------------------------------------------------------#
    #   Calculate the real box with the maximum coincidence degree with the prior box
    #   IoU_max             num_anchors,
    #   IoU_argmax          num_anchors,
    #------------------------------------------------------#
    IoU_max, IoU_argmax = torch.max(IoU, dim=1)

    #------------------------------------------------------#
    #   Find which prior boxes need to be ignored when calculating loss
    #------------------------------------------------------#
    targets = torch.ones_like(classification) * -1
    if cuda:
        targets = targets.cuda()

    #------------------------------------------#
    #   Training is done when the coincidence degree is smaller than 0.4
    #------------------------------------------#
    targets[torch.lt(IoU_max, 0.4), :] = 0

    #--------------------------------------------------#
    #   Training is done and the regression loss is calculated when the coincidence degree is larger than 0.5
    #--------------------------------------------------#
    positive_indices = torch.ge(IoU_max, 0.5)

    #--------------------------------------------------#
    #   Take the real box most corresponding to each prior box
    #--------------------------------------------------#
    assigned_annotations = bbox_annotation[IoU_argmax, :]

    #--------------------------------------------------#
    #   Set the corresponding category as 1
    #--------------------------------------------------#
    targets[positive_indices, :] = 0
    targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
    #--------------------------------------------------#
    #   Calculate the number of positive samples
    #--------------------------------------------------#
    num_positive_anchors = positive_indices.sum()
    return targets, num_positive_anchors, positive_indices, assigned_annotations

def encode_bbox(assigned_annotations, positive_indices, anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y):
    #--------------------------------------------------#
    #   Take out the real box corresponding to the prior box of positive sample
    #--------------------------------------------------#
    assigned_annotations = assigned_annotations[positive_indices, :]

    #--------------------------------------------------#
    #   Take out the prior box of positive sample
    #--------------------------------------------------#
    anchor_widths_pi = anchor_widths[positive_indices]
    anchor_heights_pi = anchor_heights[positive_indices]
    anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
    anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

    #--------------------------------------------------#
    #   Calculate the width, height and center of the real box
    #--------------------------------------------------#
    gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
    gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
    gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
    gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

    gt_widths = torch.clamp(gt_widths, min=1)
    gt_heights = torch.clamp(gt_heights, min=1)

    #---------------------------------------------------#
    #   Real box and prior box are used to encode and the predicted results are obtained
    #---------------------------------------------------#
    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
    targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
    targets_dw = torch.log(gt_widths / anchor_widths_pi)
    targets_dh = torch.log(gt_heights / anchor_heights_pi)

    targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
    targets = targets.t()
    return targets

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations, alpha = 0.25, gamma = 2.0, cuda = True):
        #---------------------------#
        #   Get the batch_size
        #---------------------------#
        batch_size = classifications.shape[0]

        #--------------------------------------------#
        #   Get prior box and transform it into the format of center,width and height
        #--------------------------------------------#
        dtype = regressions.dtype
        anchor = anchors[0, :, :].to(dtype)
        #--------------------------------------------#
        #   Transform the prior box into the format of center,width and height
        #--------------------------------------------#
        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        regression_losses = []
        classification_losses = []
        for j in range(batch_size):
            #-------------------------------------------------------#
            #   Take the real box, classification result and regression result corresponding to each picture
            #-------------------------------------------------------#
            bbox_annotation = annotations[j]
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            
            if len(bbox_annotation) == 0:
                #-------------------------------------------------------#
                #   When there is no real box in the picture, all feature points are negative samples
                #-------------------------------------------------------#
                alpha_factor = torch.ones_like(classification) * alpha
                if cuda:
                    alpha_factor = alpha_factor.cuda()
                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                
                #-------------------------------------------------------#
                #   Calculate the cross entropy corresponding to the feature points
                #-------------------------------------------------------#
                bce = - (torch.log(1.0 - classification))
                
                cls_loss = focal_weight * bce
                
                classification_losses.append(cls_loss.sum())
                #-------------------------------------------------------#
                #   The regression loss is 0
                #-------------------------------------------------------#
                if cuda:
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))
                    
                continue

            #------------------------------------------------------#
            #   Calculate the iou between teh real box and prior box.
            #   targets                 num_anchors, num_classes
            #   num_positive_anchors    正样本的数量
            #   positive_indices        num_anchors, 
            #   assigned_annotations    num_anchors, 5
            #------------------------------------------------------#
            targets, num_positive_anchors, positive_indices, assigned_annotations = get_target(anchor, 
                                                                                        bbox_annotation, classification, cuda)
            
            #------------------------------------------------------#
            #   Calculate cross entropy loss
            #------------------------------------------------------#
            alpha_factor = torch.ones_like(targets) * alpha
            if cuda:
                alpha_factor = alpha_factor.cuda()


            #------------------------------------------------------#
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = - (targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            cls_loss = focal_weight * bce

            #------------------------------------------------------#
            #   Set loss of the ignored prior box to 0
            #------------------------------------------------------#
            zeros = torch.zeros_like(cls_loss)
            if cuda:
                zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))
            
            #------------------------------------------------------#
            #   If there is a prior box corresponding to a positive sample
            #------------------------------------------------------#
            if positive_indices.sum() > 0:
                targets = encode_bbox(assigned_annotations, positive_indices, anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y)
                #---------------------------------------------------#
                #   Compare the predicted results of the network with the actual predicted results
                #   Calculate smooth l1 loss
                #---------------------------------------------------#
                regression_diff = torch.abs(targets - regression[positive_indices, :])
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if cuda:
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))
        
        # Average Loss is calculated and returned
        c_loss = torch.stack(classification_losses).mean()
        r_loss = torch.stack(regression_losses).mean()
        loss = c_loss + r_loss
        return loss, c_loss, r_loss

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

class LossHistory():
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = log_dir
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))
