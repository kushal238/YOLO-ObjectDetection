import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou



class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """

        new_boxes = torch.randn_like(boxes)
        
        new_boxes[:,0] = boxes[:,0]/self.S - 0.5*boxes[:,2]
        new_boxes[:,1] = boxes[:,1]/self.S - 0.5*boxes[:,3]
        
        new_boxes[:,2] = boxes[:,0]/self.S + 0.5*boxes[:,2]
        new_boxes[:,3] = boxes[:,1]/self.S + 0.5*boxes[:,3]
        
        return new_boxes

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : (list) [(tensor) size (-1, 5)]  
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """
        best_ious = torch.full((pred_box_list[0].shape[0], 1), -1.)
        best_boxes = torch.zeros(pred_box_list[0].shape[0], 5)
        box_target_trans = self.xywh2xyxy(box_target)
        
        for B in range(len(pred_box_list)):
            ret = compute_iou(self.xywh2xyxy(pred_box_list[B][:,:4]), box_target_trans)
            for j in range(pred_box_list[0].shape[0]):
                if ret[j,j] > best_ious[j,0]:
                    best_ious[j, 0] = ret[j,j]
                    best_boxes[j,:] = pred_box_list[B][j,:]
            
        best_ious = best_ious.detach()
        return best_ious, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """

        running_loss= 0
    
        diff_tensor_sq = (classes_pred - classes_target)**2
        sum_tensor = torch.sum(diff_tensor_sq, dim=-1)
        running_loss = (sum_tensor * has_object_map).sum()                
        
        loss = running_loss/classes_pred.shape[0]
        return loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """

        N = pred_boxes_list[0].shape[0]
        S = pred_boxes_list[0].shape[1]
        
        bb_runnin = torch.zeros(N,S,S)
        not_has_object_map = torch.logical_not(has_object_map)
        
        for bbox in range(len(pred_boxes_list)):
            dif_tensor_sq = (0 - pred_boxes_list[bbox][:,:,:,4])**2
            no_object_tensor = dif_tensor_sq * not_has_object_map
#             print(bb_runnin.device)
#             print(no_object_tensor.device)
            if bb_runnin.device.type == 'cpu':
                bb_runnin = bb_runnin.to("cuda:0")
            bb_runnin += no_object_tensor
        
        loss = bb_runnin.sum()/N
        
        return loss*self.l_noobj

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """

        loss = ((box_pred_conf - box_target_conf)**2).sum()
        return loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """

        if box_pred_response.device.type == 'cpu':
            box_pred_response = box_pred_response.to('cuda:0')
        x_loss = (((box_pred_response[:,0] - box_target_response[:,0])**2).sum())
        y_loss = (((box_pred_response[:,1] - box_target_response[:,1])**2).sum())
        
        w_loss = ((torch.sqrt(box_pred_response[:,2]) - torch.sqrt(box_target_response[:,2]))**2).sum()
        h_loss = ((torch.sqrt(box_pred_response[:,3]) - torch.sqrt(box_target_response[:,3]))**2).sum()
        
        reg_loss = self.l_coord*(x_loss + y_loss + w_loss + h_loss)
        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) where:  
                            N - batch_size
                            S - width/height of network output grid
                            B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        total_loss = 0.0

        # splitting the pred tensor from an entity to separate tensors
        pred_boxes_list = []
        for i in range(self.B):
            pred_boxes_list.append(pred_tensor[:, :, :, (i*5):((i+1)*5)])
            
        pred_cls = pred_tensor[:,:,:,(self.B*5):]

        # compcuting classification loss
        classification_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)

        # computing no-object loss
        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)

        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keeping having-object cells
        # 2) vectorising all dimensions except for the last one for faster computation
        has_object_map_new = has_object_map.unsqueeze(-1)
        has_object_map_new = has_object_map_new.expand(-1,-1,-1,5)

        target_boxes = target_boxes[has_object_map_new[:,:,:,:4] == True].view(-1,4)
        for i in range(len(pred_boxes_list)):
            pred_boxes_list[i] = pred_boxes_list[i][has_object_map_new == True].view(-1,5)
        
        # finding the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        best_ious, best_boxes = self.find_best_iou_boxes(pred_boxes_list, target_boxes)

        # computinh regression loss between the found best bbox and GT bbox for all the cell containing objects
        regression_loss = self.get_regression_loss(best_boxes[:,:4], target_boxes)/N

        # computing contain_object_loss
        contain_object_loss = self.get_contain_conf_loss(best_boxes[:,-1], best_ious[:,-1])/N
        
        # computing final loss
        final_loss = classification_loss + no_obj_loss + regression_loss + contain_object_loss
        # constructing return loss_dict
        loss_dict = dict(
            total_loss=final_loss,
            reg_loss=regression_loss,
            containing_obj_loss=contain_object_loss,
            no_obj_loss=no_obj_loss,
            cls_loss=classification_loss,
        )
        return loss_dict