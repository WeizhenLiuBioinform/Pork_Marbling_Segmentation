import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import class_weight 
from utils.lovasz_losses import lovasz_softmax
from scipy.ndimage import distance_transform_edt


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    #cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        # weight = torch.tensor([0.51,32.9])
        # weight = weight.to(device = 'cuda:3')
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.BCE = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)

    def forward(self, output, target):
        if output.size()[1] == 1:
            loss = self.BCE(torch.squeeze(output, dim=1), target.float()) # 当二分类(前景、背景)num_classes=1时, 使用BCEloss
        else:
            loss = self.CE(output, target)
        return loss

class Weight_CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(Weight_CrossEntropyLoss2d, self).__init__()
        weight=torch.tensor([0.51,32.9]).to(device = 'cuda:0')
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.BCE = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)

    def forward(self, output, target):
        
        if output.size()[1] == 1:
            loss = self.BCE(torch.squeeze(output, dim=1), target.float()) # 当二分类(前景、背景)num_classes=1时, 使用BCEloss
        else:
            loss = self.CE(output, target)
        return loss
    
class Weight_CE_LovaszSoftmaxLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(Weight_CE_LovaszSoftmaxLoss, self).__init__()
        self.smooth = smooth
        weight=torch.tensor([1,61.2]).to(device = 'cuda:2')
        self.weight_cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
        self.lovasz_softmax = LovaszSoftmax()
    
    def forward(self, output, target):
        CE_loss = self.weight_cross_entropy(output, target)
        lovas_loss = self.lovasz_softmax(output, target)
        return CE_loss + lovas_loss
    


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()

class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss

class CE_DiceLoss_Coanet(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None,batch=True):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.batch = batch
        self.dice = self.soft_dice_loss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
    
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss

    

class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
    
    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss

class CE_LovaszSoftmaxLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_LovaszSoftmaxLoss, self).__init__()
        self.smooth = smooth
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
        self.lovasz_softmax = LovaszSoftmax()
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        lovas_loss = self.lovasz_softmax(output, target)
        return CE_loss + lovas_loss
    
class CE_LovaszSoftmaxLoss_lamda(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_LovaszSoftmaxLoss_lamda, self).__init__()
        self.smooth = smooth
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
        self.lovasz_softmax = LovaszSoftmax()
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        lovas_loss = self.lovasz_softmax(output, target)
        return CE_loss + 0.05*lovas_loss

class LS_Focal(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(LS_Focal, self).__init__()
        self.smooth = smooth
        self.Focal_Loss = FocalLoss()
        self.lovasz_softmax = LovaszSoftmax()
    
    def forward(self, output, target):
        focal_loss = self.Focal_Loss(output, target)
        lovas_loss = self.lovasz_softmax(output, target)
        return focal_loss + lovas_loss



def customsoftmax(inp, multihotmask):
    """
    Custom Softmax
    """
    soft = F.softmax(inp)
    # This takes the mask * softmax ( sums it up hence summing up the classes in border
    # then takes of summed up version vs no summed version
    return torch.log(
        torch.max(soft, (multihotmask * (soft * multihotmask).sum(1, keepdim=True)))
    )

class ImgWtLossSoftNLL(nn.Module):
    """
    松弛损失
    Relax Loss
    """

    def __init__(self, classes, ignore_index=255, weights=None, upper_bound=1.0,
                 norm=False, ohem=False):
        super(ImgWtLossSoftNLL, self).__init__()
        self.weights = weights
        self.num_classes = classes
        self.ignore_index = ignore_index
        self.upper_bound = upper_bound
        self.norm = norm
        self.batch_weights = False
        # Number of epoch to use before turn off border restriction
        self.REDUCE_BORDER_EPOCH = -1

        self.fp16 = False
        self.ohem = ohem
        self.ohem_loss = OhemCrossEntropy2dTensor(self.ignore_index).cuda()

    def calculate_weights(self, target):
        """
        Calculate weights of the classes based on training crop
        """
        if len(target.shape) == 3:
            hist = np.sum(target, axis=(1, 2)) * 1.0 / target.sum()
        else:
            hist = np.sum(target, axis=(0, 2, 3)) * 1.0 / target.sum()
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist[:-1]

    def onehot2label(self, target):
        # a bug here
        label = torch.argmax(target[:, :-1, :, :], dim=1).long()
        label[target[:, -1, :, :]] = self.ignore_index
        return label

    def custom_nll(self, inputs, target, class_weights, border_weights, mask):
        """
        NLL Relaxed Loss Implementation
        """
        # if (self.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
        if (self.REDUCE_BORDER_EPOCH != -1):
            if self.ohem:
                return self.ohem_loss(inputs, self.onehot2label(target))
            border_weights = 1 / border_weights
            target[target > 1] = 1
        if self.fp16:
            loss_matrix = (-1 / border_weights *
                           (target[:, :-1, :, :].half() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :-1, :, :].half())).sum(1)) * \
                          (1. - mask.half())
        else:
            loss_matrix = (-1 / border_weights *
                           (target[:, :-1, :, :].float() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :-1, :, :].float())).sum(1)) * \
                          (1. - mask.float())

            # loss_matrix[border_weights > 1] = 0
        loss = loss_matrix.sum()

        # +1 to prevent division by 0
        loss = loss / (target.shape[0] * target.shape[2] * target.shape[3] - mask.sum().item() + 1)
        return loss

    def forward(self, inputs, target):
        # add ohem loss for the final stage
        # if (cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH) and self.ohem:
        if (self.REDUCE_BORDER_EPOCH != -1) and self.ohem:
            return self.ohem_loss(inputs, self.onehot2label(target[:,:-1,:,:]))
        if self.fp16:
            weights = target[:, :-1, :, :].sum(1).half()
        else:
            weights = target[:, :-1, :, :].sum(1).float()
        ignore_mask = (weights == 0)
        weights[ignore_mask] = 1
        loss = 0
        target_cpu = target.data.cpu().numpy()

        if self.batch_weights:
            class_weights = self.calculate_weights(target_cpu)

        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                class_weights = self.calculate_weights(target_cpu[i])
            loss = loss + self.custom_nll(inputs[i].unsqueeze(0),
                                          target[i].unsqueeze(0),
                                          class_weights=torch.Tensor(class_weights).cuda(),
                                          border_weights=weights, mask=ignore_mask[i])

        return loss

class OhemWithAux(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=10000, aux_weight=0.4):
        super(OhemWithAux, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.aux_weight = aux_weight
        self.main_loss = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)
        self.aux_loss = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)

    def forward(self, pred, target):
        x_main, x_aux = pred
        return self.main_loss(x_main, target) + self.aux_weight * self.aux_loss(x_aux, target)


class OhemCrossEntropy2dTensor(nn.Module):
    """
        Ohem Cross Entropy Tensor Version
    """
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=10000,
                 use_weight=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            # 权重怎么算的？？？？？
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       weight=weight,
                                                       ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       ignore_index=ignore_index)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class JointEdgeSegLoss(nn.Module):
    def __init__(self, classes, ignore_index=255,mode='train',
                 edge_weight=1, seg_weight=1, seg_body_weight=1, att_weight=1):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = classes
        if mode == 'train':
            self.seg_loss = OhemCrossEntropy2dTensor(ignore_index=ignore_index).cuda()
        elif mode == 'val':
            self.seg_loss = CrossEntropyLoss2d(size_average=True,
                                               ignore_index=ignore_index).cuda()

        self.seg_body_loss = ImgWtLossSoftNLL(classes=classes,
                                     ignore_index=ignore_index,
                                     upper_bound=1.0, ohem=False).cuda()
        self.edge_ohem_loss = OhemCrossEntropy2dTensor(ignore_index=ignore_index, min_kept=5000).cuda()

        self.ignore_index = ignore_index
        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.seg_body_weight = seg_body_weight


    def bce2d(self, input, target):
        n, c, h, w = input.size()

        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0


        weight = torch.from_numpy(weight).cuda()
        log_p = log_p.cuda()
        target_t = target_t.cuda()

        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        return loss

    def edge_attention(self, input, target, edge):
        filler = torch.ones_like(target) * 255
        return self.edge_ohem_loss(input, torch.where(edge.max(1)[0] > 0.8, target, filler))

    def forward(self, inputs, targets):
        seg_in, seg_body_in, edge_in = inputs
        seg_bord_mask, edgemask = targets
        segmask = self.onehot2label(seg_bord_mask)
        losses = {}

        losses['seg_loss'] = self.seg_weight * self.seg_loss(seg_in, segmask)
        losses['seg_body'] = self.seg_body_weight * self.seg_body_loss(seg_body_in, seg_bord_mask)
        losses['edge_loss'] = self.edge_weight * 20 * self.bce2d(edge_in, edgemask)
        losses['edge_ohem_loss'] = self.att_weight * self.edge_attention(seg_in, segmask, edge_in)

        return losses

    def onehot2label(self, target):
        """
        Args:
            target:
        Returns:
        """
        label = torch.argmax(target[:, :-1, :, :], dim=1).long()
        label[target[:, -1, :, :]] = self.ignore_index
        return label



################################################################
#                                                             ##
#https://github.com/JunMa11/SegLoss/tree/master/losses_pytorch##
#                                                             ##
################################################################

def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/dice_loss.py
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


class BDLoss(nn.Module):
    def __init__(self, wanted_classes):
        """
        wanted_classes: 计算想要计算的类的索引(比如不包括背景)
        compute boudary loss===>修改为支持多类，根据https://github.com/LIVIAETS/boundary-loss/issues/47
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BDLoss, self).__init__()
        self.wanted_classes = wanted_classes # 类别索引列表
        # self.do_bg = do_bg

    def forward(self, net_output, target, bound):
        """
        net_output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """
        net_output = softmax_helper(net_output)
        # print('net_output shape: ', net_output.shape)
        pc = net_output[:, self.wanted_classes, ...].type(torch.float32)
        dc = bound[:,self.wanted_classes, ...].type(torch.float32)

        multipled = torch.einsum("bcxyz,bcxyz->bcxyz", pc, dc) # 爱因斯坦加法
        bd_loss = multipled.mean()

        return bd_loss


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """

        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

class DC_and_BD_loss(nn.Module):
    """
    Dice Loss + Boundary Loss
    """
    def __init__(self, soft_dice_kwargs, bd_kwargs, aggregate="sum"):
        super(DC_and_BD_loss, self).__init__()
        self.aggregate = aggregate
        self.bd = BDLoss(**bd_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target, bound):
        dc_loss = self.dc(net_output, target)
        bd_loss = self.bd(net_output, target, bound)
        if self.aggregate == "sum":
            result = dc_loss + bd_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result       

class Adptive_weighted_DC_BD_loss(nn.Module):
    """
    @author luzx
    根据准确率、IOU自适应对每个类别进行Dice-loss和Boundary-loss的加权计算
    """
    def __init__(self,soft_dice_kwargs, bd_kwargs, aggregate="sum"):
        super(DC_and_BD_loss, self).__init__()
        self.aggregate = aggregate
        self.soft_dice_kwargs = soft_dice_kwargs
        self.bd_kwargs = bd_kwargs
        self.bd = BDLoss(**bd_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target, bound):
        wanted_classes = self.bd_kwargs['wanted_classes']
        pc = net_output[:, wanted_classes, ...].type(torch.float32)
        tc = net_output[:, wanted_classes, ...].type(torch.float32)

        intersection = torch.einsum("bcwh,bcwh->bc", pc, tc)
        union = (torch.einsum("bcwh->bc", pc) + torch.einsum("bcwh->bc", tc))
        iou = torch.sum(intersection & union, dim=(1, 2, 3)) / torch.sum(intersection | union, dim=(1, 2, 3))

        dc_loss = self.dc(net_output, target)
        bd_loss = self.bd(net_output, target, bound)
        if self.aggregate == "sum":
            result = dc_loss + bd_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result       

def compute_edts_forhdloss(segmentation):
    res = np.zeros(segmentation.shape)
    for i in range(segmentation.shape[0]):
        posmask = segmentation[i]
        negmask = ~posmask
        res[i] = distance_transform_edt(posmask) + distance_transform_edt(negmask)
    return res




def compute_edts_forPenalizedLoss(GT):
    """
    GT.shape = (batch_size, x,y,z)
    only for binary segmentation
    """
    res = np.zeros(GT.shape)
    for i in range(GT.shape[0]):
        posmask = GT[i]
        negmask = ~posmask
        pos_edt = distance_transform_edt(posmask)
        pos_edt = (np.max(pos_edt)-pos_edt)*posmask 
        neg_edt =  distance_transform_edt(negmask)
        neg_edt = (np.max(neg_edt)-neg_edt)*negmask
        
        res[i] = pos_edt/np.max(pos_edt) + neg_edt/np.max(neg_edt)
    return res

class DistBinaryDiceLoss(nn.Module):
    """
    Distance map penalized Dice loss
    Motivated by: https://openreview.net/forum?id=B1eIcvS45V
    Distance Map Loss Penalty Term for Semantic Segmentation        
    """
    def __init__(self, smooth=1e-5):
        super(DistBinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, net_output, gt):
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        net_output = softmax_helper(net_output)
        # one hot code for gt
        with torch.no_grad():
            if len(net_output.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)
        
        gt_temp = gt[:,0, ...].type(torch.float32)
        with torch.no_grad():
            dist = compute_edts_forPenalizedLoss(gt_temp.cpu().numpy()>0.5) + 1.0
        # print('dist.shape: ', dist.shape)
        dist = torch.from_numpy(dist)

        if dist.device != net_output.device:
            dist = dist.to(net_output.device).type(torch.float32)
        
        tp = net_output * y_onehot
        tp = torch.sum(tp[:,1,...] * dist, (1,2,3))
        
        dc = (2 * tp + self.smooth) / (torch.sum(net_output[:,1,...], (1,2,3)) + torch.sum(y_onehot[:,1,...], (1,2,3)) + self.smooth)

        dc = dc.mean()

        return -dc

