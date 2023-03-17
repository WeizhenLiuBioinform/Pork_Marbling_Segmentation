import cv2
import os
from skimage import io as skio
import numpy as np
from scipy import ndimage
# 计算指标
def IOU(flst,mask_pred_path, mask_gt_path):
    files= os.listdir(flst)
    fat_recall = 0
    fat_precision = 0
    fat_iou = 0
    mean_fat_recall = 0
    mean_fat_precision = 0
    mean_fat_iou = 0
    count = 0
    Total_TP = 0
    Total_FN = 0
    Total_FP = 0
    FN = 0
    FP = 0
    for f in files:
        gt_path = os.path.join(mask_gt_path, f[2:-11]+".png")
        pred_path = os.path.join(mask_pred_path, f)
        mask_gt = cv2.imread(gt_path,0)
        mask_pred = cv2.imread(pred_path,0)
        pred_ins = mask_pred == 38
        target_ins = mask_gt == 38
        inser = pred_ins[target_ins].sum()
        Total_TP = Total_TP + inser
        FN = target_ins.sum() - inser
        FP = pred_ins.sum() - inser
        Total_FN = Total_FN + FN
        Total_FP = Total_FP + FP
        union = pred_ins.sum() + target_ins.sum() - inser
        iou = inser / union #TP/(TP+FN+FP)
        precision = inser / pred_ins.sum() #TP/(TP+FP)
        recall = inser / target_ins.sum() #TP/(TP+FN) 
        print("%s"%f)
        print("major_fat val_data iou : {:.4f}, precision: {:.4f}, recall: {:.4f}".format(iou,precision,recall))
        fat_recall = fat_recall + recall
        fat_precision = fat_precision + precision
        fat_iou = fat_iou + iou
        
    
    mean_iou = float(Total_TP/(Total_FN+Total_FP+Total_TP))
    mean_precision = float(Total_TP/(Total_TP+Total_FP))
    mean_recall = float(Total_TP/(Total_TP+Total_FN))
    F1_Score = float(2*mean_recall*mean_precision/(mean_recall+mean_precision))
    print('\n')
    print("major_fat val_data mean_iou : {:.3f}, mean_precision: {:.3f}, mean_recall: {:.3f}, F1_score: {:.4f}".format(mean_iou,
    mean_precision,mean_recall,F1_Score))
    print('\n')

def mean_Dist(img_src,img_des):
    mean_dist = 0
    tmp = ndimage.distance_transform_edt(img_des == 0)

    mean_dist =np.sum(np.sum(tmp * img_src)) / np.sum(img_src[:])

    return mean_dist

def access_dev(flst,mask_pred_path, mask_gt_path, lable):
    files= os.listdir(flst)
    mean_dist_corr = 0
    mean_dist_com = 0
    all_dist_corr = 0
    all_dist_com = 0
    count = 0
    for f in files:
        gt_path = os.path.join(mask_gt_path, f[2:-11]+".png")
        pred_path = os.path.join(mask_pred_path, f)
        mask_gt = skio.imread(gt_path)
        mask_pred = skio.imread(pred_path)
        img_src = mask_gt[:,:,lable].astype(np.uint8) / 128
        img_des = mask_pred[:,:,lable].astype(np.uint8) / 128
        dist_corr = mean_Dist(img_des, img_src) #平均距离准确性
        dist_com = mean_Dist(img_src,img_des) #平均距离完整性  欧氏距离
        all_dist_corr = all_dist_corr + dist_corr
        all_dist_com = all_dist_com +dist_com
        count=count + 1

    mean_dist_corr = float(all_dist_corr/count)
    mean_dist_com = float(all_dist_com/count)
    print('\n')
    print("mean_dist_corr : {:.4f}, mean_dist_com: {:.4f}".format(mean_dist_corr,mean_dist_com))
    print('\n')



if __name__ == "__main__":
    flst = r'/home/zhangsf/code/saved/marblenet/02-28_18-19/predicted'
    save_path_cherry =  '/home/zhangsf/code/saved/marblenet/02-28_18-19/predicted'
    ground_path = '/home/zhangsf/code/saved/label_crop/'
    mask_gt_path = ground_path
    mask_pred_path = save_path_cherry
    IOU(flst,mask_pred_path,mask_gt_path)
    # access_dev(flst,mask_pred_path,mask_gt_path,0)
    