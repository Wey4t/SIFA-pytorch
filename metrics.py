import numpy as np
from medpy import metric

def dice_eval(predict,label,num_classes):
    #Computer Dice coefficient
    dice = np.zeros(num_classes)
    eps = 1e-7
    for c in range(num_classes):
        inter = 2.0 * (np.sum((predict==c)*(label==c),dtype=np.float32))
        p_sum = np.sum(predict==c,dtype=np.float32)
        gt_sum = np.sum(label==c,dtype=np.float32)
        dice[c] = (inter+eps)/(p_sum+gt_sum+eps)
    return dice[1:]
    
def assd_eval(predict,label,num_classes):
    #Average Symmetric Surface Distance (ASSD)
    assd_all = np.zeros(num_classes)
    for c in range(num_classes):
        reference = (label==c) * 1
        result = (predict==c) * 1
        assd_all[c] = metric.binary.assd(result,reference)
    return assd_all[1:]
    
def create_visual_anno(anno):
    # assert np.max(anno) < 7 # only 7 classes are supported, add new color in label2color_dict
    label2color_dict = {
        0:  [0, 0, 0],         # background
        1:  [0, 0, 255],       # blue
        2:  [0, 255, 0],       # green
        3:  [255, 0, 0],       # red
        4:  [255, 255, 0],     # yellow
        5:  [255, 0, 255],     # magenta
        6:  [0, 255, 255],     # cyan
        7:  [128, 0, 0],       # maroon
        8:  [0, 128, 0],       # dark green
        9:  [0, 0, 128],       # navy
        10: [128, 128, 0],     # olive
        11: [128, 0, 128],     # purple
        12: [0, 128, 128],     # teal
        13: [192, 192, 192],   # silver/gray
    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno