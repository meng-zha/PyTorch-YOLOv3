from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, root_path, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(root_path, path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class, pre_cur, rec_cur = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class, pre_cur, rec_cur


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--mode", type=str, default='val', help="eval or test")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    if opt.mode=='val':
        valid_path = data_config["valid"]
        root = data_config["root"]
    else:
        valid_path = data_config["test"]
        root = data_config["test_root"]

    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    mAP = []
    pre = []
    rec = []
    all_mAP_0 = []
    all_mAP_1 = []
    for i,iou in enumerate(np.arange(0.5,0.95,0.05)):
        precision, recall, AP, f1, ap_class, pre_cur, rec_cur = evaluate(
            model,
            root_path=root,
            path=valid_path,
            iou_thres=iou,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            img_size=opt.img_size,
            batch_size=8,
        )
        all_mAP_0.append(AP[0])
        all_mAP_1.append(AP[1])
        if i == 0 or i==4 or i==8:
            mAP.append(AP)
            pre.append(pre_cur)
            rec.append(rec_cur)

    os.makedirs("output/yolov3", exist_ok=True)
    for i,cl in enumerate(ap_class):
        pre_5 = pre[0][i]
        pre_5 = np.concatenate([pre_5,[0]],axis=0)
        pre_7 = pre[1][i]
        pre_7 = np.concatenate([pre_7,[0]],axis=0)
        pre_9 = pre[2][i]
        pre_9 = np.concatenate([pre_9,[0]],axis=0)
        rec_5 = rec[0][i]
        rec_5 = np.concatenate([rec_5,[rec_5[-1]]],axis=0)
        rec_7 = rec[1][i]
        rec_7 = np.concatenate([rec_7,[rec_7[-1]]],axis=0)
        rec_9 = rec[2][i]
        rec_9 = np.concatenate([rec_9,[rec_9[-1]]],axis=0)


        plt.clf()
        plt.plot(rec_5,pre_5,'m-',label='IoU=0.5')
        plt.plot(rec_7,pre_7,'b-',label='IoU=0.7')
        plt.plot(rec_9,pre_9,'c-',label='IoU=0.9')
        plt.xlim(0,1.0)
        plt.ylim(0,1.01)
        plt.title(f"precision-recall curve of {class_names[cl]}")
        plt.grid(True)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend(loc="lower left")

        name = os.path.join('output/yolov3',f'pr_curve_{class_names[cl]}.png')
        plt.savefig(name)

    print(f'mAP50:{mAP[0]}')
    print(f'mAP70:{mAP[1]}')
    print(f'mAP90:{mAP[2]}')
    print(f'mAP.5:.95:{np.array(all_mAP_0).mean()},{np.array(all_mAP_1).mean()}')
