import numpy as np
import sys, os
from .pyeval.evaluateAPAOS import evaluateDetectionAPAOS
sys.path.append(os.getcwd())

def evaluate_rcll_prec_moda_modp(res_fpath, gt_fpath):
 
    from utils.evaluation.pyeval.evaluateDetection import evaluateDetection_py

    recall, precision, moda, modp = evaluateDetection_py(res_fpath, gt_fpath)

    return recall, precision, moda, modp

def evaluate_ap_aos(res_fpath, gt_fpath):
    AP_75, AOS_75, OS_75, AP_50, AOS_50, OS_50, AP_25, AOS_25, OS_25 = evaluateDetectionAPAOS(res_fpath, gt_fpath)
    return AP_75, AOS_75, OS_75, AP_50, AOS_50, OS_50, AP_25, AOS_25, OS_25
