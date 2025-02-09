import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import os,glob
from skimage.metrics import structural_similarity as ssim
from utils import *


table = pd.read_csv("LSet_50_50.csv", sep=",")
df  = pd.DataFrame(table)
 
def compare_images(imageA, imageB):
    # Converting to grayscale
    #grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    #grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    # Calculating the SSIM of the two images
    score, _ = ssim(imageA, imageB, full=True)
    return score


def make_same_dimension(image1, image2):
    # Adjust the size to be the same as the first one
    resized_image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    # Make sure the two pictures are of same color channels
    if len(image1.shape) == 2: # if image1 is one channel
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(resized_image2.shape) == 2: # if image1 is two channels
        resized_image2 = cv2.cvtColor(resized_image2, cv2.COLOR_GRAY2BGR)
    return image1, resized_image2


def get_scores(row):
    p1 = str(row["patent_1_id"])
    #if p1[0]=="D":
    #    p1 = "D0"+p1.strip("D")
    p2 = str(row["patent_2_id"])
    d1 = row["patent_1_date"].replace("-","")
    #if p2[0]=="D":
    #    p2 = "D0"+p2.strip("D")
    d2 = row["patent_2_date"].replace("-","")
    filepath1 = "/home/guillaume/Desktop/Captsone suff/Take 2/OG Patents and Inventor ID/Final_Folders/US"+p1+"-"+d1
    filepath2 = "/home/guillaume/Desktop/Captsone suff/Take 2/OG Patents and Inventor ID/Final_Folders/US"+p2+"-"+d2
    set1 = []
    print(filepath1)
    os.chdir(filepath1)
    for file in glob.glob("*.TIF"):
        set1.append(read_single(file))
    #set1[0] = cv2.resize(set1[0], (500,500))
    #set1 = [make_same_dimension(set1[0],i)[1] for i in set1]
    set2 = []
    os.chdir(filepath2)
    for file in glob.glob("*.TIF"):
        set2.append(read_single(file))
    #set2 = [make_same_dimension(set1[0],i)[1] for i in set2]
    n1 = len(set1)
    n2 = len(set2)
    scoresLBP = []
    scoresHOG = []
    scoresSIFT = []
    scoresSSIM = []
    scoresBF = []
    with tqdm(total=n1*n2) as p_bar:
        for i in range(0,n1):
            for j in range(0,n2):
                #print(set1[i])
                scoresLBP.append(get_LBP_sim(set1[i],set2[j]))
                scoresHOG.append(get_HOG_sim(set1[i],set2[j]))
                scoresSIFT.append(get_SIFT_match(set1[i],set2[j]))
                scoresSSIM.append(compare_images(set1[i],set2[j]))
                scoresBF.append(BFmatch(set1[i],set2[j]))
                p_bar.update(1)
    #print(scores)            
    scoresLBP = sorted(scoresLBP, reverse=True)
    scoresHOG = sorted(scoresHOG, reverse=True)
    scoresSIFT = sorted(scoresSIFT, reverse=True)
    scoresSSIM = sorted(scoresSSIM, reverse=True)
    scoresBF = sorted(scoresBF, reverse=True)

    #print(scores)
    print("avg LBP: %s, avg HOG: %s,avg SIFT: %s,avg SSIM: %s, avg BF: %s," % (sum(scoresLBP)/len(scoresLBP),sum(scoresHOG)/len(scoresHOG),sum(scoresSIFT)/len(scoresSIFT),sum(scoresSSIM)/len(scoresSSIM),sum(scoresBF)/len(scoresBF)))
    return scoresLBP[0],scoresLBP[1],scoresLBP[2], sum(scoresLBP)/len(scoresLBP), np.std(scoresLBP), scoresLBP[-1], scoresHOG[0],scoresHOG[1],scoresHOG[2], sum(scoresHOG)/len(scoresHOG), np.std(scoresHOG), scoresHOG[-1], scoresSIFT[0],scoresSIFT[1],scoresSIFT[2], sum(scoresSIFT)/len(scoresSIFT), np.std(scoresSIFT), scoresSIFT[-1],scoresSSIM[0],scoresSSIM[1],scoresSSIM[2], sum(scoresSSIM)/len(scoresSSIM), np.std(scoresSSIM), scoresSSIM[-1], scoresBF[0],scoresBF[1],scoresBF[2], sum(scoresBF)/len(scoresBF), np.std(scoresBF), scoresBF[-1]

oldpwd = os.getcwd()
#df = df.head(1)
df["LBP_top1"],df["LBP_top2"],df["LBP_top3"],df["LBP_avg"],df["LBP_std"],df["LBP_worse"],df["HOG_top1"],df["HOG_top2"],df["HOG_top3"],df["HOG_avg"],df["HOG_std"],df["HOG_worse"],df["SIFT_top1"],df["SIFT_top2"],df["SIFT_top3"],df["SIFT_avg"],df["SIFT_std"],df["SIFT_worse"],df["SSIM_top1"],df["SSIM_top2"],df["SSIM_top3"],df["SSIM_avg"],df["SSIM_std"],df["SSIM_worse"],df["BF_top1"],df["BF_top2"],df["BF_top3"],df["BF_avg"],df["BF_std"],df["BF_worse"] = zip(*df.apply(get_scores, axis=1))
os.chdir(oldpwd)
df.to_csv("learnset_50_50_5scores.csv",index=False)
