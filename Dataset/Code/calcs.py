import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import os,glob
from skimage.metrics import structural_similarity as ssim


table = pd.read_csv("LSet_50_50.csv", sep=",")
df  = pd.DataFrame(table)
 
def compare_images(imageA, imageB):
    # Converting to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    # Calculating the SSIM of the two images
    score, _ = ssim(grayA, grayB, full=True)
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
        set1.append(cv2.imread(file))
    set1[0] = cv2.resize(set1[0], (500,500))
    set1 = [make_same_dimension(set1[0],i)[1] for i in set1]
    set2 = []
    os.chdir(filepath2)
    for file in glob.glob("*.TIF"):
        set2.append(cv2.imread(file))
    set2 = [make_same_dimension(set1[0],i)[1] for i in set2]
    n1 = len(set1)
    n2 = len(set2)
    scores = []
    with tqdm(total=n1*n2) as p_bar:
        for i in range(0,n1):
            for j in range(0,n2):
                scores.append(compare_images(set1[i],set2[j]))
                p_bar.update(1)
    #print(scores)            
    scores = sorted(scores, reverse=True)
    #print(scores)
    print("top1: %s,top2: %s, top3: %s, top4: %s, top5: %s, avg: %s, std: %s, worse: %s" % (scores[0],scores[1],scores[2],scores[3],scores[4], sum(scores)/len(scores), np.std(scores), scores[-1]))
    return scores[0],scores[1],scores[2],scores[3],scores[4], sum(scores)/len(scores), np.std(scores), scores[-1]

oldpwd = os.getcwd()
#df = df.head(1)
df["top1"],df["top2"],df["top3"],df["top4"],df["top5"],df["avg"],df["std"],df["worse"] = zip(*df.apply(get_scores, axis=1))
os.chdir(oldpwd)
df.to_csv("learnset_50_50.csv",index=False)
