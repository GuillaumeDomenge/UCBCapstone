import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import os,glob
from skimage.metrics import structural_similarity as ssim
from utils import *


table = pd.read_csv("learnset_50_50_5scores.csv", sep=",")
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


def get_images(row):
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
    count1=0
    for file in glob.glob("*.TIF"):
        count1+=1
    #set1[0] = cv2.resize(set1[0], (500,500))
    #set1 = [make_same_dimension(set1[0],i)[1] for i in set1]
    set2 = []
    os.chdir(filepath2)
    count2 = 0
    for file in glob.glob("*.TIF"):
        count2+=1
    #set2 = [make_same_dimension(set1[0],i)[1] for i in set2]
   

    #print(scores)
    
    return count1,count2

oldpwd = os.getcwd()
#df = df.head(1)
df["Num_Im_1"],df["Num_Im_2"] = zip(*df.apply(get_images, axis=1))
os.chdir(oldpwd)
df.to_csv("learnset_50_50_5scores2.csv",index=False)
