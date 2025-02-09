import cv2
# 导入必要的库
import cv2
import numpy as np
from skimage import feature, metrics
import random

def read_pair(path_1,path_2):
    """
    Read a pair of images from the assigned path.
    Return a pair of arrays representing the gray graph
    """
    image1 = cv2.imread(path_1)
    image2 = cv2.imread(path_2)

    # 调整图片大小，使它们的尺寸相同
    image2 = cv2.resize(image2, (500, 500), interpolation=cv2.INTER_AREA)
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return image1_gray,image2_gray

def read_single(path_1,px):
    """
    Read a pair of images from the assigned path.
    Return a pair of arrays representing the gray graph
    """
    image1 = cv2.imread(path_1)


    # 调整图片大小，使它们的尺寸相同
    image1 = cv2.resize(image1, (px, px), interpolation=cv2.INTER_AREA)
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    return image1_gray

def rotate_single(path_1,px):
    """
    Read a pair of images from the assigned path.
    Return a pair of arrays representing the gray graph
    """
    u = random.random()
    image1 = cv2.imread(path_1)
    if u<0.5:
        image1 = cv2.rotate(image1,cv2.ROTATE_180)
    else:
        image1 = cv2.rotate(image1,cv2.ROTATE_90_CLOCKWISE)

    # 调整图片大小，使它们的尺寸相同
    image1 = cv2.resize(image1, (px, px), interpolation=cv2.INTER_AREA)
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    return image1_gray



def cosine_similarity(vec1, vec2):
    vec1=vec1.flatten().squeeze()
    vec2 = vec2.flatten().squeeze()
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cos_sim = dot_product / (norm1 * norm2)
    return cos_sim


def BFmatch(des1, des2):
    """使用KNN算法进行匹配，如果dis为false，直接匹配，默认为True"""
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    return len(good)

def get_SIFT_match(image1_gray,image2_gray):
    sift = cv2.SIFT_create()
    """
    detectAndCompute: Detects keypoints and computes the descriptors
    kp: Key points; des: descriptor(or lets say, feature vector)
    """
    kp1, des1 = sift.detectAndCompute(image1_gray, None)
    kp2, des2 = sift.detectAndCompute(image2_gray, None)
    matches = BFmatch(des1, des2)
    return matches


def get_HOG_sim(image1_gray,image2_gray):
    hog = cv2.HOGDescriptor()
    """
    des: descriptor(or lets say, feature vector)
    """
    des1 = hog.compute(image1_gray)
    des2 = hog.compute(image2_gray)
    cos_sim = cosine_similarity(des1, des2)
    return cos_sim

def get_LBP_sim(image1_gray,image2_gray):
    """
    P : int
        Number of circularly symmetric neighbour set points (quantization of
        the angular space).
    R : float
        Radius of circle (spatial resolution of the operator).
    """
    des1 = feature.local_binary_pattern(image1_gray, P=8, R=1, method='uniform')
    des2 = feature.local_binary_pattern(image2_gray, P=8, R=1, method='uniform')
    cos_sim = cosine_similarity(des1, des2)
    return cos_sim
