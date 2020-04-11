import face_alignment
from skimage import io
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import *

# Measuring distance
def dist_3d(p1, p2, preds):
    x1 = preds[0][p1][0]
    y1 = preds[0][p1][1]
    z1 = preds[0][p1][2]

    x2 = preds[0][p2][0]
    y2 = preds[0][p2][1]
    z2 = preds[0][p2][2]

    d = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    return d

def algorithm(preds, ppm=1, age=4, weight=16):
    # Get the iris width dimension if form of pixel per metric(ppm) ratio and convert all the distance ratios into real dimensions(in mm) 
    # Then put in this algorithm


    ## Algorithm 
    # For Calulating Bulb Size

    # 1. Age - Take inputs from APP
    # Suppose 
    age = 4  # in months
    # weightage for age
    w_1 = 0.35

    if (age >=0 and age <=3):
        bulb_size_1 = 1
    elif (age >= 3 and age <= 9):
        bulb_size_1 = 2
    else:
        bulb_size_1 = 3
        
    # 2. Weight - Take inputs from APP
    # Suppose 
    weight = 16  # in lbs
    # weightage for weight
    w_2 = 0.05

    if (weight <= 17):
        bulb_size_2 = 1
    elif (weight >= 17 and weight <= 25):
        bulb_size_2 = 2
    else:
        bulb_size_2 = 3

    # 3. Inferior Facial Width [Go-Go] (in mm)
    # Convert the ratio to pixel per metric ratio
    go_go = dist_3d(12, 4, preds) 
    # weightage for [go-go]
    w_3 = 0.10

    if (go_go <= 77):
        bulb_size_3 = 1
    elif (go_go >= 77 and go_go <= 81.5):
        bulb_size_3 = 2
    else:
        bulb_size_3 = 3

    # 4. Width of mouth [ch-ch] (in mm)
    # Convert the ratio to pixel per metric ratio
    ch_ch = dist_3d(45, 54, preds)
    # weightage for [ch-ch]
    w_4 = 0.05

    if (ch_ch <= 28.2):
        bulb_size_4 = 1
    elif (ch_ch >= 28.2 and ch_ch <= 35.3):
        bulb_size_4 = 2
    else:
        bulb_size_4 = 3

    # 5. Width of face [zy-zy] (in mm)
    # Convert the ratio to pixel per metric ratio
    zy_zy = dist_3d(16, 0, preds)
    # weightage for [zy-zy]
    w_5 = 0.10

    if (zy_zy <= 89.5):
        bulb_size_5 = 1
    elif (zy_zy >= 89.5 and zy_zy <= 97.3):
        bulb_size_5 = 2
    else:
        bulb_size_5 = 3

    # 6. Width of palate
    w_6 = 0.45
    if (age <= 2):
        bulb_size_6 = 1
    elif (age >= 2 and age <= 6):
        bulb_size_6 = 2
    else:
        bulb_size_6 = 3

    # Weighted average
    bulb_size = (w_1*bulb_size_1 + w_2*bulb_size_2 + w_3*bulb_size_3 + w_4*bulb_size_4 + w_5*bulb_size_5 + w_6*bulb_size_6)/(w_1 + w_2 + w_3 + w_4 + w_5 + w_6)     

    # TWI Biometric Shield (Only for Tony Boom)
    # Mandibular Index 
    mand_i = dist_3d(14, 27, preds)/dist_3d(14, 8, preds)
    if (mand_i > 15.6):
        shield_type = "biometric"
    else:
        shield_type = "regular"

    return int(bulb_size), shield_type



if __name__ == '__main__':

    # Face alignment algorithm
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, face_detector='sfd', device='cpu')

    # Inputing the image
    input_image = io.imread('Baby-1.jpg')
    preds = fa.get_landmarks_from_image(input_image)
    preds = np.asarray(preds)
    
    bulb_size, shield_type = algorithm(preds)

    print("bulb_size : ", bulb_size, ", shield_type : ", shield_type)

