import face_alignment
from skimage import io
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, face_detector='sfd', device='cpu')

input = io.imread('Baby_Face.jpg')
preds = fa.get_landmarks_from_image(input)
preds = np.asarray(preds)
print(preds)

print("preds shape : ", preds.shape)
print(preds[0][0][0], preds[0][0][1], preds[0][0][2] )
