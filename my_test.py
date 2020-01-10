import face_alignment
from skimage import io
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

"""
# Detect 2D facial landmarks in picture
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

input = io.imread('Baby-1.jpg')
preds = fa.get_landmarks(input)
print(preds)

#  Plotting in matplotlib in 2D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(preds[0][:,0], preds[0][:,1], zdir='z', s=20, c=None, depthshade=True)
#ax.plot(preds[0][:,0], preds[0][:,1], color=None)
plt.show()
"""

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, face_detector='sfd', device='cpu')

#input = io.imread('/home/cmb/singularity/toothprints-pc/landmarks/face-alignment/test/assets/aflw-test.jpg')
input = io.imread('Baby-2.jpg')
preds = fa.get_landmarks_from_image(input)
preds = np.asarray(preds)
print(preds)

print("preds shape : ", preds.shape)
print(preds[0][0][0], preds[0][0][1], preds[0][0][2] )

#  Plotting in matplotlib in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(preds[0][:,0], preds[0][:,1], preds[0][:,2], zdir='z', s=20, c=None, depthshade=True)
#ax.plot(preds[0][:,0], preds[0][:,1], preds[0][:,2], color=None)  # For connecting points 

p1 = 60
p2 = 61
p3 = 62
p4 = 63
p5 = 64
p6 = 65
# Marking separate point
ax.scatter3D(preds[0][p1][0], preds[0][p1][1], preds[0][p1][2], zdir='z', s=20, c='g', depthshade=True)
ax.scatter3D(preds[0][p2][0], preds[0][p2][1], preds[0][p2][2], zdir='z', s=20, c='r', depthshade=True)
ax.scatter3D(preds[0][p3][0], preds[0][p3][1], preds[0][p3][2], zdir='z', s=20, c='m', depthshade=True)
ax.scatter3D(preds[0][p4][0], preds[0][p4][1], preds[0][p4][2], zdir='z', s=20, c='y', depthshade=True)
ax.scatter3D(preds[0][p5][0], preds[0][p5][1], preds[0][p5][2], zdir='z', s=20, c='k', depthshade=True)
ax.scatter3D(preds[0][p6][0], preds[0][p6][1], preds[0][p6][2], zdir='z', s=20, c='b', depthshade=True)

plt.xlabel('X-axis', fontdict=None, labelpad=None)
plt.ylabel('Y-axis', fontdict=None, labelpad=None)
#plt.zlabel('Z-axis', fontdict=None, labelpad=None)
plt.show()


