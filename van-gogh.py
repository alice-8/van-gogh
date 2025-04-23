# MVC Project Sem 2, Steven Strogatz Submission
# Alice Su, Per. 2
# Aprl 17, 2025

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

# load and resize image
img = Image.open("gogh.jpg").convert("RGB")
img = img.resize((100, 100))  # You can increase this if your computer is fast enough
pixels = np.array(img)

# create a 2D grid (the vector field)
Y, X = np.mgrid[0:100, 0:100]

# vector field centers for curl / divergence
cx_curl, cy_curl = 30, 30
cx_div_pos, cy_div_pos = 70, 30   # Positive divergence (outward)
cx_div_neg, cy_div_neg = 70, 70   # Negative divergence (inward)

# distances
dx_curl = X - cx_curl
dy_curl = Y - cy_curl
r_curl = np.sqrt(dx_curl**2 + dy_curl**2) + 1e-5

dx_div_pos = X - cx_div_pos
dy_div_pos = Y - cy_div_pos
r_div_pos = np.sqrt(dx_div_pos**2 + dy_div_pos**2) + 1e-5

dx_div_neg = X - cx_div_neg
dy_div_neg = Y - cy_div_neg
r_div_neg = np.sqrt(dx_div_neg**2 + dy_div_neg**2) + 1e-5

# define fields
U_curl = -dy_curl
V_curl = dx_curl

U_div_pos = dx_div_pos
V_div_pos = dy_div_pos

U_div_neg = -dx_div_neg
V_div_neg = -dy_div_neg

# strength of curl or divergence at each center
curl_strength = np.exp(-((X - cx_curl)**2 + (Y - cy_curl)**2) / 400)
div_pos_strength = np.exp(-((X - cx_div_pos)**2 + (Y - cy_div_pos)**2) / 300)
div_neg_strength = np.exp(-((X - cx_div_neg)**2 + (Y - cy_div_neg)**2) / 300)

# combine all fields
U = (curl_strength * U_curl +
     div_pos_strength * U_div_pos +
     div_neg_strength * U_div_neg)

V = (curl_strength * V_curl +
     div_pos_strength * V_div_pos +
     div_neg_strength * V_div_neg)

# normalize for direction only (so that each vectors magnitude is the same)
magnitude = np.sqrt(U**2 + V**2)
U = U / (magnitude + 1e-5)
V = V / (magnitude + 1e-5)

# get RGB colors from image
colors = pixels / 255.0
colors = colors.reshape(-1, 3)

# set up plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor('black')
q = ax.quiver(X, Y, U, V, color=colors, scale=25, width=0.004)
ax.imshow(pixels, extent=[0, 100, 0, 100])
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_aspect('equal')
ax.axis('off')

# animation 
def update(frame):
    angle = frame * np.pi / 120 # increase to increase speed
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    U_rot = cos_a * U - sin_a * V
    V_rot = sin_a * U + cos_a * V
    q.set_UVC(U_rot, V_rot)

ani = FuncAnimation(fig, update, frames=120, interval=50)
plt.show()

