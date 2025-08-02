%matplotlib widget

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh


# Function to draw a single voxel (wireframe or filled)
def draw_wire_cube(x, y, z, size=1, color='black', alpha=0.5, linewidth=0.2, fill=False):
    r = [0, size]
    vertices = np.array([[x+i, y+j, z+k] for i in r for j in r for k in r])
    faces = [
        [vertices[i] for i in [0,1,3,2]],
        [vertices[i] for i in [4,5,7,6]],
        [vertices[i] for i in [0,1,5,4]],
        [vertices[i] for i in [2,3,7,6]],
        [vertices[i] for i in [0,2,6,4]],
        [vertices[i] for i in [1,3,7,5]],
    ]
    if fill:
        poly3d = Poly3DCollection(faces, facecolors=color, edgecolors='black', linewidths=0.2, alpha=alpha)
        ax.add_collection3d(poly3d)
    else:
        edges = [
            (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),
            (2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
        ]
        for i,j in edges:
            ax.plot([vertices[i][0], vertices[j][0]],
                    [vertices[i][1], vertices[j][1]],
                    [vertices[i][2], vertices[j][2]],
                    color=color, alpha=alpha, linewidth=linewidth)

# Create 3D figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the transparent 10x10x10 wireframe cube
for x in range(12):
    for y in range(12):
        for z in range(12):
            draw_wire_cube(x, y, z)

# === MANUALLY DEFINE ALL VOXELS ===
filled_voxels = [
    # --- Body ---
    (7, 7, 7, "#7C7C7C", 0.5), (7, 6, 7, "#D58925", 0.5), (7, 5, 7, "#D58925", 0.5),
    (7, 4, 7, "#7C7C7C", 0.5), (7, 3, 7, "#7C7C7C", 0.5), (7, 2, 7, "#7C7C7C", 0.5),

    (6, 7, 7, "#7C7C7C", 0.5), (6, 6, 7, "#D58925", 0.5), (6, 5, 7, "#D58925", 0.5),
    (6, 4, 7, "#7C7C7C", 0.5), (6, 3, 7, "#7C7C7C", 0.5), (6, 2, 7, "#7C7C7C", 0.5),

    (5, 7, 7, "#7C7C7C", 0.5), (5, 6, 7, "#7C7C7C", 0.5), (5, 5, 7, "#7C7C7C", 0.5),
    (5, 4, 7, "#7C7C7C", 0.5), (5, 3, 7, "#7C7C7C", 0.5), (5, 2, 7, "#7C7C7C", 0.5),

    (7, 7, 6, "#7C7C7C", 0.5), (7, 6, 6, "#D58925", 0.5), (7, 5, 6, "#D58925", 0.5),
    (7, 4, 6, "#7C7C7C", 0.5), (7, 3, 6, "#7C7C7C", 0.5), (7, 2, 6, "#7C7C7C", 0.5),

    (6, 7, 6, "#7C7C7C", 0.5), (6, 6, 6, "#D58925", 0.5), (6, 5, 6, "#D58925", 1.0),
    (6, 4, 6, "#7C7C7C", 0.5), (6, 3, 6, "#D58925", 0.5), (6, 2, 6, "#D58925", 0.5),

    (5, 7, 6, "#7C7C7C", 0.5), (5, 6, 6, "#7C7C7C", 0.5), (5, 5, 6, "#7C7C7C", 0.5),
    (5, 4, 6, "#7C7C7C", 0.5), (5, 3, 6, "#7C7C7C", 0.5), (5, 2, 6, "#7C7C7C", 0.5),

    # Head or extended feature
    (6, 8, 7, "#7C7C7C", 0.5),
    (6, 9, 7, "#7C7C7C", 0.5),
    (5, 9, 7, "#7C7C7C", 0.5),
    (7, 9, 7, "#7C7C7C", 0.5),
    (6, 10, 7, "#7C7C7C", 0.5),
    (5, 10, 7, "#7C7C7C", 0.5),
    (7, 10, 7, "#7C7C7C", 0.5),
    (6, 11, 7, "#7C7C7C", 0.5),
    (6, 12, 7, "#7C7C7C", 0.5),

    #ears

    (5, 9, 8, "#7C7C7C", 0.5),
    (7, 9, 8, "#7C7C7C", 0.5),

  
    # tail
    (6, 1, 7, "#7C7C7C", 0.5),
    (6, 0, 7, "#7C7C7C", 0.5),
    (6, -1, 8, "#7C7C7C", 0.5),
    (6, -2, 8, "#7C7C7C", 0.5),
    (6, -3, 8, "#7C7C7C", 0.5),
    (6, -4, 7, "#7C7C7C", 0.5),

    #Legs

    (5, 2, 5, "#7C7C7C", 0.5), 
    (7, 2, 5, "#7C7C7C", 0.5), 
    (5, 7, 5, "#7C7C7C", 0.5), 
    (7, 7, 5, "#7C7C7C", 0.5), 
]

# --- Centering logic ---
coords = np.array([(x, y, z) for x, y, z, _, _ in filled_voxels])
center = coords.mean(axis=0)
offset = np.array([5.0, 5.0, 5.0]) - center  # Shift to cube center

# --- Draw shifted filled voxels ---
for x, y, z, color, alpha in filled_voxels:
    draw_wire_cube(x + offset[0], y + offset[1], z + offset[2], color=color, alpha=alpha, fill=True)



# Set view and axis properties
ax.view_init(elev=20, azim=45)
ax.set_xlim([-1, 10])
ax.set_ylim([-4, 10])
ax.set_zlim([0, 10])
ax.set_box_aspect([1, 1, 1])
ax.axis('off')
plt.tight_layout()
plt.show()





def voxel_to_glb(filled_voxels, offset, filename="voxel_mouse.glb"):
    # Create a scene
    scene = trimesh.Scene()

    for x, y, z, color, alpha in filled_voxels:
        x += offset[0]
        y += offset[1]
        z += offset[2]

        # Create cube mesh
        cube = trimesh.creation.box(extents=(1, 1, 1))
        cube.apply_translation([x + 0.5, y + 0.5, z + 0.5])  # center the cube

        # Set color
        rgba = trimesh.visual.color.hex_to_rgba(color)
        rgba[-1] = int(alpha * 255)
        cube.visual.face_colors = rgba

        # Add to scene
        scene.add_geometry(cube)

    # Export to .glb
    scene.export(filename)


voxel_to_glb(filled_voxels, offset)
