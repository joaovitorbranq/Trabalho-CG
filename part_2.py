import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import colorsys
import time

# ======== DEFINIÇÃO DO OBJETO PRISMA PENTAGONAL ========
vertices = [
    (2, 0, 0),     # v1
    (1, 2, 0),     # v2
    (-1, 2, 0),    # v3
    (-2, 0, 0),    # v4
    (0, -2, 0),    # v5
    (2, 0, 1),     # v6
    (1, 2, 1),     # v7
    (-1, 2, 1),    # v8
    (-2, 0, 1),    # v9
    (0, -2, 1)     # v10
]

faces = [
    [0, 1, 2, 3, 4],             # base
    [5, 6, 7, 8, 9],             # topo
    [0, 1, 6, 5],                # lateral 1
    [1, 2, 7, 6],                # lateral 2
    [2, 3, 8, 7],                # lateral 3
    [3, 4, 9, 8],                # lateral 4
    [4, 0, 5, 9]                 # lateral 5
]

# ======== GERAR CORES HSV E CONVERTER PARA RGB ========
qtd_faces = len(faces)
hsv_colors = []
for i in range(qtd_faces):
    h = (360 * i) / (qtd_faces + 1)
    h_norm = h / 360.0  # normalizar para o intervalo [0, 1]
    rgb = colorsys.hsv_to_rgb(h_norm, 1.0, 1.0)  # S = 1, V = 1
    hsv_colors.append(rgb) # convertemos para rgb
print(hsv_colors)

# ======== PROJEÇÃO ORTOGRÁFICA XY E PLOTAGEM ========
vertices_2d = [(x, y) for (x, y, z) in vertices] # ignora o eixo z

fig, ax = plt.subplots()
patches = []
for face in faces:
    polygon = Polygon([vertices_2d[idx] for idx in face], closed=True)
    patches.append(polygon)
collection = PatchCollection(patches, edgecolor='black')
ax.add_collection(collection)

ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title("Projeção Ortográfica 2D com Cores HSV")

# ======== ADICIONAR EIXOS X E Y VISUAIS ========
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')  # Eixo X
ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')  # Eixo Y
ax.text(2.7, 0.2, 'X', fontsize=12, color='gray')
ax.text(0.2, 2.7, 'Y', fontsize=12, color='gray')


# ======== LOOP PARA TROCA CIRCULAR DAS CORES ========
while True:
    collection.set_facecolor(hsv_colors)
    plt.draw()
    plt.pause(2)  # pausa entre as trocas
    hsv_colors = hsv_colors[1:] + [hsv_colors[0]]  # rotação circular das cores
