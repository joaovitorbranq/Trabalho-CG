import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import colorsys

# ======== DEFINIÇÃO DO OBJETO (Prisma Pentagonal) ========
vertices = [
    (2, 0, 0),     # v0
    (1, 2, 0),     # v1
    (-1, 2, 0),    # v2
    (-2, 0, 0),    # v3
    (0, -2, 0),    # v4
    (2, 0, 1),     # v5
    (1, 2, 1),     # v6
    (-1, 2, 1),    # v7
    (-2, 0, 1),    # v8
    (0, -2, 1)     # v9
]

faces = [
    [4, 3, 2, 1, 0],             # Base invertida para corrigir a normal
    [5, 6, 7, 8, 9],             # Topo
    [0, 1, 6, 5],                # lateral 1
    [1, 2, 7, 6],                # lateral 2
    [2, 3, 8, 7],                # lateral 3
    [3, 4, 9, 8],                # lateral 4
    [4, 0, 5, 9]                 # lateral 5
]


# ======== CONFIGURAÇÃO DA LUZ ========
# Luz vindo da frente (direção Z positiva)
light_dir = np.array([0, 0, 1])
light_dir = light_dir / np.linalg.norm(light_dir)

# ======== DEFINIR UMA ÚNICA COR BASE (HUE CONSTANTE) ========
H = 200 / 360  # tom azul
S_max = 1
V_max = 1


# ======== CÁLCULO DA NORMAL DE UMA FACE ========
def face_normal(face):
    v0 = np.array(vertices[face[0]])
    v1 = np.array(vertices[face[1]])
    v2 = np.array(vertices[face[2]])
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)
    return normal


# ======== CÁLCULO DA ILUMINAÇÃO (Luz ambiente + difusa) ========
def compute_color(normal):
    ambient = 0.3
    ndotl = np.dot(normal, light_dir)
    diffuse = max(ndotl, 0)

    intensity = ambient + 0.8 * diffuse
    intensity = min(intensity, 1)  # garantir limite máximo

    # Saturação constante, só o brilho varia
    r, g, b = colorsys.hsv_to_rgb(H, S_max, V_max * intensity)
    return (r, g, b)


# ======== CÁLCULO DA PROFUNDIDADE PARA O ALGORITMO DO PINTOR ========
def face_depth(face):
    zs = [vertices[i][2] for i in face]
    ys = [vertices[i][1] for i in face]
    xs = [vertices[i][0] for i in face]
    return np.mean(zs) + np.mean(ys) * 0.5  # prioriza Z e um pouco de Y para melhor percepção


# ======== ORDENAR FACES (ALGORITMO DO PINTOR) ========
face_order = sorted(
    range(len(faces)),
    key=lambda idx: face_depth(faces[idx]),
    reverse=True  # desenha as mais distantes primeiro
)


# ======== CRIAR FIGURA E EIXO 3D ========
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ======== DESENHAR FACES COM CORES ========
poly3d = []
face_colors = []

for idx in face_order:
    face = faces[idx]
    pts = [vertices[i] for i in face]
    poly3d.append(pts)
    normal = face_normal(face)
    color = compute_color(normal)
    face_colors.append(color)

# Adiciona as faces ao plot
collection = Poly3DCollection(poly3d, facecolors=face_colors, edgecolors='black', linewidths=1)
ax.add_collection3d(collection)

# ======== DESENHAR EIXOS X, Y, Z ========
comprimento = 3
ax.quiver(0, 0, 0, comprimento, 0, 0, color='r', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, comprimento, 0, color='g', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, comprimento, color='b', arrow_length_ratio=0.1)

ax.text(comprimento, 0, 0, 'X', color='r')
ax.text(0, comprimento, 0, 'Y', color='g')
ax.text(0, 0, comprimento, 'Z', color='b')

# ======== CONFIGURAÇÕES DOS LIMITES ========
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-1, 2)

# Aspecto e título
ax.set_box_aspect([1, 1, 1])
ax.set_title('Prisma Pentagonal com Iluminação (Algoritmo do Pintor)')

plt.show()
