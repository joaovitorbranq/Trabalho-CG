import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

vertices = [
    (2, 0, 0),     # v1 - 0
    (1, 2, 0),     # v2 - 1
    (-1, 2, 0),    # v3 - 2
    (-2, 0, 0),    # v4 - 3
    (0, -2, 0),    # v5 - 4
    (2, 0, 1),     # v6 - 5
    (1, 2, 1),     # v7 - 6
    (-1, 2, 1),    # v8 - 7
    (-2, 0, 1),    # v9 - 8
    (0, -2, 1)     # v10 - 9
]

arestas = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),       # base
    (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),       # topo
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)        # laterais
]

arestas_remover = [(0, 1), (1, 6), (6, 5), (5, 0)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Desenhar arestas (wireframe) - pula as da face curva
for (i, j) in arestas:
    if (i, j) in arestas_remover or (j, i) in arestas_remover:
        continue
    x = [vertices[i][0], vertices[j][0]]
    y = [vertices[i][1], vertices[j][1]]
    z = [vertices[i][2], vertices[j][2]]
    ax.plot(x, y, z, 'k')

# Face curva (lateral 1)
v0 = np.array(vertices[0])
v1 = np.array(vertices[1])
v6 = np.array(vertices[6])
v5 = np.array(vertices[5])

n = 40

baixo = np.linspace(v0, v1, n)
cima  = np.linspace(v5, v6, n)

X = []
Y = []
Z = []
curvatura = 1.2

for i in range(n):
    linha = np.linspace(baixo[i], cima[i], n)
    t = np.linspace(0, 1, n)
    curva = curvatura * (t - 0.5)**2
    normal = np.cross(v1-v0, v5-v0)
    normal = normal / np.linalg.norm(normal)
    offset = curvatura * 0.25  # offset para alinhar a superfície nos vértices!
    linha_curva = linha - curva[:, None] * normal + offset * normal
    X.append(linha_curva[:, 0])
    Y.append(linha_curva[:, 1])
    Z.append(linha_curva[:, 2])

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

brilho = np.clip(0.4 + 0.6 * (Z - np.min(Z)) / (np.max(Z) - np.min(Z)), 0, 1)
ax.plot_surface(X, Y, Z, facecolors=plt.cm.YlOrBr(brilho), shade=True, alpha=0.8, edgecolor='none')

ax.view_init(elev=30, azim=45)

comprimento = 3
ax.quiver(0, 0, 0, comprimento, 0, 0, color='r', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, comprimento, 0, color='g', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, comprimento, color='b', arrow_length_ratio=0.1)
ax.text(comprimento, 0, 0, 'X', color='r')
ax.text(0, comprimento, 0, 'Y', color='g')
ax.text(0, 0, comprimento, 'Z', color='b')

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-1, 2)
ax.set_box_aspect([1, 1, 1])
ax.set_title("Prisma Pentagonal com Uma Face Lateral Curva (Encaixe Perfeito)")
plt.show()
