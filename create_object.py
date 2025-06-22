import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

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
# total de 10 vertices

arestas = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),       # base
    (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),       # topo
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)        # laterais
]
# total de 15 arestas

faces = [
    [0, 1, 2, 3, 4],             # base com 5 pontos (pentágono)
    [5, 6, 7, 8, 9],             # topo com 5 pontos (pentágono)
    [0, 1, 6, 5],                # lateral 1
    [1, 2, 7, 6],                # lateral 2
    [2, 3, 8, 7],                # lateral 3
    [3, 4, 9, 8],                # lateral 4
    [4, 0, 5, 9]                 # lateral 5
]
# total de 7 faces

# ======== VERIFICAÇÃO DE EULER ========
V = len(vertices) # 10
E = len(arestas)  # 15
F = len(faces)    # 7

if (V - E + F) == 2:
    # 10 - 15 + 7 = 2
    print("Satisfaz fórmula de Euler")
else:
    print("Não satisfaz fórmula de Euler")

print("Vértices (V):", V)
print("Arestas (E):", E)
print("Faces (F):", F)

# ======== PLOTAGEM DO WIREFARME ========
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Desenhar arestas
for (i, j) in arestas:
    x = [vertices[i][0], vertices[j][0]]
    y = [vertices[i][1], vertices[j][1]]
    z = [vertices[i][2], vertices[j][2]]
    ax.plot(x, y, z, 'k')

# Adicionar eixos X, Y, Z
comprimento = 3
ax.quiver(0, 0, 0, comprimento, 0, 0, color='r', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, comprimento, 0, color='g', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, comprimento, color='b', arrow_length_ratio=0.1)

# Rótulos dos eixos
ax.text(comprimento, 0, 0, 'X', color='r')
ax.text(0, comprimento, 0, 'Y', color='g')
ax.text(0, 0, comprimento, 'Z', color='b')

# ====== Ajustar os limites dos eixos ======
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-1, 2)

# Ajustes finais
ax.set_title("Wireframe do Prisma Pentagonal")
ax.set_box_aspect([1, 1, 1])
plt.show()