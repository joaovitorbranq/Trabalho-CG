import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import colorsys

# ======== FUNÇÕES AUXILIARES ========

def face_normal(face, vertices):
    v0 = np.array(vertices[face[0]])
    v1 = np.array(vertices[face[1]])
    v2 = np.array(vertices[face[2]])
    vetor1 = v1 - v0
    vetor2 = v2 - v0
    normal = np.cross(vetor1, vetor2)
    return normal / np.linalg.norm(normal)

def compute_color(normal_face):
    hue = 200 / 360  # azul
    luz_ambiente = 0.3
    cos_theta = np.dot(normal_face, direcao_luz)
    luz_direta = max(cos_theta, 0)
    intensidade = min(luz_ambiente + 0.8 * luz_direta, 1)
    saturacao = min(luz_ambiente + 0.8 * luz_direta, 1)
    r, g, b = colorsys.hsv_to_rgb(hue, saturacao, intensidade)
    return (r, g, b)

def profundidade_face(face, vertices):
    zs = [vertices[i][2] for i in face]
    ys = [vertices[i][1] for i in face]
    return np.mean(zs) + np.mean(ys)

def transladar(vertices, dx, dy, dz):
    return [(x+dx, y+dy, z+dz) for (x, y, z) in vertices]

# ======== LUZ ========
direcao_luz = np.array([0, 0, 1])
direcao_luz = direcao_luz / np.linalg.norm(direcao_luz)

# ======== PRISMA BASE ========
vertices_base = [
    (2, 0, 0), (1, 2, 0), (-1, 2, 0), (-2, 0, 0), (0, -2, 0),
    (2, 0, 1), (1, 2, 1), (-1, 2, 1), (-2, 0, 1), (0, -2, 1)
]

faces_base = [
    [4, 3, 2, 1, 0],  # base
    [5, 6, 7, 8, 9],  # topo
    [0, 1, 6, 5],
    [1, 2, 7, 6],
    [2, 3, 8, 7],
    [3, 4, 9, 8],
    [4, 0, 5, 9]
]

# ======== CRIA OS DOIS PRISMAS ========
vertices1 = vertices_base
faces1 = faces_base

# Prisma 2 deslocado para o lado e à frente
vertices2 = transladar(vertices_base, dx=5, dy=2.5, dz=0.0)
faces2 = faces_base

# ======== COMBINANDO TUDO COM INDEXAÇÃO ========
vertices_total = vertices1 + vertices2
faces_total = []

# Faces do primeiro prisma
faces_total.extend([(face, vertices1) for face in faces1])

# Faces do segundo prisma (ajustando índices)
offset = len(vertices1)
faces_total.extend([([i + offset for i in face], vertices2) for face in faces2])

# ======== ORDENAR FACES PELO ALGORITMO DO PINTOR ========
faces_ordenadas = sorted(
    faces_total,
    key=lambda item: profundidade_face(item[0], vertices_total),
    reverse=True
)

# ======== PLOTAGEM ========
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

poly3d = []
face_colors = []

for face_indices, _ in faces_ordenadas:
    pts = [vertices_total[i] for i in face_indices]
    normal = face_normal(face_indices, vertices_total)
    color = compute_color(normal)
    poly3d.append(pts)
    face_colors.append(color)

collection = Poly3DCollection(poly3d, facecolors=face_colors, edgecolors='black', linewidths=1)
ax.add_collection3d(collection)

# Eixos
comprimento = 3
ax.quiver(0, 0, 0, comprimento, 0, 0, color='r', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, comprimento, 0, color='g', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, comprimento, color='b', arrow_length_ratio=0.1)
ax.text(comprimento, 0, 0, 'X', color='r')
ax.text(0, comprimento, 0, 'Y', color='g')
ax.text(0, 0, comprimento, 'Z', color='b')

# Limites da cena
ax.set_xlim(-3, 6)
ax.set_ylim(-3, 6)
ax.set_zlim(-1, 2)
ax.set_box_aspect([1, 1, 1])
ax.set_title('Dois Prismas com iluminação e algoritmo do pintor')

plt.show()
