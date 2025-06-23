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
    [4, 3, 2, 1, 0],             # base invertida para corrigir a normal
    [5, 6, 7, 8, 9],             # topo
    [0, 1, 6, 5],                # lateral 1
    [1, 2, 7, 6],                # lateral 2
    [2, 3, 8, 7],                # lateral 3
    [3, 4, 9, 8],                # lateral 4
    [4, 0, 5, 9]                 # lateral 5
]


# ======== CONFIGURAÇÃO DA LUZ ========
# luz vindo da frente (direção Z positiva)
direcao_luz = np.array([0, 1, 1])
direcao_luz = direcao_luz / np.linalg.norm(direcao_luz) # normaliza




# ======== CÁLCULO DA NORMAL DE UMA FACE ========
def face_normal(face):
    """
    Calcula a normal de uma face.
    """

    # define os vértices da face
    v0 = np.array(vertices[face[0]])
    v1 = np.array(vertices[face[1]])
    v2 = np.array(vertices[face[2]])

    # calcula os vetores de aresta
    vetor1 = v1 - v0
    vetor2 = v2 - v0

    # calcula a normal
    normal = np.cross(vetor1, vetor2) # produto vetorial
    normal = normal / np.linalg.norm(normal) # normaliza
    return normal


# ======== CÁLCULO DA ILUMINAÇÃO (Luz ambiente + direta) ========
def compute_color(normal_face):
    """
    Calcula a cor de uma face.
    Recebe a normal da face como parâmetro
    """
    hue = 200 / 360  # tom azul
    luz_ambiente = 0.3 # luz ambiente de 30%
    cos_theta = np.dot(normal_face, direcao_luz) # cosseno do angulo entre a normal e a luz
    luz_direta = max(cos_theta, 0)

    intensidade = luz_ambiente + 0.8 * luz_direta
    intensidade = min(intensidade, 1)  # garantir limite máximo de 100%

    saturacao = luz_ambiente + 0.8 * luz_direta
    saturacao = min(saturacao, 1)  # garantir limite máximo de 100%

    # Hue constante, saturação e intensidade variam com a iluminação
    r, g, b = colorsys.hsv_to_rgb(hue, saturacao, intensidade)
    return (r, g, b)


# ======== CÁLCULO DA PROFUNDIDADE PARA O ALGORITMO DO PINTOR ========
def profundidade_face(face):
    """
    Calcula uma média de profundidade para cada face.
    Prioriza o eixo Z (profundidade), que representa a distância do observador.
    """
    zs = [vertices[i][2] for i in face]
    ys = [vertices[i][1] for i in face]
    xs = [vertices[i][0] for i in face]
    return np.mean(zs) + np.mean(ys)


# cria uma lista ordenada de índices das faces, 
# ordenada da face mais distante para a mais próxima (algoritmo do pintor)
face_order = sorted(
    range(len(faces)),
    key=lambda idx: profundidade_face(faces[idx]),
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
ax.set_title('Prisma Pentagonal com Iluminação')

plt.show()
