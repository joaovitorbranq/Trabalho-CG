import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# ======== PRISMA BASE ========
vertices = [
    (2, 0, 0), (1, 2, 0), (-1, 2, 0), (-2, 0, 0), (0, -2, 0),
    (2, 0, 1), (1, 2, 1), (-1, 2, 1), (-2, 0, 1), (0, -2, 1)
]

faces_planas = [
    [4, 3, 2, 1, 0],  # base inferior
    [0, 1, 6, 5], [1, 2, 7, 6], [2, 3, 8, 7],
    [3, 4, 9, 8], [4, 0, 5, 9]  # laterais
]

# ======== ILUMINAÇÃO (Phong) ========
light_dir = np.array([0, 0, 1])
view_dir = np.array([0, 0, 1])
light_color = np.array([1.0, 1.0, 0.5])  # luz amarelada
ka, kd, ks = 0.2, 0.5, 0.3
shininess = 32

def phong(normal):
    normal = normal / np.linalg.norm(normal)
    l = light_dir / np.linalg.norm(light_dir)
    v = view_dir / np.linalg.norm(view_dir)
    r = 2 * np.dot(normal, l) * normal - l
    ambient = ka * light_color
    diffuse = kd * max(np.dot(normal, l), 0) * light_color
    specular = ks * max(np.dot(v, r), 0) ** shininess * light_color
    return np.clip(ambient + diffuse + specular, 0, 1)

# ======== SUPERFÍCIE CURVA SUAVIZADA NO TOPO ========
def gerar_domo_pentagonal(pontos, centro_elevado, niveis=5):
    """
    pontos: lista dos 5 vértices da face superior
    centro_elevado: ponto central elevado no eixo Z
    niveis: quantidade de anéis entre borda e centro
    """
    superf_vertices = []
    superf_faces = []

    for i in range(niveis + 1):
        t = i / niveis
        anel = [(1 - t) * p + t * centro_elevado for p in pontos]
        superf_vertices.append(anel)

    # Conectar anéis com faces quadradas (2 triângulos por quadrado)
    for i in range(niveis):
        atual = superf_vertices[i]
        proximo = superf_vertices[i + 1]
        for j in range(len(pontos)):
            a = atual[j]
            b = atual[(j + 1) % 5]
            c = proximo[(j + 1) % 5]
            d = proximo[j]
            superf_faces.append([a, b, c, d])

    # Achatar lista de vértices e gerar índices
    flat_vertices = []
    index_map = {}
    index = 0
    for anel in superf_vertices:
        for v in anel:
            key = tuple(np.round(v, 8))
            if key not in index_map:
                index_map[key] = index
                flat_vertices.append(v)
                index += 1

    face_indices = []
    for face in superf_faces:
        indices = [index_map[tuple(np.round(p, 8))] for p in face]
        face_indices.append(indices)

    return flat_vertices, face_indices

# Pegar vértices do topo
topo_ids = [5, 6, 7, 8, 9]
pontos_topo = [np.array(vertices[i]) for i in topo_ids]
centro = np.mean(pontos_topo, axis=0)
centro[2] += 1.2  # elevação real para curvatura perceptível

malha_vertices, malha_faces = gerar_domo_pentagonal(pontos_topo, centro, niveis=10)

# ======== PLOTAGEM ========
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Desenhar faces planas do prisma
for face in faces_planas:
    pts = [vertices[i] for i in face]
    poly = Poly3DCollection([pts], facecolors='lightblue', edgecolors='black', alpha=0.6)
    ax.add_collection3d(poly)

# Desenhar superfície suavemente curva
for face in malha_faces:
    pts = [malha_vertices[i] for i in face]
    v1 = pts[1] - pts[0]
    v2 = pts[3] - pts[0]
    normal = np.cross(v1, v2)
    if np.linalg.norm(normal) == 0:
        continue
    color = phong(normal)
    patch = Poly3DCollection([pts], facecolors=[color], edgecolors='gray')
    ax.add_collection3d(patch)

# Eixos
ax.quiver(0, 0, 0, 3, 0, 0, color='r')
ax.quiver(0, 0, 0, 0, 3, 0, color='g')
ax.quiver(0, 0, 0, 0, 0, 3, color='b')
ax.text(3, 0, 0, 'X', color='r')
ax.text(0, 3, 0, 'Y', color='g')
ax.text(0, 0, 3, 'Z', color='b')

# Visual
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-1, 3.5)
ax.set_box_aspect([1, 1, 1])
ax.set_title("Parte 4: Topo Curvo Suavizado com Sombreamento Phong")

plt.show()
