import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# --------- Função de Sombreamento Phong ----------
def sombreamento_phong_superficie(X, Y, Z, pos_luz, pos_observador, cor_luz, ka, kd, ks, brilho):
    """
    Aplica o modelo de sombreamento de Phong a uma superfície 3D definida pelas coordenadas X, Y e Z.

    Este método calcula a cor de cada ponto da superfície utilizando os componentes ambiente, difuso e especular
    do modelo de iluminação de Phong. A iluminação é feita considerando uma fonte de luz pontual e uma superfície polida.

    Parâmetros:
    ----------
    X, Y, Z : np.ndarray
        Matrizes 2D representando as coordenadas dos pontos da superfície em 3D.
    pos_luz : np.ndarray
        Vetor (x, y, z) com a posição da fonte de luz.
    pos_observador : np.ndarray
        Vetor (x, y, z) com a posição do observador (câmera).
    cor_luz : np.ndarray
        Cor da luz como vetor RGB com valores entre 0 e 1. (Ex: [1.0, 1.0, 0.3] para luz amarela)
    ka : float
        Coeficiente de refletância ambiente da superfície.
    kd : float
        Coeficiente de refletância difusa da superfície.
    ks : float
        Coeficiente de refletância especular da superfície.
    brilho : float
        Expoente de brilho (shininess) usado na reflexão especular. Quanto maior, mais focado é o brilho.

    Retorno:
    -------
    np.ndarray
        Matriz 3D (n, m, 3) contendo os valores de cor RGB calculados para cada ponto da superfície.
    """
    n, m = X.shape
    rgb = np.zeros((n, m, 3))

    def calcula_normal(i, j):
        if i < n-1:
            dXdi = np.array([X[i+1,j]-X[i,j], Y[i+1,j]-Y[i,j], Z[i+1,j]-Z[i,j]])
        else:
            dXdi = np.array([X[i,j]-X[i-1,j], Y[i,j]-Y[i-1,j], Z[i,j]-Z[i-1,j]])
        if j < m-1:
            dXdj = np.array([X[i,j+1]-X[i,j], Y[i,j+1]-Y[i,j], Z[i,j+1]-Z[i,j]])
        else:
            dXdj = np.array([X[i,j]-X[i,j-1], Y[i,j]-Y[i,j-1], Z[i,j]-Z[i,j-1]])
        normal = np.cross(dXdi, dXdj)
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        return normal

    for i in range(n):
        for j in range(m):
            posicao = np.array([X[i, j], Y[i, j], Z[i, j]])
            normal = calcula_normal(i, j)
            para_luz = pos_luz - posicao
            para_luz = para_luz / np.linalg.norm(para_luz)
            para_observador = pos_observador - posicao
            para_observador = para_observador / np.linalg.norm(para_observador)
            ambiente = ka * cor_luz
            difusa = kd * cor_luz * max(np.dot(normal, para_luz), 0)
            reflexao = 2 * np.dot(normal, para_luz) * normal - para_luz
            angulo_especular = max(np.dot(reflexao, para_observador), 0)
            especular = ks * cor_luz * (angulo_especular ** brilho)
            cor = ambiente + difusa + especular
            rgb[i, j, :] = np.clip(cor, 0, 1)
    return rgb

# --------- Restante do código ----------
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

# Desenhar arestas (exceto as da face curva)
for (i, j) in arestas:
    if (i, j) in arestas_remover or (j, i) in arestas_remover:
        continue
    x = [vertices[i][0], vertices[j][0]]
    y = [vertices[i][1], vertices[j][1]]
    z = [vertices[i][2], vertices[j][2]]
    ax.plot(x, y, z, 'k', linewidth=2)

# Face curva (lateral 1: v0-v1-v6-v5)
v0 = np.array(vertices[0])
v1 = np.array(vertices[1])
v6 = np.array(vertices[6])
v5 = np.array(vertices[5])

n = 40
borda_baixo = np.linspace(v0, v1, n)
borda_cima  = np.linspace(v5, v6, n)

X = []
Y = []
Z = []
curvatura = 1.2

for i in range(n):
    linha = np.linspace(borda_baixo[i], borda_cima[i], n)
    t = np.linspace(0, 1, n)
    curva = curvatura * (t - 0.5)**2
    normal = np.cross(v1-v0, v5-v0)
    normal = normal / np.linalg.norm(normal)
    deslocamento = curvatura * 0.25 # ajuste do deslocamento da curva para encaixar na face
    linha_curva = linha - curva[:, None] * normal + deslocamento * normal
    X.append(linha_curva[:, 0])
    Y.append(linha_curva[:, 1])
    Z.append(linha_curva[:, 2])

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

# --------- Parâmetros da luz/material (Phong) ---------
pos_luz = np.array([3, 3, 5])
cor_luz = np.array([1.0, 1.0, 0.3])  # Amarelado
ka = 0.15
kd = 0.6
ks = 0.8
brilho = 30
pos_observador = np.array([0, 0, 8])

# --------- Cores com Phong ---------
cores = sombreamento_phong_superficie(X, Y, Z, pos_luz, pos_observador, cor_luz, ka, kd, ks, brilho)
ax.plot_surface(X, Y, Z, facecolors=cores, shade=False, alpha=0.95, edgecolor='none')

# --------- Arestas da face curva ---------
ax.plot(X[0], Y[0], Z[0], 'k', linewidth=2)
ax.plot(X[-1], Y[-1], Z[-1], 'k', linewidth=2)
ax.plot(X[:,0], Y[:,0], Z[:,0], 'k', linewidth=2)
ax.plot(X[:,-1], Y[:,-1], Z[:,-1], 'k', linewidth=2)

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
ax.set_title("Prisma Pentagonal com Uma Face Lateral Curva (Phong)")
plt.show()
