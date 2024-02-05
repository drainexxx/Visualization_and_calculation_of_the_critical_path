import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

########## Функция поиска критического пути

__all__ = [
    "floyd_warshall",
    "floyd_warshall_predecessor_and_distance_max",
    "reconstruct_path",
    "floyd_warshall_numpy_r",
]


@nx._dispatch
def floyd_warshall_numpy_r(G, nodelist=None, weight="weight"):

    if nodelist is not None:
        if not (len(nodelist) == len(G) == len(set(nodelist))):
            raise nx.NetworkXError(
                "nodelist must contain every node in G with no repeats."
                "If you wanted a subgraph of G use G.subgraph(nodelist)"
            )

    # Чтобы обработать случаи, когда ребро имеет вес =0, мы должны убедиться, что
    # nonedges также не присваивается значение 0.
    A = nx.to_numpy_array(
        G, nodelist, multigraph_weight=max, weight=weight, nonedge=np.inf
    )
    n, m = A.shape
    np.fill_diagonal(A, 0)  # диагональные элементы должны быть равны нулю
    for i in range(n):
        # Второй член имеет ту же форму, что и A из-за трансляции
        A = np.maximum(A, A[i, :][np.newaxis, :] + A[:, i][:, np.newaxis])
    return A


@nx._dispatch
def floyd_warshall_predecessor_and_distance_max(G, weight="weight"):

    from collections import defaultdict

    # представление словаря из словарей для dist и pred
    # используйте здесь какую-нибудь магию defaultdict
    # для dist по умолчанию используется значение inf с плавающей запятой
    dist = defaultdict(lambda: defaultdict(lambda: float("inf")))
    for u in G:
        dist[u][u] = 0
    pred = defaultdict(dict)
    # инициализировать словарь расстояний пути в качестве матрицы смежности
    # также установите расстояние до self равным 0 (нулевая диагональ)
    undirected = not G.is_directed()
    for u, v, d in G.edges(data=True):
        e_weight = d.get(weight, 1.0)
        dist[u][v] = min(e_weight, dist[u][v])
        pred[u][v] = u
        if undirected:
            dist[v][u] = max(e_weight, dist[v][u])
            pred[v][u] = v
    for w in G:
        dist_w = dist[w]  # сохранить повторное вычисление
        for u in G:
            dist_u = dist[u]  # сохранить повторное вычисление
            for v in G:
                d = dist_u[w] + dist_w[v]
                if dist_u[v] > d:
                    dist_u[v] = d
                    pred[u][v] = pred[w][v]
    return dict(pred), dict(dist)


def reconstruct_path(source, target, predecessors):

    if source == target:
        return []
    prev = predecessors[source]
    curr = prev[target]
    path = [target, curr]
    while curr != source:
        curr = prev[curr]
        path.append(curr)
    return list(reversed(path))


@nx._dispatch
def floyd_warshall_r(G, weight="weight"):
    return floyd_warshall_predecessor_and_distance_max(G, weight=weight)[1]
###################



G = nx.DiGraph() # Создаю граф

# Создаю массив из exel файла
with open('vertexes.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    rows_1 = list(csv_reader)

data = [] # Делаю массив из всех данных exel файла
for i in rows_1:
    data.append(i)

print(f"Все данные, связанные с вершинами: {data}")

with open('communications.csv') as csv_file:
    csv_repeat = csv.reader(csv_file, delimiter=';')
    rows_2 = list(csv_repeat)

edge_list = []  # Делаю список связей
for i in rows_2:
    comun = (i[0], i[1])
    edge_list.append(comun)

print(f"Все связи между вершинами: {edge_list}")
# Добавить ребра по списку
G.add_edges_from(edge_list)

# Создаю список вершин, они же ключи
vertex_list = []
for i in data:
    connection = i[0]
    if i != '':
        vertex_list.append(connection)

print(f"Имена всех вершин: {vertex_list}")

# добавим вершины

G.add_nodes_from(vertex_list)

# Указываю положение каждой вершины
node_positions = {}

# Формирую значения для ключей(координаты вершин)
values = []
for i in data:
    summir = (int(i[1]), int(i[2])) #координаты
    values.append(summir)

node_positions = {vertex_list[i]: values[i] for i in range(0, len(vertex_list), 1)} # Словарь вершина и ее координаты
print(f"Словарь (Вершина: координата): {node_positions}")

# Формирую массив для критического пути
vesa = []
for i in rows_2:
    a_sk = i[2]
    a_sk = -int(a_sk)
    ves_prom = (i[0], i[1], a_sk)
    vesa.append(ves_prom)
print(f"Связи и их веса: {vesa}")
# Нарисовать график DAG
plt.title('Модель')  # название картинки
plt.xlim(-22, 22)  # Установить диапазон координат оси X
plt.ylim(-40, 40)  # Установить диапазон координат оси Y


# Функция для обработки событий мыши и обновления позиций вершин
def on_move(event):
    if event.inaxes is not None:
        for node in G.nodes:
            if event.inaxes == plt.gca() and node in node_positions:
                node_positions[node] = (event.xdata, event.ydata)
                break
        update_graph()

# Функция для обновления позиций вершин и перерисовки графа
def update_graph():
    plt.clf()
    nx.draw(G, pos=node_positions, with_labels=True)
    plt.draw()

# Привязка обработчика событий мыши
#plt.gcf().canvas.mpl_connect('key_press_event', on_move)

###############################

G.add_weighted_edges_from(vesa)

# расчет кратчайших путей для ВСЕХ пар вершин
predecessors, _ = floyd_warshall_predecessor_and_distance_max(G)
print(predecessors)
# кратчайший путь от вершины [s] к вершине [v]
shortest_path_s_v = nx.reconstruct_path(data[0][0], data[len(data)-1][0], predecessors)
# список ребер кратчайшего пути
edges = [(a,b) for a,b in zip(shortest_path_s_v, shortest_path_s_v[1:])]
# список всех весов ребер
weights = nx.get_edge_attributes(G, 'weight')
# рисуем кратчайший путь: [s] -> [v]
nx.draw_networkx_edges(G, node_positions, edgelist=edges, edge_color="r", width=3)
# заголовок графика
title = "Критический путь из [{}] в [{}]: {}"\
        .format(data[0][0], data[len(data)-1][0], " -> ".join(shortest_path_s_v))
plt.title(title)

#### Добавляем позднее завершение, поздний старт, продолжительность, ранний старт, ранее завершение и id

for i in  data:
    G.add_node(i[0], id=f"{i[7]}|    |{i[8]}\n |{i[0]}|{i[3]}\n{i[5]}|    |{i[6]}\n\n\n\n")

################################################

################################################
# Отображение графа
# Отображение графа с заданными свойствами узлов
node_labels = nx.get_node_attributes(G, 'id')

options = {
    'node_color': 'red',     # color of node
    'node_size': 200,          # size of node
    'width': 1,                 # line width of edges
    'arrowstyle': '-|>',        # array style for directed graph
    'arrowsize': 18,            # size of arrow
    'edge_color':'black',        # edge color
}

nx.draw_networkx(G, node_positions, with_labels=True, labels=node_labels, **options, font_size=9, font_color='black', arrows=True)
# Отображение информации о предыдущих и последующих узлах

# Отображение графа
plt.axis('off')
plt.show()


