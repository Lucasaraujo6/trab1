import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


state = st.session_state
MINX = MINY = -1
MAX_X = MAX_Y = 5

if "df" not in state:
     df = pd.DataFrame({
     'X': [*[0]*(9)],
     'Y': [*[0]*(9)],
     'Sinal': [*['>=']*(9)],
     'Valor': [*[0]*(9)]
     })
     # Criando o DataFrame com as restrições
     df = pd.DataFrame({
          'X': [ 1, -5, 3, 0, 0, 0],  # Coeficientes de x
          'Y': [ 2, 5, 5, 0, 0, 0],   # Coeficientes de y
          'Sinal': ['>=', '<=', '>=', '>=', '>=', '>='],  
          'Valor': ['1','-10', '15', '0', '0', '0'] # Valores das restrições
     })
     state.df = df

def set_const():
     qnt_constraints = state.set_const # Pegamos o valor atualizado

     current_qnt_lines = len(state.df)
     
     if current_qnt_lines > qnt_constraints:
          state.df = state.tempdf.iloc[:qnt_constraints]
     else:
          df = state.tempdf.copy()
          difference = qnt_constraints - current_qnt_lines
          
          new_rows = pd.DataFrame({
               'X': [0] * difference,
               'Y': [0] * difference,
               'Sinal': ['>='] * difference,
               'Valor': [0] * difference
          })

          df = pd.concat([df, new_rows], ignore_index=True)
          state.df = df


# Exibição do DataFrame editável
fo = st.columns([5,2,2,2,2,2])
fo[0].selectbox('Função Objetivo:',['Maximizar', 'Minimizar'])
fo[1].number_input(key='x',label='x',label_visibility='hidden')
fo[2].text_input('X', label_visibility='hidden',value='X')
fo[3].selectbox('F:',['+', '-'],label_visibility='hidden')
fo[4].number_input(key='y',label='y',label_visibility='hidden')
fo[5].text_input('Y', label_visibility='hidden',value='Y')
st.subheader("Declare aqui sua função objetivo e suas variáveis")
st.number_input(
     label="Quantidade de restrições:",
     step=1, value=6, min_value=1,
     key="set_const", on_change=set_const
)
state.tempdf = st.data_editor(state.df, hide_index=True)

def intersection(a1, b1, c1, a2, b2, c2, xlim=(-2, 10), ylim=(-2, 10)):
     A = np.array([[a1, b1], [a2, b2]])
     B = np.array([c1, c2])
     try:
          ponto = np.linalg.solve(A, B)
          ponto = np.where(np.abs(ponto) < 1e-10, 0, ponto)  # Substituir valores próximos de 0 por 0
          
          # Verificar se o ponto está dentro dos limites
          if xlim[0] <= ponto[0] <= xlim[1] and ylim[0] <= ponto[1] <= ylim[1]:
               return ponto
          else:
               return None  # Ignorar pontos fora do intervalo
     except np.linalg.LinAlgError:
          return None
# Configurar o Streamlit
st.title("Resolução Gráfica de Problemas Lineares")

# Criar o gráfico
fig, ax = plt.subplots(figsize=(8, 6))

# Criar uma grade de valores
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, y)

# Plotar as restrições
df = state.tempdf.copy()
new_rows = pd.DataFrame({
     'X': [1, 0, 1, 0],
     'Y': [0, 1, 0, 1],
     'Sinal': ['>=','>=','<=','<='],
     'Valor': [0, 0, MAX_X, MAX_Y]
})

df = pd.concat([df, new_rows], ignore_index=True)
print('df',df)
cores = ['b', 'g', 'r', 'c', 'm']
# Converter as colunas para tipos numéricos
df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")
df["X"] = pd.to_numeric(df["X"], errors="coerce")
df["Y"] = pd.to_numeric(df["Y"], errors="coerce")

for i, row in df.iloc[:].iterrows():    
     if row["Y"] == 0:  # Caso especial: Y = 0 (linha vertical)
          if row["X"] != 0:  # Evitar divisão por zero
               x_constante = row["Valor"] / row["X"]
               ax.axvline(x=x_constante, color=cores[i % len(cores)], linestyle="--", 
                         label=f"{row['X']}x ≥ {row['Valor']}" if row["Sinal"] == ">=" else f"{row['X']}x ≤ {row['Valor']}")
     else:
          # Caso geral: Y ≠ 0
          df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")
          df["X"] = pd.to_numeric(df["X"], errors="coerce")
          df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
          if row["Sinal"] == ">=":
               ax.plot(x, (row["Valor"] - row["X"] * x) / row["Y"], cores[i % len(cores)], 
                         label=f"{row['X']}x + {row['Y']}y ≥ {row['Valor']}")
          elif row["Sinal"] == "<=":
               ax.plot(x, (row["Valor"] - row["X"] * x) / row["Y"], cores[i % len(cores)], 
                         label=f"{row['X']}x + {row['Y']}y ≤ {row['Valor']}")
          elif row["Sinal"] == "==":
               ax.plot(x, (row["Valor"] - row["X"] * x) / row["Y"], cores[i % len(cores)], 
                         label=f"{row['X']}x + {row['Y']}y = {row['Valor']}")
               
# # Encontrar pontos de interseção para determinar os pontos extremos
intersection_points = []
for i in range(len(df)):
     for j in range(i + 1, len(df)):
          point = intersection(df.iloc[i]["X"], df.iloc[i]["Y"], df.iloc[i]["Valor"], 
                              df.iloc[j]["X"], df.iloc[j]["Y"], df.iloc[j]["Valor"])
          if point is not None:
               intersection_points.append(point)
               
               
extreme_points = []
# print(intersection_points)
for point in intersection_points:
     viavel = True  # Inicializamos a viabilidade como verdadeira para cada ponto
     for _, row in df.iterrows():
          # Verificamos se o ponto satisfaz cada restrição
          if row["Sinal"] == ">=":
               if not (row["X"] * point[0] + row["Y"] * point[1] >= row["Valor"] - 1e-4):
                    viavel = False
                    break
          elif row["Sinal"] == "<=":
               if not (row["X"] * point[0] + row["Y"] * point[1] <= row["Valor"] + 1e-4):
                    viavel = False
                    break
          elif row["Sinal"] == "==":
               if not (row["X"] * point[0] + row["Y"] * point[1] == row["Valor"]):
                    viavel = False
                    break
     
     # Apenas adicionamos o ponto se ele for viável
     if viavel:
          extreme_points.append(point)

     # Exibir os pontos extremos encontrados
     print("Pontos extremos:", extreme_points)
     
def get_point(point):
     # Se a parte decimal é insignificante, retornar como inteiro
     if abs(point - round(point)) < 1e-5:
          return int(point)

     # Arredondar para duas casas decimais inicialmente
     rounded = round(point, 2)

     # Verificar se é uma dízima periódica ou um número significativo
     first_decimal = int(abs(rounded * 10) % 10)  # Primeira casa decimal
     second_decimal = int(abs(rounded * 100) % 10)  # Segunda casa decimal

     # Se apenas a primeira casa decimal é significativa, retornar uma casa
     if second_decimal == 0:
          return round(point, 1)

     # Caso contrário, retornar com duas casas decimais
     return rounded

# # Plotar pontos extremos
for point in extreme_points:
     if point[0] == MAX_X or point[1] == MAX_Y:
          continue
     ax.scatter(*point, color="black", zorder=3)
     ax.text(point[0], point[1], f"({get_point(point[0])}, {get_point(point[1])})", fontsize=9)
# Adicione as retas associadas para ajudar a fechar a região
# for i, row in df.iterrows():
#      if row["Y"] != 0:  # Evitar divisões por zero
#           y_values = (row["Valor"] - row["X"] * x) / row["Y"]
#           ax.plot(x, y_values, linestyle="--", color="gray", label=f"Reta {i}")

# Quando há apenas dois pontos, trace a direção de escape
if len(extreme_points) == 2:
     p1, p2 = extreme_points[0], extreme_points[1]
     
     # Plotar a linha que conecta os dois pontos
     ax.plot(
          [p1[0], p2[0]],
          [p1[1], p2[1]],
          color="blue", linestyle="-", label="Região entre os pontos"
     )

     # Estender a linha para direções infinitas (dependendo da função-objetivo)
     if p1[0] < p2[0]:
          ax.arrow(p2[0], p2[1], 1, (p2[1] - p1[1]) / (p2[0] - p1[0]), color="red", label="Extensão infinita")
     else:
          ax.arrow(p1[0], p1[1], -1, (p2[1] - p1[1]) / (p2[0] - p1[0]), color="red", label="Extensão infinita")
          
from scipy.spatial import ConvexHull
import numpy as np

# Garantir que os pontos extremos estão formatados como um array numpy
extreme_points = np.array(extreme_points)
print(extreme_points)
# Calcular a ordem correta dos pontos usando ConvexHull
if len(extreme_points) >= 3:  # Pelo menos 3 pontos são necessários para formar um polígono
     hull = ConvexHull(extreme_points)
     vertices = hull.vertices  # Índices dos pontos que formam a envoltória convexa

     # Obter os vértices ordenados na sequência correta
     polygon_x = [extreme_points[i][0] for i in vertices]
     polygon_y = [extreme_points[i][1] for i in vertices]

     # Preencher a região viável no gráfico
     ax.fill(polygon_x, polygon_y, color="lightblue", alpha=0.5, label="Região Viável")
# Traçar vetor gradiente
# ax.quiver(0, 0, 1, 2, color="orange", scale=5, label="Vetor Gradiente")

# Traçar curvas de nível arbitrárias
# Z = X + 2 * Y
# ax.contour(X, Y, Z, levels=[5, 10, 15], colors=["purple", "gray"], linestyles=["dashed", "dotted"])

# Filtrar pontos viáveis
# pontos_viaveis = []
# for ponto in pontos_extremos:
#      viavel = True
#      for i, row in df.iterrows():
#           if row["Sinal"] == ">=":
#                viavel &= (row["X"] * ponto[0] + row["Y"] * ponto[1] >= row["Valor"])
#           elif row["Sinal"] == "<=":
#                viavel &= (row["X"] * ponto[0] + row["Y"] * ponto[1] <= row["Valor"])
#           elif row["Sinal"] == "==":
#                viavel &= (row["X"] * ponto[0] + row["Y"] * ponto[1] == row["Valor"])
#      if viavel:
#           pontos_viaveis.append(ponto)

# # Calcular a envoltória convexa
# if len(pontos_viaveis) >= 3:  # ConvexHull requer pelo menos 3 pontos
#      hull = ConvexHull(pontos_viaveis)
#      for simplex in hull.simplices:
#           plt.plot(
#                [pontos_viaveis[simplex[0]][0], pontos_viaveis[simplex[1]][0]],
#                [pontos_viaveis[simplex[0]][1], pontos_viaveis[simplex[1]][1]],
#                'k-'
#           )

#      # Preencher a região viável
#      plt.fill(
#           [pontos_viaveis[i][0] for i in hull.vertices],
#           [pontos_viaveis[i][1] for i in hull.vertices],
#           color="lightblue", alpha=0.5, label="Região Viável"
#      )
# Configuração do gráfico
ax.set_xlim(MINX, MAX_X)
ax.set_ylim(MINY, MAX_Y)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
ax.grid(True)
st.pyplot(fig)