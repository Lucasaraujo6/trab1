import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


state = st.session_state
MINX = MINY = -0.5
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
          'Valor': [1,-10, 15, 0, 0, 0] # Valores das restrições
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

st.subheader("Declare aqui sua função objetivo e suas variáveis")

# Exibição do DataFrame editável
fo = st.columns([5,2,2,2,2,2,2])
fo[0].selectbox('Função Objetivo:',['Maximizar', 'Minimizar'], key='opt')
fo[1].selectbox('F:',['+', '-'],label_visibility='hidden', key='x_signal', disabled=True)
fo[2].number_input(key='x',label='x',label_visibility='hidden', min_value=0)
fo[3].text_input('X', label_visibility='hidden',value='X', disabled=True)
fo[4].selectbox('F:',['+', '-'],label_visibility='hidden', key='y_signal', disabled=True)
fo[5].number_input(key='y',label='y',label_visibility='hidden', min_value=0)
fo[6].text_input('Y', label_visibility='hidden',value='Y', disabled=True )
function = f"{state.x_signal} {state.x} x " if state.x else ''
function += f"{state.y_signal} {state.y} y" if state.y else ''
st.write(f"**Função Objetivo:** {state.opt} {function}")
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

st.title("Resolução Gráfica de Problemas Lineares")

# Criar o gráfico
fig, ax = plt.subplots(figsize=(8, 6))
ax.quiver(0, 0, state.x, state.y, color="orange", scale=5, label=f"Gradiente da FO: {function}")

# Criar uma grade de valores
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)

# Plotar as restrições
df = state.tempdf.copy()

cores = ['b', 'g', 'r', 'c', 'm']

for i, row in df.iloc[:len(df)].iterrows():    
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
     if df.iloc[i]["Y"] != 0:  # Evitar divisão por zero
          MAX_Y = max(df.iloc[i]["Valor"] / df.iloc[i]["Y"] + 2, MAX_Y)
     if df.iloc[i]["X"] != 0:  # Evitar divisão por zero
          MAX_X = max(df.iloc[i]["Valor"] / df.iloc[i]["X"] + 2, MAX_X)
     for j in range(i + 1, len(df)):
          point = intersection(df.iloc[i]["X"], df.iloc[i]["Y"], df.iloc[i]["Valor"], 
                              df.iloc[j]["X"], df.iloc[j]["Y"], df.iloc[j]["Valor"])
          if point is not None:
               intersection_points.append(point)

new_rows = pd.DataFrame({
     'X': [1, 0, 1, 0],
     'Y': [0, 1, 0, 1],
     'Sinal': ['>=','>=','<=','<='],
     'Valor': [0, 0, MAX_X, MAX_Y]
})

df = pd.concat([df, new_rows], ignore_index=True)
for i in range(len(df)-4,len(df)):
     
     for j in range(len(df)):
          point = intersection(df.iloc[i]["X"], df.iloc[i]["Y"], df.iloc[i]["Valor"], 
                              df.iloc[j]["X"], df.iloc[j]["Y"], df.iloc[j]["Valor"])
          if point is not None:
               intersection_points.append(point)
               
extreme_points = []
for point in intersection_points:
     feasible = True  # Inicializamos a viabilidade como verdadeira para cada ponto
     for _, row in df.iterrows():
          # Verificamos se o ponto satisfaz cada restrição
          if row["Sinal"] == ">=":
               if not (row["X"] * point[0] + row["Y"] * point[1] >= row["Valor"] - 1e-4):
                    feasible = False
                    break
          elif row["Sinal"] == "<=":
               if not (row["X"] * point[0] + row["Y"] * point[1] <= row["Valor"] + 1e-4):
                    feasible = False
                    break
          elif row["Sinal"] == "==":
               if not (row["X"] * point[0] + row["Y"] * point[1] == row["Valor"]):
                    feasible = False
                    break
     
     # Apenas adicionamos o ponto se ele for viável
     if feasible:
          extreme_points.append(point)
     
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
     
# Garantir que os pontos extremos estão formatados como um array numpy
extreme_points = np.array(extreme_points)
# Calcular a ordem correta dos pontos usando ConvexHull
if len(extreme_points) >= 3:  # Pelo menos 3 pontos são necessários para formar um polígono
     hull = ConvexHull(extreme_points)
     vertices = hull.vertices  # Índices dos pontos que formam a envoltória convexa

     # Obter os vértices ordenados na sequência correta
     polygon_x = [extreme_points[i][0] for i in vertices]
     polygon_y = [extreme_points[i][1] for i in vertices]

     # Preencher a região viável no gráfico
     ax.fill(polygon_x, polygon_y, color="lightblue", alpha=0.5, label="Região Viável")
optimal_point = None
optimal_value = float('-inf') if state.opt == 'Maximizar' else float('inf')  # Inicialização adaptada

for vertex in extreme_points:
     objective_value = int(f'{state.x_signal}{state.x}') * vertex[0] + int(f'{state.y_signal}{state.y}') * vertex[1]
     
     # Condição para Maximizar ou Minimizar
     if (state.opt == 'Maximizar' and objective_value > optimal_value) or (state.opt == 'Minimizar' and objective_value < optimal_value):
          optimal_value = objective_value
          optimal_point = vertex

if optimal_point is not None:
     # Direção perpendicular ao vetor gradiente
     perpendicular_direction = np.array([-state.y, state.x])  # Adaptado para ser perpendicular
     t_values = np.linspace(-10, 10, 100)
     line_x = optimal_point[0] + t_values * perpendicular_direction[0]
     line_y = optimal_point[1] + t_values * perpendicular_direction[1]
     
    
     # Tornar o ponto maior e mais destacado
     ax.scatter(*optimal_point, color="red", edgecolor="black", zorder=3, label=f"Ponto {'Máximo' if state.opt == 'Maximizar' else 'Mínimo'}")
     ax.plot(line_x, line_y, color="gray", linestyle="dashed", label = 'Perpendicular do grad no ótimo')
          
ax.set_xlim(MINX, MAX_X)
ax.set_ylim(MINY, MAX_Y)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
ax.grid(True)
st.pyplot(fig)