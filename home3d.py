import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull


state = st.session_state
MIN_X = MIN_Y = MIN_Z = -0.5
MAX_X = MAX_Y = MAX_Z =5

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
               'Z': [0] * difference,
               'Sinal': ['>='] * difference,
               'Valor': [0] * difference
          })

          df = pd.concat([df, new_rows], ignore_index=True)
          state.df = df

def intersection(a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3):
     A = np.array([[a1, b1, c1],
                    [a2, b2, c2],
                    [a3, b3, c3]])
     B = np.array([d1, d2, d3])
     try:
          ponto = np.linalg.solve(A, B)
          ponto = np.where(np.abs(ponto) < 1e-10, 0, ponto)  # Zera valores muito próximos de 0
          
          # Verifica se está dentro do intervalo
          x, y, z = ponto
          if MIN_X <= x <= MAX_X and MIN_Y <= y <= MAX_Y and MIN_Z <= z <= MAX_Z:
               return ponto
          else:
               return None
     except np.linalg.LinAlgError:
          return None


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

if "df" not in state:
     df = pd.DataFrame({
          'X': [ 1, -5, 3],  # Coeficientes de x
          'Y': [ 2, 5, 5],   # Coeficientes de y
          'Z': [ 2, 5, 5],   # Coeficientes de y
          'Sinal': ['>=', '<=', '>='],  
          'Valor': [1,-10, 15] # Valores das restrições
     })
     state.df = df

st.subheader("Declare aqui sua função objetivo e suas variáveis")

# Exibição do DataFrame editável
fo = st.columns([5,2,2,2])
fo[0].selectbox('Objetivo:',['Minimizar', 'Maximizar'], key='opt')
fo[1].number_input(key='x',label='Multiplicador do x', min_value=0, value = 1)
fo[2].number_input(key='y',label='Multiplicador do y', min_value=0, value = 1)
fo[3].number_input(key='z',label='Multiplicador do z', min_value=0, value = 1)
function = f"{state.x} x " if state.x else ''
function += f"{'+' if function else ''} {state.y} y" if state.y else ''
function += f"{'+' if function else ''} {state.z} z" if state.z else ''
st.write(f"**Função Objetivo:** {state.opt} {function}")
st.number_input(
     label="Quantidade de restrições:",
     step=1, value=3, min_value=1,
     key="set_const", on_change=set_const
)

state.tempdf = st.data_editor(state.df, hide_index=True)

st.subheader("Resolução Gráfica do Problema Linear")

# Criar o gráfico

# Criar a figura e o eixo 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0, 0, 0, state.x*100, state.y*100, state.z*100, color="orange", label=f"Gradiente da FO: {function}")

df = state.tempdf.copy()

# Plotar as restrições
intersection_points = []
for i in range(len(df)):
     row_i = df.iloc[i]
     
     # Evitar divisão por zero
     if row_i["X"] != 0:
          MAX_X = max(row_i["Valor"] / row_i["X"] + 2, MAX_X)
     if row_i["Y"] != 0:
          MAX_Y = max(row_i["Valor"] / row_i["Y"] + 2, MAX_Y)
     if row_i["Z"] != 0:
          MAX_Z = max(row_i["Valor"] / row_i["Z"] + 2, MAX_Z)
     
     for j in range(i + 1, len(df)):
          for k in range(j + 1, len(df)):
               row_j = df.iloc[j]
               row_k = df.iloc[k]
               
               # Calcular interseção entre três planos
               point = intersection(
                    row_i["X"], row_i["Y"], row_i["Z"], row_i["Valor"],
                    row_j["X"], row_j["Y"], row_j["Z"], row_j["Valor"],
                    row_k["X"], row_k["Y"], row_k["Z"], row_k["Valor"]
               )
               if point is not None:
                    intersection_points.append(point)
               
# Criar uma grade de valores
x = np.linspace(-10, MAX_X, 200)
y = np.linspace(-10, MAX_Y, 200)
z = np.linspace(-10, MAX_Z, 200)


cores = ['b', 'g', 'r', 'c', 'm']

X_grid, Y_grid = np.meshgrid(x, y)

for i, row in df.iterrows():
     a, b, c, d = row["X"], row["Y"], row["Z"], row["Valor"]
     color = cores[i % len(cores)]

     # Caso especial: Z == 0 → plano vertical paralelo ao eixo Z
     if c == 0:
          continue  # (ou tratar separadamente se quiser superfícies infinitas no z)
     else:
          # z = (d - ax - by) / c
          Z_grid = (d - a * X_grid - b * Y_grid) / c

          # Escolher cor e label com base no sinal
          if row["Sinal"] == "==":
               label = f"{a}x + {b}y + {c}z = {d}"
          elif row["Sinal"] == "<=":
               label = f"{a}x + {b}y + {c}z ≤ {d}"
          elif row["Sinal"] == ">=":
               label = f"{a}x + {b}y + {c}z ≥ {d}"
          else:
               label = f"{a}x + {b}y + {c}z ? {d}"

          # ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.2, color=color, label=label)
               
# # Encontrar pontos de interseção para determinar os pontos extremos


new_rows = pd.DataFrame({
     'X': [1, 0, 0, 1, 0, 0],
     'Y': [0, 1, 0, 0, 1, 0],
     'Z': [0, 0, 1, 0, 0, 1],
     'Sinal': ['>=','>=','>=','<=','<=','<='],
     'Valor': [0, 0, 0, MAX_X, MAX_Y, MAX_Z]
})

df = pd.concat([df, new_rows], ignore_index=True)

# Garantir que há pelo menos 3 planos
for i in range(len(df)):
     for j in range(len(df)):
          for k in range(len(df)):
               row_i = df.iloc[i]
               row_j = df.iloc[j]
               row_k = df.iloc[k]

               point = intersection(
                    row_i["X"], row_i["Y"], row_i["Z"], row_i["Valor"],
                    row_j["X"], row_j["Y"], row_j["Z"], row_j["Valor"],
                    row_k["X"], row_k["Y"], row_k["Z"], row_k["Valor"]
               )

               if point is not None:
                    intersection_points.append(point)

print('df', df)
extreme_points = []
print('intersection_points', intersection_points)
for point in intersection_points:
     feasible = True  # Inicializamos a viabilidade como verdadeira para cada ponto

     for _, row in df.iterrows():
          valor_lhs = row["X"] * point[0] + row["Y"] * point[1] + row["Z"] * point[2]

          if row["Sinal"] == ">=":
               if not (valor_lhs >= row["Valor"] - 1e-4):
                    feasible = False
                    break
          elif row["Sinal"] == "<=":
               if not (valor_lhs <= row["Valor"] + 1e-4):
                    feasible = False
                    break
          elif row["Sinal"] == "==":
               if not (abs(valor_lhs - row["Valor"]) <= 1e-4):
                    feasible = False
                    break

     # Apenas adicionamos o ponto se ele for viável
     if feasible:
          extreme_points.append(point)
print('extreme_points', extreme_points)

# Plotar pontos extremos
for point in extreme_points:
     if np.isclose(point[0], MAX_X) or np.isclose(point[1], MAX_Y) or np.isclose(point[2], MAX_Z):
          continue
     ax.scatter(point[0], point[1], point[2], color="black", zorder=3)
     ax.text(point[0], point[1], point[2],
               f"({get_point(point[0])}, {get_point(point[1])}, {get_point(point[2])})",
               fontsize=9, zorder=3)

# Garantir que os pontos extremos estão formatados como um array numpy
extreme_points = np.array(extreme_points)
# Envoltória convexa em 3D
if len(extreme_points) >= 4:  # Pelo menos 4 pontos são necessários para formar um volume 3D
     hull = ConvexHull(extreme_points)
     for s in hull.simplices:
          s = np.append(s, s[0])  # Fechar o polígono
          ax.plot(extreme_points[s, 0], extreme_points[s, 1], extreme_points[s, 2],
                    color="lightblue", alpha=0.5)

optimal_point = None
optimal_value = float('-inf') if state.opt == 'Maximizar' else float('inf')

for vertex in extreme_points:
     objective_value = (
          int(f'{state.x}') * vertex[0] +
          int(f'{state.y}') * vertex[1] +
          int(f'{state.z}') * vertex[2]
     )

     if (state.opt == 'Maximizar' and objective_value > optimal_value) or \
          (state.opt == 'Minimizar' and objective_value < optimal_value):
          optimal_value = objective_value
          optimal_point = vertex

if optimal_point is not None:
     # Direção perpendicular ao gradiente (vetor normal à FO)
     gradient_vector = np.array([state.x, state.y, state.z])
     direction_vector = np.cross(gradient_vector, [1, 0, 0])  # vetor qualquer não paralelo
     if np.allclose(direction_vector, 0):
          direction_vector = np.cross(gradient_vector, [0, 1, 0])
     t_values = np.linspace(-10, 10, 100)
     line_x = optimal_point[0] + t_values * direction_vector[0]
     line_y = optimal_point[1] + t_values * direction_vector[1]
     line_z = optimal_point[2] + t_values * direction_vector[2]

     ax.scatter(optimal_point[0], optimal_point[1], optimal_point[2],
                    color="red", edgecolor="black", s=100, zorder=3,
                    label=f"Ponto {'Máximo' if state.opt == 'Maximizar' else 'Mínimo'}: {optimal_point.tolist()}")

     ax.plot(line_x, line_y, line_z, color="gray", linestyle="dashed", label="Direção ortogonal ao gradiente")

ax.set_xlim(MIN_X, MAX_X)
ax.set_ylim(MIN_Y, MAX_Y)
ax.set_zlim(MIN_Z, MAX_Z)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.invert_yaxis()
ax.invert_xaxis()

ax.legend()
ax.grid(True)
st.pyplot(fig)
