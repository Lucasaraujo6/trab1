import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Variável global
GLOBAL_VAR = "Streamlit é incrível!"

# Container
with st.container():
     st.title("Exemplos de Funcionalidades do Streamlit")
     st.write("Explore as principais funcionalidades do Streamlit aqui!")
     st.write(f"Variável global: {GLOBAL_VAR}")

# Session State
if "contador" not in st.session_state:
     st.session_state.contador = 0

if st.button("Incrementar contador"):
     st.session_state.contador += 1
st.write(f"Valor do contador no `session_state`: {st.session_state.contador}")

# DataFrame
df = pd.DataFrame({
     "Categoria": ["A", "B", "C"],
     "Valor": [10, 20, 30]
})
st.subheader("Exemplo de DataFrame")
st.write(df)

# Plot de imagem (gráfico)
st.subheader("Gráfico gerado com matplotlib")
fig, ax = plt.subplots()
ax.bar(df["Categoria"], df["Valor"])
ax.set_title("Exemplo de Gráfico de Barras")
ax.set_xlabel("Categoria")
ax.set_ylabel("Valor")
st.pyplot(fig)

# Upload de imagem
uploaded_image = st.file_uploader("Faça upload de uma imagem", type=["png", "jpg", "jpeg"])
if uploaded_image is not None:
     st.subheader("Imagem carregada")
     st.image(uploaded_image, caption="Imagem carregada pelo usuário")