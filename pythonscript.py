import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import math as m
import networkx as nx
import streamlit as st
from scipy.stats import norm

def OptionsValE(n, S, K, r, v, T, PC):
    dt = T/n                    
    u = m.exp(v*m.sqrt(dt)) 
    d = 1/u                     
    p = (m.exp(r*dt)-d)/(u-d)   
    Pm = np.zeros((n+1, n+1))   
    Cm = np.zeros((n+1, n+1))
    tmp = np.zeros((2,n+1))
    for j in range(n+1):
        tmp[0,j] = S*m.pow(d,j)
        tmp[1,j] = S*m.pow(u,j)
    tot = np.unique(tmp)
    c = n
    for i in range(c+1):
        for j in range(c+1):
            Pm[i,j-c-1] = tot[(n-i)+j]
        c=c-1
    for j in range(n+1, 0, -1):
        for i in range(j):
            if (PC == "put"):                               
                if(j == n+1):
                    Cm[i,j-1] = max(K-Pm[i,j-1], 0)     
                else:
                    Cm[i,j-1] = m.exp(-.05*dt) * (p*Cm[i,j] + (1-p)*Cm[i+1,j]) 
            if (PC == "call"):                               
                if (j == n + 1):
                    Cm[i,j-1] = max(Pm[i,j-1]-K, 0)     
                else:
                    Cm[i,j-1] = m.exp(-.05*dt) * (p*Cm[i,j] + (1-p)*Cm[i+1,j])  
    return [Pm,Cm]


def OptionsValA(n, S, K, r, v, T, PC):
    dt = T / n
    u = m.exp(v * m.sqrt(dt))
    d = 1 / u
    p = (m.exp(r * dt) - d) / (u - d)
    Pm = np.zeros((n + 1, n + 1))
    Cm = np.zeros((n + 1, n + 1))
    tmp = np.zeros((2, n + 1))
    
    for j in range(n + 1):
        tmp[0, j] = S * m.pow(d, j)
        tmp[1, j] = S * m.pow(u, j)
    tot = np.unique(tmp)
    
    c = n
    for i in range(c + 1):
        for j in range(c + 1):
            Pm[i, j - c - 1] = tot[(n - i) + j]
        c = c - 1
    
    for j in range(n + 1, 0, -1):
        for i in range(j):
            if PC == "put":  # American call option
                if j == n + 1:
                    Cm[i, j - 1] = max(K - Pm[i, j - 1], 0)
                else:
                    Cm[i, j - 1] = m.exp(-r * dt) * (
                        p * Cm[i, j] + (1 - p) * Cm[i + 1, j]
                    )
                # Check for early exercise
                Cm[i, j - 1] = max(Cm[i, j - 1], K - Pm[i, j - 1])
            if PC == "call":  # American put option
                if j == n + 1:
                    Cm[i, j - 1] = max(Pm[i, j - 1] - K, 0)
                else:
                    Cm[i, j - 1] = m.exp(-r * dt) * (
                        p * Cm[i, j] + (1 - p) * Cm[i + 1, j]
                    )
                # Check for early exercise
                Cm[i, j - 1] = max(Cm[i, j - 1], Pm[i, j - 1] - K)
    
    return [Pm, Cm]



def OptionsValintPresicion(option_type, n, S, K, r, v, T, PC):
    if option_type == "Américain":
        Pm,Cm = OptionsValE(n, S, K, r, v, T, PC)
    if option_type == "Européen":
        Pm,Cm = OptionsValA(n, S, K, r, v, T, PC)
    return [Pm,Cm]

def OptionsValint(option_type, n, S, K, r, v, T, PC):
    if option_type == "Américain":
        Pm,Cm = OptionsValE(n, S, K, r, v, T, PC)
    if option_type == "Européen":
        Pm,Cm = OptionsValA(n, S, K, r, v, T, PC)
    Pmi=np.matrix(np.round(Pm, decimals=1))
    Cmi=np.matrix(np.round(Cm, decimals=1))
    return [Pmi,Cmi]



def binomial_grid(option_type, n, S, K, r, v, T, PC, graph_type):

    [Pmi,Cmi]=OptionsValint(option_type,n,S,K,r,v,T,PC)
    if graph_type == "Prix" :
        matrix=Pmi
    if graph_type == "Option" :
        matrix=Cmi
    G = nx.Graph()

    for i in range(0, n + 1):
        for j in range(1, i + 2):
            if i < n:
                G.add_edge((i, j), (i + 1, j))
                G.add_edge((i, j), (i + 1, j + 1))

    posG = {}  # Dictionary with nodes position
    for node in G.nodes():
        posG[node] = (node[0], n + 2 + node[0] - 2 * node[1])

    nx.draw(G, pos=posG, with_labels=False, node_size=700)  # Set with_labels to False

    # Add labels to the nodes
    labels = {node: matrix.T[node[0],node[1]-1] for node in G.nodes()}
    nx.draw_networkx_labels(G, posG, labels, font_size=8, font_color='white')

    plt.show()


def black_scholes(PC, S, K, r, v, T, t = 0, div=0):
    """
    Calcule le prix de l'option européenne ou américaine (call ou put) en utilisant le modèle Black-Scholes.
    
    option_type : str
        Type d'option, "call" pour une option d'achat, "put" pour une option de vente.
    S : float
        Prix du sous-jacent.
    K : float
        Prix d'exercice.
    r : float
        Taux d'intérêt sans risque (annuel).
    v : float
        Volatilité du sous-jacent (annuelle).
    T : float
        Durée jusqu'à l'expiration de l'option (en années).
    t : float
        Durée jusqu'à l'expiration de l'option (en années).
    div : float, optional
        Taux de dividende (annuel), par défaut à 0.

    Returns
    -------
    option_price_BS : float
        Prix de l'option.
    """
    d1 = (np.log(S / K) + (r - div + 0.5 * v**2) * (T - t)) / (v * np.sqrt(T - t))
    d2 = d1 - v * np.sqrt(T - t)
    
    if PC == "call":
        option_price_BS = S * np.exp(-div * (T - t)) * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
    elif PC == "put":
        option_price_BS = K * np.exp(-r * (T - t)) * norm.cdf(-d2) - S * np.exp(-div * (T - t)) * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'.")
    
    return option_price_BS



# Titre de l'application
st.title("Valorisation d'Options")

# Create a two-column layout
input_col, graph_col = st.columns(2)

# Input parameters on the left column
with input_col:
    option_type = st.selectbox("Type d'option", ["Américain", "Européen"])
    PC = st.selectbox("Type d'option", ["put", "call"])
    S = st.number_input("Prix du sous-jacent (S)", value=100)
    K = st.number_input("Strike (K)", value=100)
    r = st.number_input("Taux d'intérêt (r)", value=0.05)
    v = st.number_input("Volatilité (v)", value=0.3)
    T = st.number_input("Échéance de l'option (T)", value=20 / 36)
    n = st.number_input("Nombre de périodes (n)", min_value=1, step=1, value=4)

# Graph on the right column

with graph_col:
    if st.button("Calculer et afficher les Graphes"):
        fig = plt.figure(figsize=(6, 8))  # Adjust the figure size as needed
        
        # Create the "Prix" graph on the top
        plt.subplot(2, 1, 1)  # 2 rows, 1 column, the first subplot
        binomial_grid(option_type, n, S, K, r, v, T, PC, "Prix")
        plt.title("graph des prix de sous-jacent")

        # Create the "Option" graph on the bottom
        plt.subplot(2, 1, 2)  # 2 rows, 1 column, the second subplot
        binomial_grid(option_type, n, S, K, r, v, T, PC, "Option")
        plt.title("graph des prix de l'option")

        st.pyplot(fig)

    # Display the price of the option
    [_, Cm] = OptionsValintPresicion(option_type, n, S, K, r, v, T, PC)
    option_price = Cm[0, 0]
    option_price_BS = black_scholes(PC, S, K, r, v, T)
    st.write(f"Prix de l'option selon le modele binomiale : {option_price}")
    st.write(f"Prix de l'option selon le modele black and scholes : {option_price_BS}")
