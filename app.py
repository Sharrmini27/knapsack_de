import streamlit as st
import numpy as np  # Added this to fix the NameError in your screenshot
import data_loader
import de_optimizer
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="DE Knapsack Solver", layout="wide")
st.title("📦 Multi-Objective Knapsack: Differential Evolution")

st.sidebar.header("Settings")
inst_id = st.sidebar.selectbox("Select Instance ID", [0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
pop_size = st.sidebar.slider("Population Size", 10, 100, 50)
gens = st.sidebar.slider("Generations", 10, 200, 100)

if st.button("🚀 Run Optimization"):
    data = data_loader.load_instance("mknapcb3.txt", inst_id)
    
    if data:
        history, final_pop = de_optimizer.run_de(data, pop_size, gens, 0.5, 0.7)
        
        st.subheader("Convergence Curve")
        st.line_chart(history)
        
        st.subheader("Pareto Front Exploration")
        profits = [np.sum((p > 0.5).astype(int) * data['values']) for p in final_pop]
        w2_usage = [np.sum((p > 0.5).astype(int) * data['w2']) for p in final_pop]
        df = pd.DataFrame({"Profit": profits, "W2_Usage": w2_usage})
        fig = px.scatter(df, x="Profit", y="W2_Usage", title="Trade-off: Max Profit vs Min W2")
        st.plotly_chart(fig)
