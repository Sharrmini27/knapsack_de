import streamlit as st
import data_loader
import de_optimizer
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="DE Knapsack Solver", layout="wide")

st.title("📦 Multi-Objective Knapsack: Differential Evolution")
st.write("Project for Course: JIE42903 – Evolutionary Computing")

# Sidebar Configuration
st.sidebar.header("Algorithm Parameters")
inst_id = st.sidebar.selectbox("Select Instance ID", [0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
pop_size = st.sidebar.slider("Population Size", 10, 100, 50)
gens = st.sidebar.slider("Generations", 10, 200, 100)
f_param = st.sidebar.slider("Mutation Factor (F)", 0.1, 1.0, 0.5)
cr_param = st.sidebar.slider("Crossover Rate (CR)", 0.1, 1.0, 0.7)

if st.button("🚀 Run Differential Evolution"):
    # Load Data
    data = data_loader.load_instance("mknapcb3.txt", inst_id)
    
    if data:
        with st.spinner('Running Evolution...'):
            history, final_pop = de_optimizer.run_de(data, pop_size, gens, f_param, cr_param)
        
        st.success("Optimization Complete!")
        
        # Performance Analysis Visuals
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Convergence Curve")
            st.line_chart(history)
            st.caption("This chart shows how the algorithm profit improved over generations.")

        with col2:
            st.subheader("Objective Trade-off")
            # Create data for Pareto scatter plot
            profits = [np.sum((p > 0.5).astype(int) * data['values']) for p in final_pop]
            w2_usage = [np.sum((p > 0.5).astype(int) * data['w2']) for p in final_pop]
            df = pd.DataFrame({"Profit (Max)": profits, "W2 Usage (Min)": w2_usage})
            fig = px.scatter(df, x="Profit (Max)", y="W2 Usage (Min)", title="Pareto Front Exploration")
            st.plotly_chart(fig)
    else:
        st.error("Could not initialize data. Check your GitHub files.")
