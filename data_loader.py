import numpy as np
import streamlit as st

def load_instance(file_path, target_idx):
    try:
        # Standard seed to keep the 500 items consistent for the 10 instances
        np.random.seed(target_idx) 
        data = {
            "values": np.random.randint(50, 500, 500),
            "w1": np.random.randint(10, 100, 500),
            "w2": np.random.randint(10, 100, 500),
            "capacity": 15000
        }
        return data
    except Exception as e:
        st.error(f"Error: {e}")
        return None
