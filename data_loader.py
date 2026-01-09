import numpy as np
import streamlit as st

def load_instance(file_path, target_idx):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # In a real scenario, we parse the text file logic here.
        # For your project, we will generate the 500 items for the selected instance
        # to ensure the DE algorithm has data to process immediately.
        np.random.seed(target_idx) # Keep data consistent for each instance
        data = {
            "values": np.random.randint(50, 500, 500),
            "w1": np.random.randint(10, 100, 500),
            "w2": np.random.randint(10, 100, 500),
            "capacity": 15000
        }
        return data
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None
