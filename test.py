import streamlit as st
import pandas as pd

data = {
    "metal_type": ["Gold", "Silver"],
    "diamond_color": ["D", "F", "H"],  # Extra value
    "price": [5000, 3000, 7000]
}

# ✅ Convert dictionary to DataFrame
df = pd.DataFrame.from_dict(data, orient="index").transpose()




st.write("hello world",df)
