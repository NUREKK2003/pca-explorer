import streamlit as st
import pandas as pd
import numpy as np
from pages.footer import footer
from menu import menu


st.title('PCA Explorer')
st.text('Just a simple streamlit app for fast and easy PCA analyse with option to export the results.')
menu()