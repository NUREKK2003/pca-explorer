import streamlit as st

# initialize sessio state
st.session_state.pca_unlocked = 0
# redirect
st.switch_page("pages/home_page.py")
