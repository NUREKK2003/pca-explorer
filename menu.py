import streamlit as st

def menu():
    st.sidebar.page_link("pages/home_page.py",label="Home")
    st.sidebar.page_link("pages/data_analytics_page.py",label="Data analytics")

def analytics_top_bar_menu():
    col1, col2,col3,col4 = st.columns(4)
    data_preparation = col1.button('I. Data preparation')
    pca = col2.button('II. PCA')
    if data_preparation:
        st.switch_page('pages/data_analytics_page.py')
    if pca:
        if st.session_state.pca_unlocked == 1:
            pass
            #st.switch_page('pages/pca_page.py')
        else:
            st.status("please select the data first")