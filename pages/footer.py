import streamlit as st

def footer():
    footer="""<style>
    a:link , a:visited{
    color: blue;
    background-color: transparent;
    text-decoration: underline;
    }

    a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
    }

    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
    }
    </style>
    <div class="footer">
    <p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://www.heflin.dev/" target="_blank">Heflin Stephen Raj S</a></p>
    </div>
    """
    st.markdown(footer,unsafe_allow_html=True)