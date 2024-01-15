import streamlit as st
import base64
import os

st.set_page_config(
    page_title="PDF Chat BoT",
    page_icon="🤖",
)


css = """
<style>
    .stApp {
        background: Black;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .header {
        background-color: black; /* White background for the header */
        color: white; /* Black text for the header */
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
        padding: 10px;
        border-radius: 10px;
    }
    .sub-header, .content {
        color: white; /* White text for sub-header and content */
        text-align: center;
        font-size: 1.2em;
    }
    .content {
        padding: 20px;
        font-size: 1.1em;
    }
</style>
"""


st.markdown(css, unsafe_allow_html=True)

# Introduction image for Debate-AI
st.image('img_1.png', caption='PDF Chat BoT', use_column_width='always')


# Custom header for the page
st.markdown("<div class='header'>Welcome to the PDF Chat BoT</div>", unsafe_allow_html=True)
