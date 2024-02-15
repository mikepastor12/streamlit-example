
#   ! pip install PyPDF2


import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()

    #  pip install InstructorEmbedding
    #  pip install sentence-transformers==2.2.2
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    #  from InstructorEmbedding import INSTRUCTOR
    # model = INSTRUCTOR('hkunlp/instructor-xl')
    # sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
    # instruction = "Represent the Science title:"
    # embeddings = model.encode([[instruction, sentence]])

    # embeddings = model.encode(text_chunks)
    print('have Embeddings:   ')

    # text_chunks="this is a test"
    #   FAISS,  Chroma and other vector databases
    #
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print('FAISS succeeds:   ')

    return vectorstore


st.set_page_config(page_title="MLP Chat with multiple PDFs",
               page_icon=":books:")

st.write('Hello world2', unsafe_allow_html=True)

st.header("Mike's PDF Chat :books:")

user_question = st.text_input("Ask a question about your documents:")

print( 'Hello World' )











# """
# # Welcome to Streamlit!

# Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
# If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
# forums](https://discuss.streamlit.io).

# In the meantime, below is an example of what you can do with just a few lines of code:
# """

# num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
# num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

# indices = np.linspace(0, 1, num_points)
# theta = 2 * np.pi * num_turns * indices
# radius = indices

# x = radius * np.cos(theta)
# y = radius * np.sin(theta)

# df = pd.DataFrame({
#     "x": x,
#     "y": y,
#     "idx": indices,
#     "rand": np.random.randn(num_points),
# })

# st.altair_chart(alt.Chart(df, height=700, width=700)
#     .mark_point(filled=True)
#     .encode(
#         x=alt.X("x", axis=None),
#         y=alt.Y("y", axis=None),
#         color=alt.Color("idx", legend=None, scale=alt.Scale()),
#         size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
#     ))
