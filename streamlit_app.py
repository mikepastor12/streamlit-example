
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

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,    #  1000
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

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

#   ###############################################
def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    #  llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    #  google/bigbird-roberta-base     facebook/bart-large
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain


def handle_userinput(user_question):

    response = st.session_state.conversation({'question': user_question})
    # response = st.session_state.conversation({'summarization': user_question})
    st.session_state.chat_history = response['chat_history']
    
    
    # st.empty()
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)




################################################################################

st.set_page_config(page_title="MLP Chat with multiple PDFs",
               page_icon=":books:")

st.write('Hello world2', unsafe_allow_html=True)

st.header("Mike's PDF Chat :books:")

user_question = st.text_input("Ask a question about your documents:")
if user_question:
    handle_userinput(user_question)

# st.write( user_template, unsafe_allow_html=True)
# st.write(user_template.replace( "{{MSG}}", "Hello robot!"), unsafe_allow_html=True)
# st.write(bot_template.replace( "{{MSG}}", "Hello human!"), unsafe_allow_html=True)


with st.sidebar:

    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

    # Upon button press
    if st.button("Process these files"):
        with st.spinner("Processing..."):

            #################################################################
            #  Track the overall time for file processing into Vectors
            # #
            from datetime import datetime
            global_now = datetime.now()
            global_current_time = global_now.strftime("%H:%M:%S")
            st.write("Vectorizing Files - Current Time =", global_current_time)

            # get pdf text
            raw_text = get_pdf_text(pdf_docs)
            #  st.write(raw_text)

            # # get the text chunks
            text_chunks = get_text_chunks(raw_text)
            # st.write(text_chunks)

            # # create vector store
            vectorstore = get_vectorstore(text_chunks)

            # # create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)

            # Mission Complete!
            global_later = datetime.now()
            st.write("Files Vectorized - Total EXECUTION Time =",
                     (global_later - global_now), global_later)

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
