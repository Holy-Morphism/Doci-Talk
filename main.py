from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
import streamlit as st
from groq import Groq
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()


@st.cache_resource
def load_data(temp_file):
    with st.spinner(text="Loading and Indexing"):
        Settings.llm = OpenAI(model="gpt-4o-mini")
        docs = SimpleDirectoryReader(input_dir='./' + temp_file, recursive=True).load_data()
        index = VectorStoreIndex.from_documents(docs)
        return index


client = Groq()
 
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("Doci-Talk with Whisper")

uploaded_files = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    st.success("Files uploaded successfully!")
    
    index = load_data(UPLOAD_DIR)

    audio_value = st.audio_input("Ask a question")

    if audio_value:
        st.audio(audio_value)

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_value.name)[1]) as tmp_file:
            tmp_file.write(audio_value.getvalue())
            tmp_file_path = tmp_file.name
        
        with open(tmp_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(tmp_file_path, file.read()),
                model="whisper-large-v3-turbo",
                prompt="Specify context or spelling",
                response_format="json",
                language="en",
                temperature=0.0
            )

        
        st.write("Transcription:")
        st.write(transcription.text)
        query_engine = index.as_query_engine()
        response = query_engine.query(transcription.text)
        st.write("Response:")
        st.write(response.response)


