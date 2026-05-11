import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

from langchain_core.output_parsers import StrOutputParser


# -----------------------------------
# Streamlit UI
# -----------------------------------

st.set_page_config(page_title="ATS Resume Analyzer")

st.title("ATS Resume Analyzer + ATS Optimizer")


# -----------------------------------
# API Key (Streamlit Secrets - SECURE)
# -----------------------------------

api_key = st.secrets["GEMINI_API_KEY"]


# -----------------------------------
# Upload PDF
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)


# -----------------------------------
# Main Logic
# -----------------------------------

if uploaded_file:

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    st.success("PDF Loaded Successfully")

    # Split Documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )

    texts = text_splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key
    )

    # System Prompt
    system_prompt = """
    You are an expert ATS Resume Analyzer and Career Assistant.

    Your tasks:
    - Analyze resumes professionally
    - Identify technical skills
    - Evaluate ATS compatibility
    - Suggest improvements
    - Match resumes with job roles

    Use only the provided context.

    If information is not found in the resume,
    say:
    'Information not found in the resume.'
    """

    # User Question
    question = st.text_input("Ask Your Question")

    if question:

        # Retrieval
        response = vector_store.similarity_search(
            query=question,
            k=3
        )

        # Context
        context = "\n".join([doc.page_content for doc in response])

        # Prompt Template
        prompt_template = ChatPromptTemplate.from_messages([

            SystemMessagePromptTemplate.from_template(system_prompt),

            HumanMessagePromptTemplate.from_template(
                """
                Context:
                {context}

                Question:
                {question}
                """
            )
        ])

        # Chain
        chain = prompt_template | llm | StrOutputParser()

        # Response
        result = chain.invoke({
            "context": context,
            "question": question
        })

        st.subheader("Generated Response")
        st.write(result)
