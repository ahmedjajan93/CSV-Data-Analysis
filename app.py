from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="CSV Data Analysis", layout="wide")
st.title('🔍 Advanced CSV Analysis Tool')

# File uploader
data = st.file_uploader('Upload your CSV file', type=['csv'])

# Updated model configuration
MODEL_CONFIG = {
    "Mistral-7B (Accurate)": {
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.1",
        "params": {
            "temperature": 0.1,
            "max_new_tokens": 256
        }
    },
    "FLAN-T5 (Fast)": {
        "repo_id": "google/flan-t5-large",
        "params": {
            "temperature": 0.3,
            "max_length": 512
        }
    }
    
}

model_choice = st.selectbox(
    "Select AI Model",
    options=list(MODEL_CONFIG.keys()),
    index=0  # Default to FLAN-T5
)

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_resource
def get_llm(model_name):
    config = MODEL_CONFIG[model_name]
    return HuggingFaceHub(
        repo_id=config["repo_id"],
        huggingfacehub_api_token=os.getenv("google-flan"),
        model_kwargs=config["params"]
    )

if data:
    df = load_data(data)
    st.success(f"Data loaded successfully! Shape: {df.shape}")

    with st.expander("Preview data"):
        st.dataframe(df.head())

    question = st.text_area(
        "Enter your question about the data:",
        placeholder="E.g.: What's the average value in column X?"
    )

    if st.button("Analyze", type="primary") and question:
        with st.spinner("Analyzing..."):
            try:
                llm = get_llm(model_choice)

                # Convert first 10 rows to CSV text for context
                sample_data = df.head(10).to_csv(index=False)

                prompt = PromptTemplate(
                    input_variables=["data", "question"],
                    template="""
                    You are a data expert. Use the CSV sample below to answer the user's question.

                    CSV Data:
                    {data}

                    Question: {question}

                    Answer concisely:
                    """
                                )

                chain = LLMChain(llm=llm, prompt=prompt)
                result = chain.run(data=sample_data, question=question)

                st.subheader("Answer")
                st.write(result)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.info("Try simplifying your question or using a different model.")
