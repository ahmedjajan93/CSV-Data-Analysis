from langchain_community.llms import HuggingFaceHub
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("google-flan")

# Streamlit UI setup
st.set_page_config(page_title="CSV Data Analysis", layout="wide")
st.title('Make some analysis on your data 🔎')
st.header('Please upload your CSV here:')

# File uploader
data = st.file_uploader('Upload Your File', type='csv')

# Model selection
model_choice = st.radio("Choose Model", 
                        ["FLAN-T5 (Fast)", "Mistral-7B (Accurate)", "StarCoder (Code)"])

# Cached function to load CSV
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

# Cached function to load the model
@st.cache_resource
def load_model(choice):
    repo_ids = {
        "FLAN-T5 (Fast)": "google/flan-t5-large",
        "Mistral-7B (Accurate)": "mistralai/Mistral-7B-v0.1",
        "StarCoder (Code)": "bigcode/starcoder"
    }
    
    model_kwargs = {
        "FLAN-T5 (Fast)": {"temperature": 0.7, "max_length": 512},
        "Mistral-7B (Accurate)": {"temperature": 0.1, "max_new_tokens": 100, "do_sample": False},
        "StarCoder (Code)": {"temperature": 0.7, "max_length": 512}
    }
    
    return HuggingFaceHub(
        repo_id=repo_ids[choice],
        huggingfacehub_api_token=api_key,
        model_kwargs=model_kwargs[choice]
    )

# Input for natural language query
query = st.text_area('Enter your query')
button = st.button('Generate Response')

# Show preview if data uploaded
if data:
    df = load_csv(data)
    st.subheader("Data Preview")
    st.dataframe(df.head(3), use_container_width=True)

# When the button is clicked
if button:
    if data and query:
        with st.spinner("🧠 Thinking..."):
            try:
                llm = load_model(model_choice)
                result = llm.predict(query)

                st.subheader("Analysis Result")
                st.write(result)

                with st.expander("See generated code"):
                    st.code(f"# Query: {query}\n# Response:\n{result}")

            except Exception as e:
                st.write("The server is busy. Please try again later.")
                
        st.warning("Please upload a CSV file and enter a query.")
