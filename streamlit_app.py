"""
Streamlit app for MEDAI Medical Assistant.
Modules: Chatbot (retrieval QA), Image Classifier, and Report Summarizer.
"""

import os
import streamlit as st
from PIL import Image
from Chatbot.retrieval_qa import get_retrieval_chain
from Skin_Disease_Classifier.skin_disease_classifier import predict, save_and_resize_image,load_model
from Report_Summarizer.report_summarizer import summarize_report

# Streamlit configuration
st.set_page_config(page_title="MEDAI Medical Assistant", layout="wide")
st.title("🩺 MEDAI Medical Assistant")

# --- Caching for chatbot chain ---
@st.cache_resource
def get_cached_qa_chain():
    return get_retrieval_chain()

# --- Caching for image classifier 
@st.cache_resource
def get_cached_predictor():
    return load_model()

# --- Caching for report summarizer ---
@st.cache_resource
def get_cached_summarizer():
    return summarize_report

# Tabs for each module
tab1, tab2, tab3 = st.tabs([
    "💬 Chatbot",
    "🖼️ Image Classifier",
    "📄 Report Summarizer"
])

# --- Tab 1: Chatbot ---
with tab1:
    st.subheader("Skin Disease Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = get_cached_qa_chain()

    user_input = st.text_input("Ask your medical question here:")

    if user_input:
        try:
            result = st.session_state.qa_chain.invoke({"question": user_input})
            response = result.get("answer", result)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response))
        except Exception as e:
            response = f"Error: Something went wrong. ({e})"
            st.session_state.chat_history.append(("Bot", response))

    for sender, message in st.session_state.chat_history:
        st.write(f"**{sender}:** {message}")

# --- Tab 2: Image Classifier ---
with tab2:
    st.subheader("Upload an image for skin disease classification")
    uploaded_file = st.file_uploader("Choose an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        with st.spinner("Classifying..."):
            try:
                image_path = save_and_resize_image(uploaded_file)
                df = predict(image_path)
                if df is not None and not df.empty:
                    st.success("Prediction Results:")
                    st.dataframe(df)
                else:
                    st.warning("No disease detected or classification failed.")
            except Exception as e:
                st.error(f"Failed to classify the image: {e}")

# --- Tab 3: Report Summarizer ---
with tab3:
    st.subheader("Summarize a Medical Report (PDF or Image)")
    report_file = st.file_uploader("Upload a report (PDF, JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"], key="report")
    if report_file is not None:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, report_file.name)
        with open(temp_path, "wb") as f:
            f.write(report_file.getbuffer())

        with st.spinner("Extracting and summarizing..."):
            try:
                summary = summarize_report(temp_path)
                st.success("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"Failed to summarize the report: {e}")
            finally:
                os.remove(temp_path)