"""
MEDAI: Streamlit App for Medical Assistant
Modules:
- Chatbot (LLM + Vectorstore + Memory)
- Report Summarizer (PDF/Image OCR + Summarization)
- Skin Disease Classifier (YOLOv11s)
"""

import os
import shutil
import streamlit as st
from pathlib import Path

# Import modules
from Chatbot.vectorstorebuilder import get_vectorstore
from Chatbot.retrievalqa import get_retrieval_chain
from Skin_Disease_Classifier.skin_disease_classifier import save_uploaded_image, predict
from Report_Summarizer.report_summarizer import summarize_report

# Streamlit setup
st.set_page_config(page_title="MEDAI: Medical Assistant", layout="wide")
st.title("MEDAI - Medical Assistant for Skin Health")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "Chatbot (Skin Q&A)",
    "Report Summarizer",
    "Skin Disease Classifier"
])

# Chatbot Tab 
with tab1:
    st.header("Medical Skin Assistant Chatbot")

    if "chat_chain" not in st.session_state:
        try:
            st.session_state.chat_chain = get_retrieval_chain()
            st.session_state.chat_history = []
        except Exception as e:
            st.error(f"Failed to load chatbot: {e}")

    if "chat_chain" in st.session_state:
        with st.form("chatbot_form", clear_on_submit=True):
            user_input = st.text_area("Ask a question (e.g., symptoms or skin condition):", height=70)
            submit = st.form_submit_button("Send")

        if submit and user_input.strip():
            try:
                result = st.session_state.chat_chain.invoke({"question": user_input})
                bot_response = result.get("answer", str(result))

                # Save to session history
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("bot", bot_response))

            except Exception as e:
                st.session_state.chat_history.append(("bot", f"Error: {e}"))

       
        if "chat_history" in st.session_state:
            for role, msg in st.session_state.chat_history[-8:]:
                with st.chat_message(role):
                    st.markdown(msg)

# Report Summarizer Tab 
with tab2:
    st.header("Summarize Medical Reports")

    uploaded_report = st.file_uploader(
        "Upload a PDF or image file",
        type=["pdf", "png", "jpg", "jpeg"],
        key="summary_uploader"
    )

    if uploaded_report:
        temp_dir = Path(__file__).resolve().parent / "Report_Summarizer" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded_report.name

        with open(temp_path, "wb") as f:
            f.write(uploaded_report.getbuffer())

        with st.spinner("Summarizing report..."):
            summary = summarize_report(str(temp_path))

        st.subheader("Summary:")
        st.write(summary)

        #  Cleanup temp folder
        shutil.rmtree(temp_dir, ignore_errors=True)

# Skin Disease Classifier Tab 
with tab3:
    st.header("Skin Disease Classifier")

    uploaded_img = st.file_uploader(
        "Upload a skin image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="img_uploader"
    )

    if uploaded_img:
        try:
            img_path = save_uploaded_image(uploaded_img)
            st.image(img_path, caption="Uploaded Image", width=300)

            with st.spinner("Classifying..."):
                df = predict(img_path)

            if not df.empty:
                st.subheader("Prediction Results")
                st.dataframe(df, hide_index=True)
                top = df.iloc[0]
                st.success(f"Most likely: {top['class']} (confidence: {top['confidence'] * 100:.1f}%)")
            else:
                st.warning("No prediction could be made. Try another image.")

            # Cleanup temp folder
            temp_dir = Path(__file__).resolve().parent / "Skin_Disease_Classifier" / "temp"
            shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            st.error(f"Classification failed: {e}")


