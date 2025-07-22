import os
import streamlit as st
from Chatbot.retrievalqa import chat 
from Skin_Disease_Classifier.skin_disease_classifier import save_uploaded_image, predict
from Report_Summarizer.report_summarizer import summarize_report

# Set page configuration
st.set_page_config(page_title="MEDAI Assistant", layout="wide")
st.title("🩺 MEDAI: Your Medical AI Assistant")

# Sidebar Navigation
mode = st.sidebar.radio("Select Module:", ["💬 Chatbot", "🖼️ Skin Disease Classifier", "📄 Report Summarizer"])

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "medai_chat"  

#  CHATBOT 
if mode == "💬 Chatbot":
    st.subheader("Chat with the Medical Assistant")

    # Button to clear chat
    if st.button("🗑️ Start New Chat"):
        st.session_state.chat_history = []

    # Show previous chat
    for role, message in st.session_state.chat_history:
        st.chat_message(role).markdown(message)

    # Chat input
    user_input = st.chat_input("Ask a medical question (e.g., 'What is eczema?' or 'I have redness and itching'):")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append(("user", user_input))

        with st.spinner("Thinking..."):
            try:
                result = chat(user_input, session_id=st.session_state.session_id)
                answer = result.get("answer", "No response generated.")
                st.chat_message("bot").markdown(answer)
                st.session_state.chat_history.append(("bot", answer))
            except Exception as e:
                st.error(f"An error occurred: {e}")

#  SKIN DISEASE CLASSIFIER  
elif mode == "🖼️ Skin Disease Classifier":
    st.subheader("Upload an image for skin disease prediction")
    uploaded_img = st.file_uploader("Choose an image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        img_path = save_uploaded_image(uploaded_img)
        with st.spinner("Classifying image..."):
            df = predict(img_path)
        st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)
        if not df.empty:
            st.success("Prediction Results:")
            st.dataframe(df)
        else:
            st.warning("No prediction made. Please try another image.")

# REPORT SUMMARIZER
elif mode == "📄 Report Summarizer":
    st.subheader("Upload a medical report (PDF/Image) for summarization")
    report_file = st.file_uploader("Choose a report file (PDF, JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"])

    if report_file:
        temp_dir = "temp_reports"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, report_file.name)
        with open(file_path, "wb") as f:
            f.write(report_file.getbuffer())

        with st.spinner("Summarizing report..."):
            summary = summarize_report(file_path)
        st.text_area("Summary", summary, height=250)

#  DISCLAIMER  
st.markdown("---")
st.caption("Disclaimer: This tool is for educational and awareness purposes only. Please consult a doctor for any medical concerns.")