import os
import streamlit as st
from Chatbot.retrievalqa import get_retrieval_chain
from Skin_Disease_Classifier.skin_disease_classifier import save_uploaded_image, predict
from Report_Summarizer.report_summarizer import summarize_report

# Initialize session state for chatbot
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chain = get_retrieval_chain()

st.set_page_config(page_title="MEDAI Assistant", layout="wide")
st.title("🩺 MEDAI: Your Medical AI Assistant")

# Sidebar Navigation
mode = st.sidebar.radio("Select Module:", ["💬 Chatbot", "🖼️ Skin Disease Classifier", "📄 Report Summarizer"])

if mode == "💬 Chatbot":
    st.subheader("Chat with the Medical Assistant")
    user_input = st.text_input("Ask a medical question (e.g., 'What is eczema?' or 'I have redness and itching'):", key="chat_input")

    if user_input:
        with st.spinner("Thinking..."):
            result = st.session_state.chain.invoke({"question": user_input})
            answer = result.get("answer", "No response.")
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("bot", answer))

            # Display the answer
            st.chat_message("bot").markdown(answer)

            # Display source document chunks (RAG proof)
            if "source_documents" in result:
                st.markdown("###  Retrieved Chunks from Vectorstore")
                for i, doc in enumerate(result["source_documents"]):
                    chunk = doc.page_content.strip()[:600]  # Limit to 600 chars
                    source_file = doc.metadata.get("source", "Unknown source")
                    st.markdown(f"**Chunk {i+1} from `{source_file}`:**")
                    st.code(chunk, language="markdown")

elif mode == "🖼️ Skin Disease Classifier":
    st.subheader("Upload an image for skin disease prediction")
    uploaded_img = st.file_uploader("Choose an image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        img_path = save_uploaded_image(uploaded_img)
        with st.spinner("Classifying image..."):
            df = predict(img_path)
        st.image(uploaded_img, caption="Uploaded Image", use_container_width =True)
        if not df.empty:
            st.success("Prediction Results:")
            st.dataframe(df)
        else:
            st.warning("No prediction made. Please try another image.")

elif mode == "📄 Report Summarizer":
    st.subheader("Upload a lab report (PDF/Image) for summarization")
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

st.markdown("---")
st.caption(" Disclaimer: This tool is for educational and awareness purposes only. Please consult a doctor for any medical concerns.")
