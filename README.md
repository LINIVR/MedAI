MedAI – AI-Powered Medical Assistant 🩺
MedAI is a multi-modular healthcare application integrating LLMs, computer vision, and OCR to deliver medical insights, skin disease classification, and report summarization via a clean Streamlit interface. The system is designed to promote medical awareness while maintaining ethical safeguards such as disclaimers and avoiding hallucinated information.

📌 Features
 1. Medical Chatbot (RAG + LLM)
Handles two types of skin-related queries:

Symptom-based Queries: Retrieves matching diseases from a FAISS Vector Store and prompts users to choose.

Disease-specific Queries: Fetches relevant chunks and generates structured responses.

Key Technologies:

FAISS for document retrieval

Hugging Face embeddings (sentence-transformers/all-MiniLM-L6-v2)

LangChain for RAG orchestration

Gemma 2 9B via Groq API for response generation

 2. Skin Disease Classifier
Classifies dermatological conditions from user-uploaded images.

Fine-tuned YOLOv11-small (YOLOv11s) on a custom dataset (4,773 images) with three classes:

Acne

Atopic Dermatitis

Basal Cell Carcinoma

Best Performance:

YOLOv11s: 92.86% test accuracy

Outputs class name + probability score for predictions.

Key Technologies: Ultralytics YOLO, Pandas, Matplotlib

 3. Medical Report Summarizer (OCR + NLP)
Upload a PDF or image of a medical report to get a structured summary.

Workflow:

Image → Text via Tesseract OCR

PDF → Text via PyPDF

Summarization using Hugging Face model (sshleifer/distilbart-cnn-12-6)

## 📄 Additional Documentation
For full technical details and test cases, [download the PDF here](documentation/MedAI_Documentation.pdf).


