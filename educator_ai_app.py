# educator_ai_app.py (Final Version)

import os
import uuid
import fitz  # PyMuPDF
import faiss
import shutil
import tempfile
import numpy as np
from typing import List
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from groq import Groq
import gradio as gr
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
import pyttsx3
import base64
from io import BytesIO
from dotenv import load_dotenv



# === Configuration ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)


load_dotenv()  # This loads the .env file

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
  # Replace with your actual key

# === Global Variables ===
texts = []
global_text = ""
global_summary = ""

# === FastAPI App ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Utility Functions ===
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)

def chunk_text(text: str, max_tokens: int = 500):
    sentences = text.split('.')
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) < max_tokens:
            current += sent + "."
        else:
            chunks.append(current.strip())
            current = sent + "."
    if current:
        chunks.append(current.strip())
    return chunks

def summarize_text(text: str):
    chunks = chunk_text(text, 1000)
    summaries = []
    for chunk in chunks:
        max_len = min(150, len(chunk))
        summary = summarizer(chunk, max_length=max_len, min_length=40, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return " ".join(summaries)

def build_faiss_index(chunks: List[str]):
    global texts, index
    texts = chunks
    index = faiss.IndexFlatL2(embedding_dim)
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    index.add(embeddings)

def query_index(question: str, top_k=3):
    if index.ntotal == 0:
        return []
    query_embedding = embedding_model.encode([question], convert_to_numpy=True)
    D, I = index.search(query_embedding, top_k)
    return [texts[i] for i in I[0] if i != -1 and i < len(texts)]

def get_answer_from_context(question: str, context: str):
    prompt = f"""
    You are an AI educator. Based only on the given context, answer the question below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_mcqs_from_text(text: str, num_questions: int = 5):
    prompt = f"""
    You are an AI educator. Based on the following content, generate {num_questions} multiple choice questions with 4 options each.
    Clearly mark the correct answer with (\u2714). Format the output as:

    Q1. ...
    a) ...
    b) ...
    c) ...
    d) ...
    Answer: ...

    Content:
    {text}
    """
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_pdf_report(summary, question, answer, mcqs, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y = height - 50

    def draw_text_block(title, content, y_pos):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, y_pos, title)
        y_pos -= 20
        c.setFont("Helvetica", 12)
        wrapped = simpleSplit(content, "Helvetica", 12, width - 80)
        for line in wrapped:
            c.drawString(40, y_pos, line)
            y_pos -= 16
            if y_pos < 100:
                c.showPage()
                y_pos = height - 50
        return y_pos - 10

    y = draw_text_block("\ud83d\udcc4 Summary", summary, y)
    y = draw_text_block("\u2753 Question", question, y)
    y = draw_text_block("\u2705 Answer", answer, y)
    y = draw_text_block("\ud83d\udcdd MCQs", mcqs, y)
    c.save()

# === FastAPI Upload Endpoint (API users only) ===
@app.post("/upload")
async def upload_pdf(file: UploadFile):
    global global_text, global_summary
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        pdf_path = tmp.name

    global_text = extract_text_from_pdf(pdf_path)
    global_summary = summarize_text(global_text)
    chunks = chunk_text(global_text)
    build_faiss_index(chunks)
    os.remove(pdf_path)

    return {"summary": global_summary}

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QuestionRequest):
    context = "\n".join(query_index(req.question))
    answer = get_answer_from_context(req.question, context)
    return {"answer": answer}

# === Gradio UI ===
def upload_and_summarize_gr(file_path):
    global global_text, global_summary
    global_text = extract_text_from_pdf(file_path)
    global_summary = summarize_text(global_text)
    chunks = chunk_text(global_text)
    build_faiss_index(chunks)
    return global_summary

def tts_summary(summary_text):
    import pyttsx3
    import os
    import uuid
    import tempfile

    temp_path = os.path.join(tempfile.gettempdir(), f"summary_{uuid.uuid4().hex}.wav")
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)
    engine.save_to_file(summary_text, temp_path)
    engine.runAndWait()
    # Ensure the file exists and is not empty
    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
        return temp_path
    else:
        return None

def ask_and_generate_gr(question):
    context = "\n".join(query_index(question))
    if not context:
        return "No relevant content found.", "", None
    answer = get_answer_from_context(question, context)
    mcqs = generate_mcqs_from_text(global_summary)
    pdf_path = os.path.join(tempfile.gettempdir(), f"report_{uuid.uuid4().hex}.pdf")
    generate_pdf_report(global_summary, question, answer, mcqs, pdf_path)
    return answer, mcqs, pdf_path

# Custom CSS for beautiful styling
custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    padding: 20px;
}

.main-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 30px;
    margin: 20px auto;
    max-width: 1200px;
}

.header {
    text-align: center;
    margin-bottom: 40px;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    color: white;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}

.header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    color: white;
}

.header p {
    font-size: 1.1rem;
    margin: 10px 0 0 0;
    opacity: 0.9;
    color: white;
}

.upload-section, .qa-section {
    background: white;
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(0, 0, 0, 0.05);
}

.section-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.file-upload {
    border: 2px dashed #cbd5e0;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    transition: all 0.3s ease;
    background: #f7fafc;
    min-height: 150px;
}

.file-upload:hover {
    border-color: #667eea;
    background: #edf2f7;
}

.textbox {
    border-radius: 10px;
    border: 2px solid #e2e8f0;
    transition: all 0.3s ease;
}

.textbox:focus-within {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 10px;
    color: white;
    font-weight: 600;
    padding: 12px 24px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
}

.output-box {
    background: #f8fafc;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    padding: 15px;
    margin: 10px 0;
}

.answer-box {
    background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
    color: white;
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3);
}

.mcq-box {
    background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
    color: white;
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 4px 15px rgba(237, 137, 54, 0.3);
}

.download-box {
    background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
    color: white;
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 4px 15px rgba(66, 153, 225, 0.3);
}

.icon {
    font-size: 1.2em;
    margin-right: 8px;
}

.loading {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
}

.spinner {
    border: 3px solid #f3f3f3;
    border-top: 3px solid #667eea;
    border-radius: 50%;
    width: 30px;
    height: 30px;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.success-message {
    background: #48bb78;
    color: white;
    padding: 10px 15px;
    border-radius: 8px;
    margin: 10px 0;
    font-weight: 500;
}

.error-message {
    background: #f56565;
    color: white;
    padding: 10px 15px;
    border-radius: 8px;
    margin: 10px 0;
    font-weight: 500;
}

#tts-summary-btn {
    display: inline-block;
    margin-left: 8px;
    vertical-align: middle;
    font-size: 1.2em;
    padding: 4px 10px;
}
"""

# Gradio UI Setup
with gr.Blocks(css=custom_css, title="Educator AI - Smart Learning Assistant") as gradio_app:
    
    # Header Section
    with gr.Row():
        with gr.Column(elem_classes=["header"]):
            gr.HTML("""
                <div style="text-align: center;">
                    <h1>🎓 Educator AI</h1>
                    <p>Your intelligent learning companion - Upload PDFs, ask questions, and generate MCQs</p>
                </div>
            """)
    
    # Main Content
    with gr.Row():
        with gr.Column(elem_classes=["main-container"]):
            
            # Upload and Summary Section (side by side)
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["upload-section"]):
                    gr.HTML('<div class="section-title">📄 <span>Upload Your Document</span></div>')
                    file_input = gr.File(
                        label="📁 Choose PDF File",
                        file_types=[".pdf"],
                        elem_classes=["file-upload"]
                    )
                with gr.Column(scale=1, elem_classes=["summary-section"]):
                    summary_output = gr.Textbox(
                        label="📝 Document Summary",
                        lines=8,
                        placeholder="Upload a PDF to see the summary here...",
                        elem_classes=["output-box"]
                    )
                    tts_button = gr.Button("🔊", elem_id="tts-summary-btn", elem_classes=["button"], size="sm",)
                    tts_audio = gr.Audio(label="Listen to Summary", type="filepath", visible=True)
                # Add the event handler after both components are defined
                file_input.change(
                    fn=upload_and_summarize_gr,
                    inputs=file_input,
                    outputs=summary_output
                )
                tts_button.click(
                    fn=tts_summary,
                    inputs=summary_output,
                    outputs=tts_audio
                )
            
            # Q&A Section
            with gr.Row():
                with gr.Column(elem_classes=["qa-section"]):
                    gr.HTML('<div class="section-title">❓ <span>Ask Questions & Generate MCQs</span></div>')
                    
                    question_input = gr.Textbox(
                        label="💭 Ask a Question",
                        placeholder="Type your question here...",
                        lines=2,
                        elem_classes=["textbox"]
                    )
                    
                    with gr.Row():
                        ask_btn = gr.Button(
                            "🔍 Ask Question",
                            variant="primary",
                            elem_classes=["button"]
                        )
                        generate_mcq_btn = gr.Button(
                            "📝 Generate MCQs",
                            variant="secondary",
                            elem_classes=["button"]
                        )
                    
                    # Answer Section
                    with gr.Row():
                        with gr.Column():
                            answer_output = gr.Textbox(
                                label="✅ AI Answer",
                                lines=6,
                                placeholder="Your answer will appear here...",
                                elem_classes=["answer-box"]
                            )
                    
                    # MCQ Section
                    with gr.Row():
                        with gr.Column():
                            mcq_output = gr.Textbox(
                                label="📋 Generated Multiple Choice Questions",
                                lines=12,
                                placeholder="MCQs will be generated here...",
                                elem_classes=["mcq-box"]
                            )
                    
                    # Download Section
                    with gr.Row():
                        with gr.Column():
                            download_output = gr.File(
                                label="📥 Download Report (PDF)",
                                elem_classes=["download-box"]
                            )
                    
                    # Event handlers
                    question_input.submit(
                        fn=ask_and_generate_gr,
                        inputs=question_input,
                        outputs=[answer_output, mcq_output, download_output]
                    )
                    
                    ask_btn.click(
                        fn=ask_and_generate_gr,
                        inputs=question_input,
                        outputs=[answer_output, mcq_output, download_output]
                    )
                    
                    generate_mcq_btn.click(
                        fn=lambda: ("", generate_mcqs_from_text(global_summary), None),
                        outputs=[answer_output, mcq_output, download_output]
                    )
            
            # Footer
            with gr.Row():
                gr.HTML("""
                    <div style="text-align: center; margin-top: 30px; color: #718096; font-size: 0.9rem;">
                        <p>🚀 Powered by AI • 📚 Smart Learning • 🎯 Intelligent Education</p>
                    </div>
                """)

# === Launch ===
if __name__ == "__main__":
    gradio_app.launch()
