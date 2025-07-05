# Educator AI - Smart Learning Assistant

Educator AI is an intelligent learning companion that allows you to upload PDF documents, receive AI-generated summaries, ask questions, and generate multiple-choice questions (MCQs) for study and revision. It features a beautiful Gradio web interface and robust backend powered by state-of-the-art AI models.

---

## 🚀 Features

- **PDF Upload & Summarization:** Upload any PDF and get a concise summary.
- **AI Q&A:** Ask questions about your document and get context-aware answers.
- **MCQ Generation:** Instantly generate multiple-choice questions from your document.
- **PDF Report:** Download a report containing the summary, your question, the answer, and generated MCQs.
- **Text-to-Speech:** Listen to the summary with a single click.
- **Modern UI:** Beautiful, user-friendly Gradio interface with custom styling.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/mariatignatious/Educator-AI.git
cd Educator-AI
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
python -m venv myenv1
# On Windows:
myenv1\Scripts\activate
# On Mac/Linux:
source myenv1/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root with your [Groq API key](https://console.groq.com/):
```
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🏃‍♂️ Running the App

```bash
python educator_ai_app.py
```

The Gradio web interface will launch in your browser. Follow the on-screen instructions to upload a PDF, ask questions, and generate MCQs.

---

## 🖥️ API Endpoints

The app also exposes a FastAPI backend with the following endpoints:

- `POST /upload` — Upload a PDF and get a summary (API users only)
- `POST /ask` — Ask a question about the uploaded document

---

## 📦 Dependencies
- Python 3.8+
- [Gradio](https://gradio.app/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [faiss-cpu](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [Transformers](https://huggingface.co/transformers/)
- [Groq API](https://console.groq.com/)
- [ReportLab](https://www.reportlab.com/)
- [pyttsx3](https://pyttsx3.readthedocs.io/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

Install all dependencies with `pip install -r requirements.txt`.

---

## 📝 Usage Tips
- For best results, upload clear, text-based PDFs (not scanned images).
- The Groq API key is required for AI-powered Q&A and MCQ generation.
- All processing is done locally except for calls to the Groq API.

---

## Credits
- UI powered by [Gradio](https://gradio.app/).
- PDF processing by [PyMuPDF](https://pymupdf.readthedocs.io/).
- Summarization and embeddings by [HuggingFace Transformers](https://huggingface.co/transformers/) and [Sentence Transformers](https://www.sbert.net/).
- MCQ and Q&A powered by [Groq LLMs](https://console.groq.com/).

---

## 📄 License
This project is for educational and research purposes. See `LICENSE` for details. 