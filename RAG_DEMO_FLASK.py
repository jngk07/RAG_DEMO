
from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load embedding and generator models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("t5-small")
generator_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Global variables for dynamic index and chunks
faiss_index = None
doc_chunks = []

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def split_text(text, chunk_size=300):
    sentences = text.split(". ")
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < chunk_size:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def build_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global faiss_index, doc_chunks
    file = request.files["pdf"]
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        text = extract_text_from_pdf(filepath)
        doc_chunks = split_text(text)
        faiss_index = build_faiss_index(doc_chunks)
        return jsonify({"message": "PDF uploaded and indexed successfully."})
    return jsonify({"error": "No file uploaded"}), 400

@app.route("/rag", methods=["POST"])
def rag():
    global faiss_index, doc_chunks
    if faiss_index is None:
        return jsonify({"error": "No PDF uploaded yet."}), 400

    query = request.json.get("query")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    query_embedding = embedding_model.encode([query])
    _, indices = faiss_index.search(np.array(query_embedding).astype("float32"), k=2)
    retrieved_docs = [doc_chunks[i] for i in indices[0]]

    input_text = " ".join(retrieved_docs) + " " + query
    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = generator_model.generate(inputs, max_length=50, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(port=5050)
