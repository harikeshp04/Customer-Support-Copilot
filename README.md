# Customer Support Copilot

An **AI-powered customer support copilot** that:
- Classifies support tickets by **topic, sentiment, and priority**.
- Retrieves relevant knowledge base docs using **RAG (Retrieval-Augmented Generation)**.
- Generates **AI-assisted responses** with sources.
- Runs in a simple **Streamlit web app**.

---

## 🚀 Features
- **Bulk Ticket Classification**  
  Upload a CSV of tickets or use the provided sample.  
  Each ticket is classified into:
  - Tags (e.g., *How-to, API/SDK, Connector*).
  - Sentiment (e.g., *Angry, Curious, Neutral*).
  - Priority (`P0`, `P1`, `P2`).

- **Interactive AI Agent**  
  Submit a ticket interactively and get:
  - Internal Analysis (tags, sentiment, priority).
  - Final AI-generated response (with sources).

- **Retrieval-Augmented Generation (RAG)**  
  - Retrieves docs from the knowledge base (scraped into `kb/`).
  - Uses embeddings + FAISS (or cosine similarity fallback).
  - Answers generated using:
    - **OpenAI GPT-3.5** (if API key available).  
    - **Local Hugging Face model (`flan-t5-base`)** as fallback (no key required).

---

# Customer Support Copilot

## 📂 Project Structure
customer-support-copilot/
│
├── app.py # Streamlit frontend  
├── classifier.py # Ticket classification (topics, sentiment, priority)  
├── rag.py # Retrieval-Augmented Generation engine  
├── create_sample_tickets.py # Generate sample_tickets.csv  
├── sample_tickets.csv # Example tickets  
├── kb/ # Knowledge base text files  
├── requirements.txt # Python dependencies  
└── README.md # This file  

---

📘 Usage
- Bulk Ticket Classification
- Upload a sample_tickets.csv (or use default).
- View classifications in a table.
- Download classified tickets.
- Interactive AI Agent
- Submit a new ticket.
- See internal analysis (tags, sentiment, priority).
- View final AI-generated response with sources.

🔍 Knowledge Base
- Add .txt files into kb/ folder or use fetch_and_save(url) in rag.py to scrape docs.
- Build/rebuild index from the Streamlit sidebar.

🧩 Tech Stack
- Frontend: Streamlit
- Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
- Retrieval: FAISS / cosine similarity
- Classification: Hugging Face zero-shot (facebook/bart-large-mnli) + VADER
Generation:
- OpenAI GPT-3.5 (if API key provided)
- Hugging Face flan-t5-base (local fallback)
 --- 

## 🚀 Setup Instructions

1. Clone and enter the project
```bash

git clone <repo-url>
cd customer-support-copilot

2. Create and activate virtual environment

Windows (PowerShell):
python -m venv venv
venv\Scripts\Activate.ps1

3. Install requirements
pip install -r requirements.txt

4. (Optional) Set OpenAI API Key
$env:OPENAI_API_KEY="your-key-here"

5.Run the App
streamlit run app.py





