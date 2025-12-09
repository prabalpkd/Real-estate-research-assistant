# 🏣 Real Estate Research Assistant

The **Real Estate Research Assistant** is an AI-powered tool designed for real estate analysts to quickly extract insights from online articles. 
Instead of spending time reading long articles, analysts can provide URLs and ask questions to get concise, AI-generated answers along with references to the source content.

This tool leverages **document embeddings**, a **vector database**, and a **large language model (LLM)** to deliver fast and accurate answers to user queries.

---

## 🔹 Key Features

- **Process Multiple URLs:** Paste one or multiple URLs, and the tool scrapes and processes the content.
- **Text Chunking & Embeddings:** Long articles are split into chunks and stored as embeddings in a vector database for fast retrieval.
- **AI Question Answering:** Ask any question related to the articles, and the LLM retrieves and summarizes relevant information.
- **Sources Tracking:** All answers include the original sources for verification.
- **Interactive Web Interface:** Built using Streamlit for easy interaction.

---

## 🛠️ Installation

1. **Clone the Repository**

```bash
git clone <repository-url>
cd real-estate-research-assistant
```

## 🖥️ How to Use

1. **Run the Streamlit App**

```bash
streamlit run main.py
```
2. **Add URLs**
   
   -->Use the sidebar to enter one or more URLs of real estate news, blogs, or articles.

   -->Click Process URLs to scrape and store the articles in a vector database.

   -->The tool splits the content into smaller chunks for better retrieval by the AI.

3. **Ask Questions**

      Once URLs are processed, enter your question in the text box.
      
      Example questions:
      
         --> “What was the 30-year fixed mortgage rate last month?”
      
         --> “Summarize the current property trends in Mumbai.”
      
      The AI will provide a concise answer and list the sources from which the information was extracted.

## 🏗️ Architecture Overview

The tool is built in a modular way for easy maintenance and scalability:

1. Frontend (main.py)

   --> Streamlit interface for input URLs and questions.

   --> Displays processing status and AI answers.

2. Backend (rag.py)

   --> Handles URL scraping and text extraction.

   --> Splits text into smaller chunks for better embeddings.

   --> Creates embeddings using HuggingFace models.

   --> Stores embeddings in a Chroma vector database for fast retrieval.

   --> Uses ChatGroq LLM to generate answers based on retrieved content.

3. Vector Database

   --> Stores chunks of documents as embeddings.

   --> Enables semantic search for relevant content.

4. LLM

   --> ChatGroq LLM generates answers from retrieved content.

## ⚡ Requirements

1. Python 3.10+

2. Streamlit

3. LangChain

4. Chroma Vector Database

5. HuggingFace Embeddings (Alibaba-NLP/gte-base-en-v1.5)

6. ChatGroq LLM

## 📝 Notes

   --> Always provide at least one valid URL before asking a question.

   --> The first run may take a few minutes because it initializes the LLM and the vector database.

   --> The tool works best with articles that are in English.


