# PDF Analysis and Chat Interface

A powerful PDF analysis tool that combines RAG (Retrieval Augmented Generation) with a chat interface to analyze and interact with PDF documents. The application extracts text and images from PDFs, creates embeddings for efficient search, and provides a user-friendly interface for querying document content.

## Features

- PDF text extraction with page number tracking
- Figure and table reference detection
- Image extraction from PDFs
- RAG-based question answering
- Interactive chat interface
- Context-aware responses with page and reference numbers
- Vector similarity search using FAISS

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install Ollama (for LLM functionality):
```bash
# On macOS
curl https://ollama.ai/install.sh | sh

# On Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

5. Pull the required Ollama model:
```bash
ollama pull llama2
```

## Dependencies

- `ollama`: For LLM functionality
- `gradio`: For the web interface
- `pypdf`: For PDF text extraction
- `Pillow`: For image processing
- `PyMuPDF`: For advanced PDF processing
- `langchain`: For RAG implementation
- `langchain-community`: For vector stores
- `langchain-huggingface`: For HuggingFace embeddings
- `faiss-cpu`: For vector similarity search
- `sentence-transformers`: For text embeddings

## Usage

1. Start the application:
```bash
python research.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:7860)

3. Upload a PDF file using the interface

4. Click "Process PDF" to extract content and create the vector store

5. Start asking questions about the PDF content in the chat interface

## Technical Details

### Core Components

1. **PDF Processing**
   - `extract_images_from_pdf`: Extracts images using PyMuPDF
   - `extract_text_with_metadata`: Extracts text with page numbers and reference tracking

2. **Vector Store**
   - `create_vector_store`: Creates FAISS vector store from text chunks
   - `get_relevant_context`: Retrieves relevant context using similarity search

3. **Chat Interface**
   - `ChatBot` class: Manages PDF processing and chat interactions
   - `create_gradio_interface`: Sets up the Gradio web interface

### How It Works

1. **Document Processing**
   - PDF is processed to extract text and images
   - Text is split into chunks with metadata (page numbers, figure/table references)
   - Chunks are converted to embeddings using HuggingFace's model

2. **Query Processing**
   - User query is matched against document embeddings
   - Most relevant chunks are retrieved using FAISS
   - Context is sent to Ollama for generating responses

3. **Response Generation**
   - LLM generates responses based on relevant context
   - Responses include page numbers and reference numbers when applicable

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 