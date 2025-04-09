"""
PDF Analysis and Chat Interface using RAG with Ollama and Gradio.

This module implements a PDF analysis system that extracts text and images from PDFs,
creates embeddings for text chunks, and provides a chat interface for querying the PDF content.
It uses RAG (Retrieval Augmented Generation) with Ollama for LLM responses and Gradio for the UI.

Classes:
    ChatBot: Manages PDF processing and chat interactions.

Functions:
    extract_images_from_pdf: Extracts images from a PDF file.
    extract_text_with_metadata: Extracts text with page numbers and figure/table references.
    create_vector_store: Creates a FAISS vector store from text documents.
    get_relevant_context: Retrieves relevant context for a query.
    create_gradio_interface: Creates and configures the Gradio interface.
"""

import ollama
import gradio as gr
from pypdf import PdfReader
from PIL import Image
import fitz
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import io
import re
from typing import List, Dict, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_images_from_pdf(pdf_path: str) -> List[Image.Image]:
    """
    Extract images from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[Image.Image]: List of PIL Image objects extracted from the PDF.

    Raises:
        FileNotFoundError: If the PDF file is not found.
        ValueError: If the PDF file is corrupted or invalid.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Failed to open PDF file {pdf_path}: {e}")
        raise
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)

    return images


def extract_text_with_metadata(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text content from PDF file with page numbers and figure/table references.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing text content and metadata.
            Each dictionary has keys:
            - content (str): Text content with metadata markers
            - page_number (int): Page number of the content

    Raises:
        FileNotFoundError: If the PDF file is not found.
        ValueError: If the PDF file is corrupted or invalid.
    """
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        logging.error(f"Failed to read PDF file {pdf_path}: {e}")
        raise
    documents = []

    # Compile regex patterns for figures and tables
    figure_pattern = re.compile(r'(Figure|Fig\.?)\s*(\d+)', re.IGNORECASE)
    table_pattern = re.compile(r'Table\s*(\d+)', re.IGNORECASE)

    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()

        # Find all figure and table references
        figures = figure_pattern.finditer(text)
        tables = table_pattern.finditer(text)

        # Add metadata markers to the text
        for match in figures:
            fig_num = match.group(2)
            text = text.replace(match.group(
                0), f"[Figure {fig_num} (Page {page_num})]")

        for match in tables:
            table_num = match.group(1)
            text = text.replace(match.group(
                0), f"[Table {table_num} (Page {page_num})]")

        # Add page number context
        text = f"[Page {page_num}]\n{text}"

        documents.append({
            "content": text,
            "page_number": page_num
        })

    return documents


def create_vector_store(text_documents: List[Dict[str, Any]]) -> FAISS:
    """
    Create a FAISS vector store from text documents with metadata.

    Args:
        text_documents (List[Dict[str, Any]]): List of text documents with metadata.

    Returns:
        FAISS: Vector store containing text embeddings.

    Note:
        Uses HuggingFace embeddings model 'sentence-transformers/all-mpnet-base-v2'
        for creating embeddings.
    """
    logging.info("Creating vector store from text documents.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    # Process each document while preserving metadata
    chunks = []
    for doc in text_documents:
        split_texts = text_splitter.split_text(doc["content"])
        # Ensure each chunk has the page number context
        chunks.extend([
            f"[Page {doc['page_number']}] {chunk}"
            for chunk in split_texts
        ])

    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vector_store = FAISS.from_texts(chunks, embeddings)

    return vector_store


def get_relevant_context(query: str, vector_store: FAISS) -> str:
    """
    Get relevant context for a query using similarity search.

    Args:
        query (str): The query string.
        vector_store (FAISS): The vector store to search in.

    Returns:
        str: Concatenated relevant context from the most similar documents.
    """
    logging.info("Retrieving relevant context for the query.")
    results = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in results])
    return context


class ChatBot:
    """
    A chatbot class that processes PDFs and provides chat functionality using RAG.

    This class manages the PDF processing pipeline and chat interactions,
    using Ollama for LLM responses and FAISS for vector storage.

    Attributes:
        vector_store (Optional[FAISS]): The vector store containing document embeddings.
        history (List[Tuple[str, str]]): Chat history as list of (message, response) tuples.
    """

    def __init__(self):
        """Initialize the ChatBot with empty vector store and history."""
        logging.info("Initializing ChatBot.")
        self.vector_store: Optional[FAISS] = None
        self.history: List[Tuple[str, str]] = []

    def process_pdf(self, pdf_file: Any) -> str:
        """
        Process a PDF file and initialize the vector store.

        Args:
            pdf_file: The PDF file object from Gradio.

        Returns:
            str: Status message indicating the number of images extracted.
        """
        try:
            images = extract_images_from_pdf(pdf_file.name)
            text_documents = extract_text_with_metadata(pdf_file.name)
            self.vector_store = create_vector_store(text_documents)
            logging.info(f"Processed PDF. Extracted {len(images)} images.")
            return f"Processed PDF. Extracted {len(images)} images and created vector store."
        except Exception as e:
            logging.error(f"Error processing PDF: {e}")
            return "Failed to process PDF. Please check the file and try again."

    def chat(self, message: str) -> str:
        """
        Process a chat message and return a response.

        Args:
            message (str): The user's message.

        Returns:
            str: The chatbot's response based on the PDF content.

        Note:
            If no PDF has been processed yet, returns a message asking to upload a PDF first.
        """
        if not self.vector_store:
            return "Please upload a PDF first."

        # Get relevant context
        context = get_relevant_context(message, self.vector_store)

        # Create prompt with context and instruction about references
        prompt = f"""Context: {context}

Question: {message}

Please answer based on the context provided. When referring to figures, tables, or specific content, include their page numbers and reference numbers as they appear in the [Figure X (Page Y)] or [Table X (Page Y)] format in the context."""

        # Get response from Ollama
        response = ollama.chat(model="llama2", messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. Always include page numbers and figure/table numbers when referencing them in your answers."
            },
            {"role": "user", "content": prompt}
        ])

        return response['message']['content']


def create_gradio_interface() -> gr.Blocks:
    """
    Create and configure the Gradio interface for the PDF chat application.

    Returns:
        gr.Blocks: The configured Gradio interface.
    """
    with gr.Blocks() as demo:
        gr.Markdown("# PDF Analysis Chatbot")

        with gr.Row():
            pdf_input = gr.File(label="Upload PDF")
            process_button = gr.Button("Process PDF")

        status_output = gr.Textbox(label="Status")

        chatbot_interface = gr.Chatbot(
            label="Chat History",
            type="messages"  # Use the new message format
        )
        msg_input = gr.Textbox(label="Ask a question about the PDF")
        send_button = gr.Button("Send")

        def respond(message: str, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
            bot_response = chatbot.chat(message)
            # Convert to the new message format
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": bot_response})
            return "", history

        process_button.click(
            fn=chatbot.process_pdf,
            inputs=[pdf_input],
            outputs=[status_output]
        )

        send_button.click(
            fn=respond,
            inputs=[msg_input, chatbot_interface],
            outputs=[msg_input, chatbot_interface]
        )

        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot_interface],
            outputs=[msg_input, chatbot_interface]
        )

    return demo


# Initialize chatbot
chatbot = ChatBot()

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        share=False,            # Don't create public link
        server_name="0.0.0.0",  # Allow external connections
    )
