import streamlit as st
import google.generativeai as genai
from typing import List, Dict, Optional
import os
import json
from pathlib import Path
import fitz  # PyMuPDF for PDF text extraction
from datetime import datetime
import re

class PDFChatbot:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the PDF Chatbot with Gemini 1.5
        
        Args:
            api_key: Your Google AI API key
            model_name: Gemini model to use (default: gemini-1.5-flash)
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.pdf_content = ""
        self.chat_history = []
        self.pdf_filename = ""
        self.page_count = 0
        
    def load_pdf_text(self, pdf_text: str, filename: str = "uploaded_text"):
        """
        Load PDF text content into the chatbot
        
        Args:
            pdf_text: The extracted text content from PDF
            filename: Name of the source file
        """
        self.pdf_content = pdf_text
        self.pdf_filename = filename
        return f"PDF content loaded successfully. Length: {len(pdf_text)} characters"
        
    def load_pdf_from_file(self, file_path: str):
        """
        Load PDF text from a file (supports both .pdf and .txt files)
        
        Args:
            file_path: Path to the PDF or text file
        """
        try:
            file_path = Path(file_path)
            self.pdf_filename = file_path.name
            
            if file_path.suffix.lower() == '.pdf':
                # Extract text from PDF using PyMuPDF
                pdf_document = fitz.open(str(file_path))
                text_content = ""
                self.page_count = pdf_document.page_count
                
                for page_num in range(self.page_count):
                    page = pdf_document.load_page(page_num)
                    page_text = page.get_text()
                    text_content += f"\n--- PAGE {page_num + 1} ---\n"
                    text_content += page_text
                    text_content += "\n"
                
                pdf_document.close()
                self.pdf_content = text_content
                return f"PDF extracted: {self.page_count} pages, {len(text_content)} characters"
                
            elif file_path.suffix.lower() == '.txt':
                # Read text file directly
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.pdf_content = file.read()
                return f"Text loaded: {len(self.pdf_content)} characters"
            
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
        except Exception as e:
            return f"Error loading file: {e}"
    
    def load_pdf_from_bytes(self, pdf_bytes: bytes, filename: str):
        """
        Load PDF from bytes (for Streamlit file upload)
        
        Args:
            pdf_bytes: PDF file as bytes
            filename: Name of the uploaded file
        """
        try:
            self.pdf_filename = filename
            
            if filename.lower().endswith('.pdf'):
                # Extract text from PDF bytes using PyMuPDF
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                text_content = ""
                self.page_count = pdf_document.page_count
                
                for page_num in range(self.page_count):
                    page = pdf_document.load_page(page_num)
                    page_text = page.get_text()
                    text_content += f"\n--- PAGE {page_num + 1} ---\n"
                    text_content += page_text
                    text_content += "\n"
                
                pdf_document.close()
                self.pdf_content = text_content
                return f"PDF extracted: {self.page_count} pages, {len(text_content)} characters"
                
            elif filename.lower().endswith('.txt'):
                # Read text file directly
                self.pdf_content = pdf_bytes.decode('utf-8')
                return f"Text loaded: {len(self.pdf_content)} characters"
            
            else:
                raise ValueError(f"Unsupported file format")
                
        except Exception as e:
            return f"Error loading file: {e}"
    
    def create_context_prompt(self, query: str) -> str:
        """
        Create a context-aware prompt combining PDF content with user query
        
        Args:
            query: User's question
            
        Returns:
            Formatted prompt with context
        """
        context_prompt = f"""
You are an AI assistant that answers questions based on the provided PDF document content.

PDF Document Content:
{self.pdf_content}

User Question: {query}

Instructions:
1. Answer the question based solely on the information provided in the PDF document above
2. If the answer cannot be found in the document, clearly state that the information is not available in the provided content
3. Provide specific quotes or references from the document when possible
4. Be concise but comprehensive in your response
5. When identifying table of contents or page mappings, be very precise about page numbers

Answer:
"""
        return context_prompt
    
    def chat(self, query: str) -> str:
        """
        Chat with the PDF content using Gemini
        
        Args:
            query: User's question about the PDF content
            
        Returns:
            Gemini's response based on the PDF content
        """
        if not self.pdf_content:
            return "No PDF content loaded. Please load PDF content first."
        
        try:
            # Create context-aware prompt
            prompt = self.create_context_prompt(query)
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            # Store in chat history
            self.chat_history.append({
                "query": query,
                "response": response.text,
                "timestamp": datetime.now().isoformat()
            })
            
            return response.text
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def get_table_of_contents_json(self) -> Optional[Dict]:
        """
        Extract table of contents and return as JSON with page mappings
        
        Returns:
            Dictionary containing TOC with page numbers
        """
        if not self.pdf_content:
            return None
            
        toc_query = """
        Identify the entire table of contents from the given PDF and produce a structured mapping of sections/topics with their corresponding page numbers.
        
        Please return the response in the following JSON format:
        {
            "document_title": "Title of the document",
            "total_pages": number_of_pages,
            "table_of_contents": [
                {
                    "section_number": "1",
                    "title": "Section Title",
                    "page_number": page_number,
                    "subsections": [
                        {
                            "section_number": "1.1",
                            "title": "Subsection Title",
                            "page_number": page_number
                        }
                    ]
                }
            ]
        }
        
        If no clear table of contents exists, extract major headings and their page numbers instead.
        """
        
        try:
            response = self.chat(toc_query)
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    toc_data = json.loads(json_str)
                    # Add metadata
                    toc_data["extracted_at"] = datetime.now().isoformat()
                    toc_data["source_file"] = self.pdf_filename
                    return toc_data
                except json.JSONDecodeError:
                    pass
            
            # If JSON extraction fails, create a structured response
            return {
                "document_title": self.pdf_filename,
                "total_pages": self.page_count,
                "extracted_at": datetime.now().isoformat(),
                "source_file": self.pdf_filename,
                "raw_response": response,
                "note": "Unable to parse structured JSON, see raw_response for details"
            }
            
        except Exception as e:
            return {
                "error": f"Error extracting TOC: {e}",
                "document_title": self.pdf_filename,
                "extracted_at": datetime.now().isoformat()
            }
    
    def get_chat_history(self) -> List[Dict]:
        """Get the chat history"""
        return self.chat_history
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []


def main():
    st.set_page_config(
        page_title="PDF Chatbot with TOC Extractor",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF Chatbot with Table of Contents Extractor")
    st.markdown("Upload a PDF and chat with its content, plus extract table of contents as JSON!")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'toc_data' not in st.session_state:
        st.session_state.toc_data = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google AI API Key",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        
        # Model selection
        model_name = st.selectbox(
            "Model",
            ["gemini-1.5-flash", "gemini-1.5-pro"],
            help="Choose the Gemini model to use"
        )
        
        st.divider()
        
        # File upload
        st.header("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF or text file",
            type=['pdf', 'txt'],
            help="Upload a PDF or text file to chat with"
        )
        
        if uploaded_file and api_key:
            if st.button("Load Document"):
                with st.spinner("Loading document..."):
                    # Initialize chatbot
                    st.session_state.chatbot = PDFChatbot(api_key, model_name)
                    
                    # Load the file
                    result = st.session_state.chatbot.load_pdf_from_bytes(
                        uploaded_file.read(), 
                        uploaded_file.name
                    )
                    st.success(result)
                    
                    # Clear previous chat history and TOC data
                    st.session_state.chat_history = []
                    st.session_state.toc_data = None
        
        st.divider()
        
        # Extract TOC button
        if st.session_state.chatbot and st.session_state.chatbot.pdf_content:
            if st.button("Extract Table of Contents"):
                with st.spinner("Extracting table of contents..."):
                    st.session_state.toc_data = st.session_state.chatbot.get_table_of_contents_json()
                    st.success("Table of contents extracted!")
        
        # Download TOC as JSON
        if st.session_state.toc_data:
            json_str = json.dumps(st.session_state.toc_data, indent=2)
            st.download_button(
                label="📥 Download TOC as JSON",
                data=json_str,
                file_name=f"toc_{st.session_state.chatbot.pdf_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Main content area
    if not api_key:
        st.warning("Please enter your Google AI API key in the sidebar to get started.")
        st.info("You can get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    if not uploaded_file:
        st.info("Please upload a PDF or text file in the sidebar to begin.")
        return
    
    if not st.session_state.chatbot:
        st.info("Please click 'Load Document' in the sidebar to process your file.")
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chat with Your Document")
        
        # Chat interface
        user_query = st.text_input(
            "Ask a question about your document:",
            placeholder="e.g., What is the main topic of this document?"
        )
        
        if st.button("Send") and user_query:
            with st.spinner("Generating response..."):
                response = st.session_state.chatbot.chat(user_query)
                st.session_state.chat_history.append({
                    "query": user_query,
                    "response": response,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q: {chat['query'][:50]}... ({chat['timestamp']})"):
                    st.write("**Question:**", chat['query'])
                    st.write("**Answer:**", chat['response'])
        
        # Quick action buttons
        st.subheader("Quick Actions")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("📋 Get Document Summary"):
                with st.spinner("Generating summary..."):
                    summary = st.session_state.chatbot.chat("Provide a comprehensive summary of this document including main topics, key points, and structure.")
                    st.session_state.chat_history.append({
                        "query": "Document Summary",
                        "response": summary,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
        
        with col_b:
            if st.button("📑 Extract TOC"):
                with st.spinner("Extracting table of contents..."):
                    toc_response = st.session_state.chatbot.chat("Identify the entire table of contents from the given PDF and produce a mapping of section start and PDF page numbers.")
                    st.session_state.chat_history.append({
                        "query": "Table of Contents",
                        "response": toc_response,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
        
        with col_c:
            if st.button("🔍 Key Topics"):
                with st.spinner("Identifying key topics..."):
                    topics = st.session_state.chatbot.chat("What are the main topics and themes covered in this document? List them with brief descriptions.")
                    st.session_state.chat_history.append({
                        "query": "Key Topics",
                        "response": topics,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
    
    with col2:
        st.header("Document Info")
        
        if st.session_state.chatbot:
            st.write("**Filename:**", st.session_state.chatbot.pdf_filename)
            st.write("**Pages:**", st.session_state.chatbot.page_count)
            st.write("**Content Length:**", f"{len(st.session_state.chatbot.pdf_content):,} characters")
        
        # Display TOC data if available
        if st.session_state.toc_data:
            st.header("Table of Contents")
            
            if "error" in st.session_state.toc_data:
                st.error(st.session_state.toc_data["error"])
            elif "raw_response" in st.session_state.toc_data:
                st.warning("Structured JSON parsing failed. Raw response:")
                st.text_area("Raw TOC Response", st.session_state.toc_data["raw_response"], height=300)
            else:
                st.json(st.session_state.toc_data)
        
        # Clear history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            if st.session_state.chatbot:
                st.session_state.chatbot.clear_history()
            st.success("Chat history cleared!")


if __name__ == "__main__":
    main()