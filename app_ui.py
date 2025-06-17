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
        self.pdf_contents = {}  # Dictionary to store multiple PDF contents
        self.chat_history = []
        self.pdf_filenames = []
        self.total_pages = 0
        self.combined_content = ""
        
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
    
    def load_multiple_pdfs_from_bytes(self, uploaded_files: List, combine_mode: str = "separate") -> str:
        """
        Load multiple PDFs from bytes (for Streamlit file upload)
        
        Args:
            uploaded_files: List of uploaded files
            combine_mode: "separate" or "combined" - how to handle multiple PDFs
        """
        results = []
        self.pdf_contents = {}
        self.pdf_filenames = []
        self.total_pages = 0
        combined_text = ""
        
        for uploaded_file in uploaded_files:
            try:
                filename = uploaded_file.name
                self.pdf_filenames.append(filename)
                
                if filename.lower().endswith('.pdf'):
                    # Extract text from PDF bytes using PyMuPDF
                    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    text_content = ""
                    page_count = pdf_document.page_count
                    self.total_pages += page_count
                    
                    # Add document header
                    doc_header = f"\n{'='*50}\nDOCUMENT: {filename}\nPAGES: {page_count}\n{'='*50}\n"
                    text_content += doc_header
                    
                    for page_num in range(page_count):
                        page = pdf_document.load_page(page_num)
                        page_text = page.get_text()
                        text_content += f"\n--- PAGE {page_num + 1} (Document: {filename}) ---\n"
                        text_content += page_text
                        text_content += "\n"
                    
                    pdf_document.close()
                    self.pdf_contents[filename] = {
                        "content": text_content,
                        "pages": page_count,
                        "type": "pdf"
                    }
                    combined_text += text_content
                    results.append(f"✅ {filename}: {page_count} pages extracted")
                    
                elif filename.lower().endswith('.txt'):
                    # Read text file directly
                    text_content = uploaded_file.read().decode('utf-8')
                    doc_header = f"\n{'='*50}\nDOCUMENT: {filename}\n{'='*50}\n"
                    full_content = doc_header + text_content
                    
                    self.pdf_contents[filename] = {
                        "content": full_content,
                        "pages": 1,
                        "type": "txt"
                    }
                    combined_text += full_content
                    self.total_pages += 1
                    results.append(f"✅ {filename}: text file loaded")
                
                else:
                    results.append(f"❌ {filename}: unsupported format")
                    
            except Exception as e:
                results.append(f"❌ {filename}: error - {e}")
        
        # Set combined content for chat
        self.combined_content = combined_text
        
        summary = f"""
📊 **Processing Summary:**
- Total documents: {len(self.pdf_filenames)}
- Total pages: {self.total_pages}
- Total content length: {len(combined_text):,} characters

📋 **Results:**
{chr(10).join(results)}
        """
        
        return summary
    
    def create_context_prompt(self, query: str) -> str:
        """
        Create a context-aware prompt combining PDF content with user query
        
        Args:
            query: User's question
            
        Returns:
            Formatted prompt with context
        """
        context_prompt = f"""
You are an AI assistant that answers questions based on the provided document content from multiple PDF files.

Document Content:
{self.combined_content}

User Question: {query}

Instructions:
1. Answer the question based solely on the information provided in the documents above
2. When referencing information, specify which document it comes from
3. If the answer cannot be found in any document, clearly state that the information is not available
4. Provide specific quotes or references from the documents when possible
5. Be concise but comprehensive in your response
6. When identifying table of contents or page mappings, be very precise about page numbers and document names

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
        if not self.combined_content:
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
    
    def get_table_of_contents_json(self, mode: str = "combined") -> Optional[Dict]:
        """
        Extract table of contents and return as JSON with page mappings
        
        Args:
            mode: "combined" for all docs together, "separate" for individual docs
            
        Returns:
            Dictionary containing TOC with page numbers
        """
        if not self.combined_content:
            return None
        
        if mode == "combined":
            return self._get_combined_toc()
        else:
            return self._get_separate_tocs()
    
    def _get_combined_toc(self) -> Dict:
        """Get combined TOC for all documents"""
        toc_query = f"""
        Analyze all the provided documents and create a comprehensive table of contents that includes sections from all {len(self.pdf_filenames)} documents.
        
        Please return the response in the following JSON format:
        {{
            "analysis_type": "combined",
            "total_documents": {len(self.pdf_filenames)},
            "document_list": {json.dumps(self.pdf_filenames)},
            "total_pages": {self.total_pages},
            "combined_table_of_contents": [
                {{
                    "document_name": "document_name.pdf",
                    "section_number": "1",
                    "title": "Section Title",
                    "page_number": page_number,
                    "subsections": [
                        {{
                            "section_number": "1.1",
                            "title": "Subsection Title",
                            "page_number": page_number
                        }}
                    ]
                }}
            ]
        }}
        
        Make sure to identify sections from each document and specify which document each section belongs to.
        """
        
        try:
            response = self.chat(toc_query)
            return self._parse_toc_response(response, "combined")
        except Exception as e:
            return {"error": f"Error extracting combined TOC: {e}"}
    
    def _get_separate_tocs(self) -> Dict:
        """Get separate TOCs for each document"""
        all_tocs = {
            "analysis_type": "separate",
            "total_documents": len(self.pdf_filenames),
            "documents": {}
        }
        
        for filename, content_data in self.pdf_contents.items():
            toc_query = f"""
            Analyze only the document "{filename}" and extract its table of contents.
            
            Document content for {filename}:
            {content_data['content']}
            
            Please return a JSON response with the table of contents for this specific document:
            {{
                "document_name": "{filename}",
                "pages": {content_data['pages']},
                "table_of_contents": [
                    {{
                        "section_number": "1",
                        "title": "Section Title", 
                        "page_number": page_number,
                        "subsections": [
                            {{
                                "section_number": "1.1",
                                "title": "Subsection Title",
                                "page_number": page_number
                            }}
                        ]
                    }}
                ]
            }}
            """
            
            try:
                # Create temporary chatbot instance for individual document
                temp_content = self.combined_content
                self.combined_content = content_data['content']
                response = self.chat(toc_query)
                self.combined_content = temp_content  # Restore original content
                
                doc_toc = self._parse_toc_response(response, filename)
                all_tocs["documents"][filename] = doc_toc
                
            except Exception as e:
                all_tocs["documents"][filename] = {"error": f"Error extracting TOC: {e}"}
        
        all_tocs["extracted_at"] = datetime.now().isoformat()
        return all_tocs
    
    def _parse_toc_response(self, response: str, doc_identifier: str) -> Dict:
        """Parse TOC response and extract JSON"""
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                toc_data = json.loads(json_str)
                # Add metadata
                toc_data["extracted_at"] = datetime.now().isoformat()
                if doc_identifier == "combined":
                    toc_data["source_files"] = self.pdf_filenames
                else:
                    toc_data["source_file"] = doc_identifier
                return toc_data
            except json.JSONDecodeError:
                pass
        
        # If JSON extraction fails, create a structured response
        return {
            "document_identifier": doc_identifier,
            "total_pages": self.total_pages if doc_identifier == "combined" else self.pdf_contents.get(doc_identifier, {}).get('pages', 0),
            "extracted_at": datetime.now().isoformat(),
            "source_files": self.pdf_filenames if doc_identifier == "combined" else [doc_identifier],
            "raw_response": response,
            "note": "Unable to parse structured JSON, see raw_response for details"
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
        
        # File upload - allow multiple files
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF or text files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload one or more PDF or text files to chat with"
        )
        
        # Processing mode selection
        if uploaded_files:
            processing_mode = st.radio(
                "Processing Mode:",
                ["combined", "separate"],
                help="Combined: Treat all files as one document. Separate: Keep documents separate for individual analysis."
            )
        
        if uploaded_files and api_key:
            if st.button("Load Documents"):
                with st.spinner("Loading documents..."):
                    # Initialize chatbot
                    st.session_state.chatbot = PDFChatbot(api_key, model_name)
                    
                    # Load the files
                    result = st.session_state.chatbot.load_multiple_pdfs_from_bytes(
                        uploaded_files, 
                        processing_mode
                    )
                    st.success("Documents loaded successfully!")
                    st.markdown(result)
                    
                    # Clear previous chat history and TOC data
                    st.session_state.chat_history = []
                    st.session_state.toc_data = None
        
        st.divider()
        
        # Extract TOC button with mode selection
        if st.session_state.chatbot and st.session_state.chatbot.combined_content:
            st.subheader("Extract Table of Contents")
            
            toc_mode = st.radio(
                "TOC Extraction Mode:",
                ["combined", "separate"],
                help="Combined: Single TOC for all documents. Separate: Individual TOC for each document."
            )
            
            if st.button("Extract Table of Contents"):
                with st.spinner("Extracting table of contents..."):
                    st.session_state.toc_data = st.session_state.chatbot.get_table_of_contents_json(toc_mode)
                    st.success("Table of contents extracted!")
        
        # Download TOC as JSON
        if st.session_state.toc_data:
            json_str = json.dumps(st.session_state.toc_data, indent=2)
            
            # Create filename based on mode
            if st.session_state.toc_data.get("analysis_type") == "combined":
                filename_prefix = "combined_toc"
            else:
                filename_prefix = "separate_tocs"
            
            st.download_button(
                label="📥 Download TOC as JSON",
                data=json_str,
                file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Main content area
    if not api_key:
        st.warning("Please enter your Google AI API key in the sidebar to get started.")
        st.info("You can get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    if not uploaded_files:
        st.info("Please upload one or more PDF or text files in the sidebar to begin.")
        return
    
    if not st.session_state.chatbot:
        st.info("Please click 'Load Documents' in the sidebar to process your files.")
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chat with Your Documents")
        
        # Chat interface
        user_query = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main topics across all documents?"
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
            if st.button("📋 Get Documents Summary"):
                with st.spinner("Generating summary..."):
                    summary = st.session_state.chatbot.chat("Provide a comprehensive summary of all documents including main topics, key points, and structure for each document.")
                    st.session_state.chat_history.append({
                        "query": "Documents Summary",
                        "response": summary,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
        
        with col_b:
            if st.button("📑 Extract All TOCs"):
                with st.spinner("Extracting table of contents..."):
                    toc_response = st.session_state.chatbot.chat("Identify the table of contents from each document and produce a mapping of sections with PDF page numbers for each document.")
                    st.session_state.chat_history.append({
                        "query": "All Table of Contents",
                        "response": toc_response,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
        
        with col_c:
            if st.button("🔍 Compare Documents"):
                with st.spinner("Comparing documents..."):
                    comparison = st.session_state.chatbot.chat("Compare and contrast the main topics, themes, and content across all the uploaded documents. Highlight similarities and differences.")
                    st.session_state.chat_history.append({
                        "query": "Document Comparison",
                        "response": comparison,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
    
    with col2:
        st.header("Documents Info")
        
        if st.session_state.chatbot:
            st.write("**Total Documents:**", len(st.session_state.chatbot.pdf_filenames))
            st.write("**Total Pages:**", st.session_state.chatbot.total_pages)
            st.write("**Total Content Length:**", f"{len(st.session_state.chatbot.combined_content):,} characters")
            
            # Show individual document info
            if st.session_state.chatbot.pdf_filenames:
                with st.expander("📄 Individual Document Details"):
                    for filename in st.session_state.chatbot.pdf_filenames:
                        if filename in st.session_state.chatbot.pdf_contents:
                            doc_info = st.session_state.chatbot.pdf_contents[filename]
                            st.write(f"**{filename}:**")
                            st.write(f"  - Pages: {doc_info['pages']}")
                            st.write(f"  - Type: {doc_info['type'].upper()}")
                            st.write(f"  - Length: {len(doc_info['content']):,} chars")
                            st.write("---")
        
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