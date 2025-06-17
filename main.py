import streamlit as st
import google.generativeai as genai
from typing import List, Dict, Optional
import os
import json
from pathlib import Path
import fitz  # PyMuPDF for PDF text extraction
from datetime import datetime
import io
import tempfile
from dotenv import load_dotenv

load_dotenv()

class MultiDocumentPDFChatbot:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the Multi-Document PDF Chatbot with Gemini 1.5
        
        Args:
            api_key: Your Google AI API key
            model_name: Gemini model to use (default: gemini-1.5-flash)
        """
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.documents = {}  # Dictionary to store multiple documents
            self.chat_history = []
            self.api_configured = True
        except Exception as e:
            st.error(f"Error configuring Gemini API: {e}")
            self.api_configured = False
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text from uploaded PDF file
        
        Args:
            pdf_file: Streamlit uploaded file object
            
        Returns:
            Extracted text content
        """
        try:
            # Create a temporary file to work with PyMuPDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Extract text using PyMuPDF
            pdf_document = fitz.open(tmp_file_path)
            text_content = ""
            page_count = pdf_document.page_count
            
            for page_num in range(page_count):
                page = pdf_document.load_page(page_num)
                text_content += f"[Page {page_num + 1}]\n"
                text_content += page.get_text()
                text_content += "\n\n"
            
            pdf_document.close()
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return text_content
            
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def add_document(self, doc_name: str, content: str, file_type: str = "pdf"):
        """
        Add a document to the chatbot's knowledge base
        
        Args:
            doc_name: Name identifier for the document
            content: Text content of the document
            file_type: Type of file (pdf, txt, etc.)
        """
        self.documents[doc_name] = {
            "content": content,
            "type": file_type,
            "added_at": datetime.now().isoformat(),
            "length": len(content)
        }
    
    def remove_document(self, doc_name: str):
        """Remove a document from the knowledge base"""
        if doc_name in self.documents:
            del self.documents[doc_name]
            return True
        return False
    
    def get_combined_content(self) -> str:
        """
        Get combined content from all loaded documents
        
        Returns:
            Combined text content with document separators
        """
        if not self.documents:
            return ""
        
        combined_content = ""
        for doc_name, doc_info in self.documents.items():
            combined_content += f"\n{'='*50}\n"
            combined_content += f"DOCUMENT: {doc_name}\n"
            combined_content += f"{'='*50}\n"
            combined_content += doc_info["content"]
            combined_content += f"\n{'='*50}\n"
        
        return combined_content
    
    def create_context_prompt(self, query: str, output_format: str = "text") -> str:
        """
        Create a context-aware prompt combining all document content with user query
        
        Args:
            query: User's question
            output_format: Desired output format ("text" or "json")
            
        Returns:
            Formatted prompt with context
        """
        combined_content = self.get_combined_content()
        
        format_instruction = ""
        if output_format.lower() == "json":
            format_instruction = """
Format your response as a valid JSON object with the following structure:
{
    "answer": "Your main answer here",
    "sources": ["List of document names that were referenced"],
    "confidence": "High/Medium/Low",
    "additional_info": "Any additional relevant information",
    "page_references": ["Specific page references if available"]
}
"""
        
        context_prompt = f"""
You are an AI assistant that answers questions based on the provided document content from multiple sources.

Document Content:
{combined_content}

User Question: {query}

Instructions:
1. Answer the question based solely on the information provided in the documents above
2. If the answer cannot be found in the documents, clearly state that the information is not available
3. When referencing information, mention which document it came from
4. Provide specific quotes or references when possible
5. Be concise but comprehensive in your response
{format_instruction}

Answer:
"""
        return context_prompt
    
    def chat(self, query: str, output_format: str = "text") -> Dict:
        """
        Chat with the document content using Gemini
        
        Args:
            query: User's question about the document content
            output_format: Output format ("text" or "json")
            
        Returns:
            Dictionary containing response and metadata
        """
        if not self.documents:
            return {
                "response": "No documents loaded. Please upload documents first.",
                "format": "text",
                "error": True
            }
        
        if not self.api_configured:
            return {
                "response": "API not configured properly. Please check your API key.",
                "format": "text", 
                "error": True
            }
        
        try:
            # Create context-aware prompt
            prompt = self.create_context_prompt(query, output_format)
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Parse JSON if requested
            parsed_response = response_text
            if output_format.lower() == "json":
                try:
                    parsed_response = json.loads(response_text)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract JSON from the response
                    try:
                        start_idx = response_text.find('{')
                        end_idx = response_text.rfind('}') + 1
                        if start_idx != -1 and end_idx > start_idx:
                            json_str = response_text[start_idx:end_idx]
                            parsed_response = json.loads(json_str)
                        else:
                            parsed_response = {
                                "answer": response_text,
                                "sources": list(self.documents.keys()),
                                "confidence": "Medium",
                                "additional_info": "Response was not in valid JSON format",
                                "page_references": []
                            }
                    except:
                        parsed_response = {
                            "answer": response_text,
                            "sources": list(self.documents.keys()),
                            "confidence": "Low",
                            "additional_info": "Failed to parse as JSON",
                            "page_references": []
                        }
            
            # Store in chat history
            chat_entry = {
                "query": query,
                "response": response_text,
                "parsed_response": parsed_response,
                "format": output_format,
                "timestamp": datetime.now().isoformat(),
                "documents_used": list(self.documents.keys())
            }
            self.chat_history.append(chat_entry)
            
            return {
                "response": parsed_response,
                "format": output_format,
                "error": False,
                "raw_response": response_text
            }
            
        except Exception as e:
            error_response = {
                "response": f"Error generating response: {e}",
                "format": "text",
                "error": True
            }
            return error_response
    
    def get_document_summary(self) -> str:
        """
        Get a summary of all loaded documents
        
        Returns:
            Summary of all documents
        """
        if not self.documents:
            return "No documents loaded."
        
        doc_list = ""
        for doc_name, doc_info in self.documents.items():
            doc_list += f"- {doc_name}: {doc_info['length']} characters, {doc_info['type']} file\n"
        
        combined_content = self.get_combined_content()
        
        summary_prompt = f"""
Please provide a comprehensive summary of the following documents:

{combined_content[:10000]}  # First 10000 characters to avoid token limits

For each document, provide:
1. Main topic/subject
2. Key sections or chapters
3. Important concepts mentioned
4. Document type/purpose

Documents loaded:
{doc_list}

Summary:
"""
        
        try:
            response = self.model.generate_content(summary_prompt)
            return response.text
        except Exception as e:
            return f"Error generating summary: {e}"


def main():
    st.set_page_config(
        page_title="Multi-Document PDF Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Multi-Document PDF Chatbot")
    st.markdown("Upload multiple PDF documents and ask questions with text or JSON output format")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google AI API Key",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        
        # Model selection
        model_name = st.selectbox(
            "Select Model",
            ["gemini-1.5-flash", "gemini-1.5-pro"],
            help="Choose the Gemini model to use"
        )
        
        # Initialize chatbot
        if api_key and st.button("Initialize Chatbot"):
            st.session_state.chatbot = MultiDocumentPDFChatbot(api_key, model_name)
            st.success("Chatbot initialized successfully!")
        
        st.divider()
        
        # Document management
        st.header("📄 Document Management")
        
        if st.session_state.chatbot:
            # Display loaded documents
            if st.session_state.chatbot.documents:
                st.subheader("Loaded Documents:")
                for doc_name, doc_info in st.session_state.chatbot.documents.items():
                    with st.expander(f"📄 {doc_name}"):
                        st.write(f"**Type:** {doc_info['type']}")
                        st.write(f"**Length:** {doc_info['length']:,} characters")
                        st.write(f"**Added:** {doc_info['added_at'][:19]}")
                        if st.button(f"Remove {doc_name}", key=f"remove_{doc_name}"):
                            st.session_state.chatbot.remove_document(doc_name)
                            st.rerun()
            else:
                st.info("No documents loaded yet")
        
        # Clear all documents
        if st.session_state.chatbot and st.session_state.chatbot.documents:
            if st.button("🗑️ Clear All Documents", type="secondary"):
                st.session_state.chatbot.documents.clear()
                st.success("All documents cleared!")
                st.rerun()
    
    # Main content area
    if not st.session_state.chatbot:
        st.warning("Please configure your API key in the sidebar to get started.")
        return
    
    # File upload section
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyze"
    )
    
    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Extract text from PDF
            text_content = st.session_state.chatbot.extract_text_from_pdf(uploaded_file)
            
            if text_content:
                # Add document to chatbot
                doc_name = uploaded_file.name
                st.session_state.chatbot.add_document(doc_name, text_content, "pdf")
                
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("✅ All documents processed successfully!")
        st.success(f"Loaded {len(uploaded_files)} document(s)")
    
    # Chat interface
    if st.session_state.chatbot.documents:
        st.header("💬 Chat with Documents")
        
        # Output format selection
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Ask a question about your documents:",
                placeholder="e.g., What are the main topics discussed in these documents?"
            )
        with col2:
            output_format = st.selectbox(
                "Output Format",
                ["text", "json"],
                help="Choose the response format"
            )
        
        if st.button("📤 Send Query", type="primary"):
            if query:
                with st.spinner("Generating response..."):
                    result = st.session_state.chatbot.chat(query, output_format)
                    
                    if not result["error"]:
                        st.success("Response generated!")
                        
                        # Display response based on format
                        if output_format == "json":
                            st.subheader("📋 JSON Response:")
                            st.json(result["response"])
                            
                            # Also show formatted view
                            if isinstance(result["response"], dict):
                                st.subheader("📖 Formatted View:")
                                if "answer" in result["response"]:
                                    st.write("**Answer:**", result["response"]["answer"])
                                if "sources" in result["response"]:
                                    st.write("**Sources:**", ", ".join(result["response"]["sources"]))
                                if "confidence" in result["response"]:
                                    st.write("**Confidence:**", result["response"]["confidence"])
                        else:
                            st.subheader("📝 Response:")
                            st.write(result["response"])
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "query": query,
                            "response": result["response"],
                            "format": output_format,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    else:
                        st.error(result["response"])
            else:
                st.warning("Please enter a question.")
        
        # Document Summary
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Generate Document Summary"):
                with st.spinner("Generating summary..."):
                    summary = st.session_state.chatbot.get_document_summary()
                    st.subheader("📋 Document Summary:")
                    st.write(summary)
        
        with col2:
            if st.session_state.chat_history and st.button("📥 Download Chat History"):
                # Create downloadable chat history
                chat_data = {
                    "chat_history": st.session_state.chat_history,
                    "documents": list(st.session_state.chatbot.documents.keys()),
                    "export_time": datetime.now().isoformat()
                }
                
                json_str = json.dumps(chat_data, indent=2)
                st.download_button(
                    label="💾 Download as JSON",
                    data=json_str,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Chat History Display
        if st.session_state.chat_history:
            st.header("💭 Chat History")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                with st.expander(f"💬 Query {len(st.session_state.chat_history) - i + 1}: {chat['query'][:50]}..."):
                    st.write(f"**Time:** {chat['timestamp']}")
                    st.write(f"**Format:** {chat['format']}")
                    st.write("**Query:**", chat['query'])
                    
                    if chat['format'] == 'json' and isinstance(chat['response'], dict):
                        st.json(chat['response'])
                    else:
                        st.write("**Response:**", chat['response'])
    
    else:
        st.info("👆 Please upload PDF documents to start chatting!")


if __name__ == "__main__":
    main()