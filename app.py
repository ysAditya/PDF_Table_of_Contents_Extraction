import google.generativeai as genai
from typing import List, Dict
import os
from pathlib import Path
import fitz  # PyMuPDF for PDF text extraction
from dotenv import load_dotenv

load_dotenv()

class PDFChatbot:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the PDF Chatbot with Gemini 1.5
        
        Args:
            api_key: Your Google AI API key
            model_name: Gemini model to use (default: gemini-1.5-pro)
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.pdf_content = ""
        self.chat_history = []
        
    def load_pdf_text(self, pdf_text: str):
        """
        Load PDF text content into the chatbot
        
        Args:
            pdf_text: The extracted text content from PDF
        """
        self.pdf_content = pdf_text
        print(f"PDF content loaded successfully. Length: {len(pdf_text)} characters")
        
    def load_pdf_from_file(self, file_path: str):
        """
        Load PDF text from a file (supports both .pdf and .txt files)
        
        Args:
            file_path: Path to the PDF or text file
        """
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.pdf':
                # Extract text from PDF using PyMuPDF
                pdf_document = fitz.open(str(file_path))
                text_content = ""
                page_count = pdf_document.page_count
                
                for page_num in range(page_count):
                    page = pdf_document.load_page(page_num)
                    text_content += page.get_text()
                    text_content += "\n"  # Add newline between pages
                
                pdf_document.close()
                self.pdf_content = text_content
                print(f"PDF content extracted from: {file_path}")
                print(f"Number of pages: {page_count}")
                print(f"Content length: {len(text_content)} characters")
                
            elif file_path.suffix.lower() == '.txt':
                # Read text file directly
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.pdf_content = file.read()
                print(f"Text content loaded from: {file_path}")
                print(f"Content length: {len(self.pdf_content)} characters")
            
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: .pdf, .txt")
                
        except Exception as e:
            print(f"Error loading file: {e}")
            self.pdf_content = ""
    
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
            return "No PDF content loaded. Please load PDF content first using load_pdf_text() or load_pdf_from_file()."
        
        try:
            # Create context-aware prompt
            prompt = self.create_context_prompt(query)
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            # Store in chat history
            self.chat_history.append({
                "query": query,
                "response": response.text,
                "timestamp": __import__('datetime').datetime.now().isoformat()
            })
            
            return response.text
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def get_chat_history(self) -> List[Dict]:
        """
        Get the chat history
        
        Returns:
            List of chat interactions
        """
        return self.chat_history
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        print("Chat history cleared.")
    
    def get_pdf_summary(self) -> str:
        """
        Get a summary of the loaded PDF content
        
        Returns:
            Summary of the PDF content
        """
        if not self.pdf_content:
            return "No PDF content loaded."
        
        summary_prompt = f"""
Please provide a brief summary of the following document content:

{self.pdf_content[:5000]}  # First 5000 characters to avoid token limits

Provide:
1. Main topic/subject
2. Key sections or chapters
3. Important concepts mentioned
4. Document type/purpose

Summary:
"""
        
        try:
            response = self.model.generate_content(summary_prompt)
            return response.text
        except Exception as e:
            return f"Error generating summary: {e}"


# Example usage and demo
def main():
    # Initialize chatbot (you need to set your API key)
    API_KEY = os.getenv('GEMINI_API_KEY')  # Replace with your actual API key
    
    # You can get your API key from: https://makersuite.google.com/app/apikey
    chatbot = PDFChatbot(API_KEY)
    
    # Example 1: Load PDF text directly
    # sample_pdf_text = """
    # Chapter 1: Introduction to Machine Learning Pipeline
    
    # The first step in our pipeline is Data Collection, where we gather raw data from various sources.
    # This is followed by Data Preprocessing, which includes cleaning and transformation.
    # The third step is Feature Engineering, where we create meaningful features for our models.
    # Finally, we have Model Training and Evaluation phases.
    
    # Chapter 2: Data Collection Methods
    # Data collection involves several techniques including web scraping, API calls, and database queries.
    # The quality of data collection directly impacts the performance of downstream processes.
    # """
    
    # Comment out the sample text loading since we're using the PDF file
    # chatbot.load_pdf_text(sample_pdf_text)
    
    # Example 2: Load from your PDF file
    chatbot.load_pdf_from_file("Sample_PDF.pdf")
    
    print("=== PDF Chatbot Demo ===\n")
    
    # Demo queries
    queries = [
        "Identify the entire table of contents from the given pdf."
    ]
    
    for query in queries:
        print(f"Q: {query}")
        response = chatbot.chat(query)
        print(f"A: {response}\n")
        print("-" * 50 + "\n")
    
    # Get document summary
    print("=== Document Summary ===")
    summary = chatbot.get_pdf_summary()
    print(summary)


# Interactive chatbot function
def interactive_chat():
    """
    Run an interactive chat session
    """
    API_KEY = input("Enter your Google AI API key: ").strip()
    
    if not API_KEY:
        print("API key is required. Get one from: https://makersuite.google.com/app/apikey")
        return
    
    chatbot = PDFChatbot(API_KEY)
    
    # Load PDF content
    choice = input("Load PDF content from (1) direct text input or (2) file? Enter 1 or 2: ").strip()
    
    if choice == "1":
        print("Enter your PDF text content (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and len(lines) > 0 and lines[-1] == "":
                break
            lines.append(line)
        pdf_text = "\n".join(lines[:-1])  # Remove the last empty line
        chatbot.load_pdf_text(pdf_text)
    
    elif choice == "2":
        file_path = input("Enter the path to your text file: ").strip()
        chatbot.load_pdf_from_file(file_path)
    
    else:
        print("Invalid choice.")
        return
    
    print("\n=== Interactive PDF Chatbot ===")
    print("Type 'quit' to exit, 'summary' for document summary, 'history' for chat history")
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() == 'quit':
            break
        elif query.lower() == 'summary':
            print("\nDocument Summary:")
            print(chatbot.get_pdf_summary())
        elif query.lower() == 'history':
            print("\nChat History:")
            for i, chat in enumerate(chatbot.get_chat_history(), 1):
                print(f"{i}. Q: {chat['query']}")
                print(f"   A: {chat['response'][:100]}...")
        else:
            response = chatbot.chat(query)
            print(f"\nAnswer: {response}")


if __name__ == "__main__":
    # Run demo
    print("Choose mode:")
    print("1. Demo with sample data")
    print("2. Interactive chat")
    
    mode = input("Enter choice (1 or 2): ").strip()
    
    if mode == "1":
        main()
    elif mode == "2":
        interactive_chat()
    else:
        print("Invalid choice")