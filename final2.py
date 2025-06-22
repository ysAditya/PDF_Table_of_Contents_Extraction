import google.generativeai as genai
from typing import List, Dict, Tuple
import os
from pathlib import Path
import fitz  # PyMuPDF for PDF text extraction
from dotenv import load_dotenv
import re
from collections import defaultdict

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
        self.topic_page_mapping = {}
        self.toc_end_page = 0
        
    def extract_toc_topics(self, pdf_document) -> List[Dict]:
        """
        Extract topics from Table of Contents section
        
        Args:
            pdf_document: PyMuPDF document object
            
        Returns:
            List of topics with their TOC-listed page numbers
        """
        toc_topics = []
        toc_patterns = [
            # Common TOC patterns
            r'^(.+?)\s*\.{2,}\s*(\d+)\s*$',  # Topic...Page
            r'^(.+?)\s+(\d+)\s*$',  # Topic Page
            r'^(\d+\.?\d*)\s+(.+?)\s*\.{2,}\s*(\d+)\s*$',  # 1.1 Topic...Page
            r'^(\d+\.?\d*)\s+(.+?)\s+(\d+)\s*$',  # 1.1 Topic Page
            r'^(.+?)\s*-{2,}\s*(\d+)\s*$',  # Topic--Page
        ]
        
        # First, identify TOC pages
        toc_pages = []
        for page_num in range(min(10, pdf_document.page_count)):  # Check first 10 pages
            page = pdf_document.load_page(page_num)
            text = page.get_text().lower()
            
            # Look for TOC indicators
            if any(keyword in text for keyword in ['table of contents', 'contents', 'index']):
                toc_pages.append(page_num)
        
        # If no explicit TOC found, assume first few pages
        if not toc_pages:
            toc_pages = list(range(min(5, pdf_document.page_count)))
        
        # Extract topics from TOC pages
        for page_num in toc_pages:
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if len(line) < 3:
                    continue
                    
                # Try each pattern
                for pattern in toc_patterns:
                    match = re.match(pattern, line)
                    if match:
                        if len(match.groups()) == 2:
                            topic, page_ref = match.groups()
                            toc_topics.append({
                                'topic': topic.strip(),
                                'toc_page': int(page_ref),
                                'section_num': None
                            })
                        elif len(match.groups()) == 3:
                            section_num, topic, page_ref = match.groups()
                            toc_topics.append({
                                'topic': topic.strip(),
                                'toc_page': int(page_ref),
                                'section_num': section_num.strip()
                            })
        
        # Set TOC end page
        if toc_pages:
            self.toc_end_page = max(toc_pages) + 1
        
        return toc_topics
    
    def find_topic_actual_pages(self, pdf_document, toc_topics: List[Dict]) -> Dict[str, int]:
        """
        Find the actual starting page for each TOC topic in the document content
        
        Args:
            pdf_document: PyMuPDF document object
            toc_topics: List of topics from TOC
            
        Returns:
            Dictionary mapping topic names to their actual starting page numbers
        """
        topic_mapping = {}
        
        # Start searching from after TOC pages
        start_page = max(self.toc_end_page, 1)
        
        for topic_info in toc_topics:
            topic = topic_info['topic']
            toc_page = topic_info['toc_page']
            section_num = topic_info['section_num']
            
            # Clean topic for better matching
            cleaned_topic = self._clean_topic_for_search(topic)
            
            # Search for the topic starting from TOC end page
            found_page = self._search_topic_in_pages(pdf_document, cleaned_topic, start_page, section_num)
            
            if found_page:
                topic_mapping[topic] = found_page
            else:
                # Fallback: use TOC page reference if we can't find actual occurrence
                # But only if it's after TOC pages
                if toc_page > self.toc_end_page:
                    topic_mapping[topic] = toc_page
        
        return topic_mapping
    
    def _clean_topic_for_search(self, topic: str) -> str:
        """
        Clean topic text for better search matching
        """
        # Remove common prefixes/suffixes
        topic = re.sub(r'^(chapter|section|part)\s+\d+[:\s]*', '', topic, flags=re.IGNORECASE)
        topic = re.sub(r'\s*\.\.\.*\s*$', '', topic)  # Remove trailing dots
        topic = re.sub(r'\s+', ' ', topic.strip())
        return topic
    
    def _search_topic_in_pages(self, pdf_document, topic: str, start_page: int, section_num: str = None) -> int:
        """
        Search for topic occurrence in document pages
        
        Args:
            pdf_document: PyMuPDF document object
            topic: Topic to search for
            start_page: Page to start searching from
            section_num: Section number if available
            
        Returns:
            Page number where topic is found, or None
        """
        # Create search variations
        search_variations = [topic]
        
        if section_num:
            search_variations.extend([
                f"{section_num} {topic}",
                f"{section_num}. {topic}",
                f"Chapter {section_num} {topic}",
                f"Section {section_num} {topic}"
            ])
        
        # Also try with common prefixes
        search_variations.extend([
            f"Chapter {topic}",
            f"Section {topic}",
            topic.upper(),
            topic.title()
        ])
        
        # Search through pages starting from after TOC
        for page_num in range(start_page, pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            
            # Get text with formatting info to identify headings
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        is_heading = False
                        
                        for span in line["spans"]:
                            text = span["text"].strip()
                            # Check if this looks like a heading (bold, large font)
                            if span["flags"] & 16 or span["size"] >= 12:  # Bold or large font
                                is_heading = True
                            line_text += text + " "
                        
                        line_text = line_text.strip()
                        
                        # If this looks like a heading, check for topic match
                        if is_heading and len(line_text) > 3:
                            for variation in search_variations:
                                if self._topics_match(line_text, variation):
                                    return page_num + 1  # Return 1-based page number
            
            # Also check plain text for exact matches
            plain_text = page.get_text()
            for variation in search_variations:
                if self._find_topic_in_text(plain_text, variation):
                    return page_num + 1
        
        return None
    
    def _topics_match(self, text1: str, text2: str) -> bool:
        """
        Check if two topic strings match with fuzzy logic
        """
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Exact match
        if text1 == text2:
            return True
        
        # Check if one contains the other (with word boundaries)
        words1 = set(re.findall(r'\b\w+\b', text1))
        words2 = set(re.findall(r'\b\w+\b', text2))
        
        # If significant word overlap (>70% of smaller set)
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            min_words = min(len(words1), len(words2))
            if overlap / min_words > 0.7:
                return True
        
        return False
    
    def _find_topic_in_text(self, text: str, topic: str) -> bool:
        """
        Find topic in text using regex patterns
        """
        # Look for topic at beginning of lines (likely headings)
        pattern = r'^' + re.escape(topic) + r'\b'
        if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
            return True
        
        # Look for topic after numbers (section headings)
        pattern = r'^\d+\.?\s*' + re.escape(topic) + r'\b'
        if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
            return True
        
        return False
        
    def load_pdf_from_file(self, file_path: str):
        """
        Load PDF text from a file and extract topic-page mappings from TOC
        
        Args:
            file_path: Path to the PDF file
        """
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.pdf':
                pdf_document = fitz.open(str(file_path))
                
                print("Step 1: Extracting topics from Table of Contents...")
                toc_topics = self.extract_toc_topics(pdf_document)
                print(f"Found {len(toc_topics)} topics in TOC")
                
                print("Step 2: Finding actual starting pages for each topic...")
                self.topic_page_mapping = self.find_topic_actual_pages(pdf_document, toc_topics)
                
                # Extract full text content with page markers
                text_content = ""
                page_count = pdf_document.page_count
                
                for page_num in range(page_count):
                    page = pdf_document.load_page(page_num)
                    page_text = page.get_text()
                    text_content += f"\n--- PAGE {page_num + 1} ---\n"
                    text_content += page_text
                    text_content += "\n"
                
                pdf_document.close()
                self.pdf_content = text_content
                
                print(f"PDF processed: {file_path}")
                print(f"Total pages: {page_count}")
                print(f"TOC ends at page: {self.toc_end_page}")
                print(f"Topics mapped: {len(self.topic_page_mapping)}")
                
            elif file_path.suffix.lower() == '.txt':
                # Read text file directly
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.pdf_content = file.read()
                print(f"Text content loaded from: {file_path}")
            
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: .pdf, .txt")
                
        except Exception as e:
            print(f"Error loading file: {e}")
            self.pdf_content = ""
            self.topic_page_mapping = {}
    
    def get_topic_page_mapping(self) -> Dict[str, int]:
        """
        Get the mapping of topics to their actual starting page numbers
        """
        return self.topic_page_mapping
    
    def print_topic_mapping(self):
        """
        Print the topic-page mapping in a formatted way
        """
        if not self.topic_page_mapping:
            print("No topic mappings available. Please load a PDF file first.")
            return
        
        print(f"\n=== TOPIC MAPPING (Starting Pages After TOC Page {self.toc_end_page}) ===")
        print("-" * 70)
        
        # Sort by page number
        sorted_topics = sorted(self.topic_page_mapping.items(), key=lambda x: x[1])
        
        for topic, page_num in sorted_topics:
            print(f"Page {page_num:3d}: {topic}")
        
        print("-" * 70)
        print(f"Total topics mapped: {len(self.topic_page_mapping)}")
    
    def create_context_prompt(self, query: str) -> str:
        """
        Create a context-aware prompt combining PDF content with user query
        """
        # Include topic mapping in context if available
        topic_context = ""
        if self.topic_page_mapping:
            topic_context = f"\n\nTopic Starting Pages (Post-TOC):\n"
            sorted_topics = sorted(self.topic_page_mapping.items(), key=lambda x: x[1])
            for topic, page_num in sorted_topics:
                topic_context += f"Page {page_num}: {topic}\n"
        
        context_prompt = f"""
You are an AI assistant that answers questions based on the provided PDF document content.

PDF Document Content:
{self.pdf_content}
{topic_context}

User Question: {query}

Instructions:
1. Answer the question based solely on the information provided in the PDF document above
2. Use the topic starting pages mapping to provide accurate page references
3. Focus on where topics actually begin in the document content (not TOC page references)
4. Provide specific quotes or references from the document when possible
5. Be concise but comprehensive in your response

Answer:
"""
        return context_prompt
    
    def chat(self, query: str) -> str:
        """
        Chat with the PDF content using Gemini
        """
        if not self.pdf_content:
            return "No PDF content loaded. Please load PDF content first using load_pdf_from_file()."
        
        try:
            prompt = self.create_context_prompt(query)
            response = self.model.generate_content(prompt)
            
            self.chat_history.append({
                "query": query,
                "response": response.text,
                "timestamp": __import__('datetime').datetime.now().isoformat()
            })
            
            return response.text
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def get_chat_history(self) -> List[Dict]:
        """Get the chat history"""
        return self.chat_history
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        print("Chat history cleared.")


# Example usage and demo
def main():
    API_KEY = os.getenv('GEMINI_API_KEY')
    
    if not API_KEY:
        print("Please set your GEMINI_API_KEY environment variable")
        return
    
    chatbot = PDFChatbot(API_KEY)
    
    # Load PDF and extract TOC-based mappings
    chatbot.load_pdf_from_file("Sample_PDF_2.pdf")
    
    print("\n=== PDF TOC-Based Topic Mapping ===\n")
    
    # Print the topic mappings
    chatbot.print_topic_mapping()
    
    # Demo query
    query = "Show me all topics from the table of contents with their actual starting page numbers in the document."
    print(f"\nQ: {query}")
    response = chatbot.chat(query)
    print(f"A: {response}")


# Interactive chatbot function
def interactive_chat():
    """Run an interactive chat session"""
    API_KEY = input("Enter your Google AI API key: ").strip()
    
    if not API_KEY:
        print("API key is required. Get one from: https://makersuite.google.com/app/apikey")
        return
    
    chatbot = PDFChatbot(API_KEY)
    
    file_path = input("Enter the path to your PDF file: ").strip()
    chatbot.load_pdf_from_file(file_path)
    
    # Show topic mapping
    chatbot.print_topic_mapping()
    
    print("\n=== Interactive PDF Chatbot ===")
    print("Type 'quit' to exit, 'topics' to show topic mapping, 'history' for chat history")
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() == 'quit':
            break
        elif query.lower() == 'topics':
            chatbot.print_topic_mapping()
        elif query.lower() == 'history':
            print("\nChat History:")
            for i, chat in enumerate(chatbot.get_chat_history(), 1):
                print(f"{i}. Q: {chat['query']}")
                print(f"   A: {chat['response'][:100]}...")
        else:
            response = chatbot.chat(query)
            print(f"\nAnswer: {response}")


if __name__ == "__main__":
    print("Choose mode:")
    print("1. Extract TOC topic mappings and demo")
    print("2. Interactive chat")
    
    mode = input("Enter choice (1 or 2): ").strip()
    
    if mode == "1":
        main()
    elif mode == "2":
        interactive_chat()
    else:
        print("Invalid choice")