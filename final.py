import google.generativeai as genai
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path
import fitz  # PyMuPDF for PDF text extraction
from dotenv import load_dotenv
import re

load_dotenv()

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
        self.page_content_map = {}  # Maps actual PDF page number to content
        self.page_headings_map = {}  # Maps page number to list of headings with formatting
        self.page_text_blocks = {}  # Maps page number to text blocks with formatting
        self.toc_topics = []
        self.chat_history = []
        self.toc_end_page = None
        
    def load_pdf_from_file(self, file_path: str):
        """
        Load PDF text from a file and extract heading information with formatting
        
        Args:
            file_path: Path to the PDF file
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.suffix.lower() == '.pdf':
                raise ValueError(f"Only PDF files supported. Got: {file_path.suffix}")
            
            # Extract text from PDF using PyMuPDF with formatting info
            pdf_document = fitz.open(str(file_path))
            full_content = ""
            page_count = pdf_document.page_count
            self.page_content_map = {}
            self.page_headings_map = {}
            self.page_text_blocks = {}
            
            print(f"Processing {page_count} pages with formatting analysis...")
            
            for page_num in range(page_count):
                page = pdf_document.load_page(page_num)
                
                # Actual PDF page number (1-indexed)
                actual_page_num = page_num + 1
                
                # Get plain text
                page_text = page.get_text().strip()
                self.page_content_map[actual_page_num] = page_text
                
                # Get text with formatting information
                text_dict = page.get_text("dict")
                headings = self._extract_headings_from_page(text_dict, actual_page_num)
                text_blocks = self._extract_text_blocks_from_page(text_dict, actual_page_num)
                
                self.page_headings_map[actual_page_num] = headings
                self.page_text_blocks[actual_page_num] = text_blocks
                
                # Add to full content with clear page boundaries
                full_content += f"\n\n=== ACTUAL_PDF_PAGE_{actual_page_num} ===\n"
                full_content += page_text
                full_content += f"\n=== END_PAGE_{actual_page_num} ===\n"
                
                # Progress indicator
                if actual_page_num % 10 == 0:
                    print(f"Processed page {actual_page_num}/{page_count}")
            
            pdf_document.close()
            self.pdf_content = full_content
            
            # Find TOC end page
            self.toc_end_page = self._find_toc_end_page()
            
            print(f"PDF loaded successfully!")
            print(f"Total pages: {page_count}")
            print(f"Content length: {len(full_content)} characters")
            print(f"Headings extracted: {sum(len(h) for h in self.page_headings_map.values())}")
            if self.toc_end_page:
                print(f"Table of Contents ends at page: {self.toc_end_page}")
                
        except Exception as e:
            print(f"Error loading PDF file: {e}")
            self.pdf_content = ""
            self.page_content_map = {}
            self.page_headings_map = {}
            self.page_text_blocks = {}
    
    def _extract_headings_from_page(self, text_dict: dict, page_num: int) -> List[Dict]:
        """
        Extract potential headings from a page based on formatting characteristics
        
        Args:
            text_dict: PyMuPDF text dictionary
            page_num: Page number
            
        Returns:
            List of heading dictionaries with text, font size, and formatting info
        """
        headings = []
        
        try:
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        font_size = span.get("size", 0)
                        font_flags = span.get("flags", 0)
                        font_name = span.get("font", "")
                        
                        # Skip very short text or common words
                        if len(text) < 3 or text.lower() in ['the', 'and', 'or', 'in', 'on', 'at', 'to', 'for']:
                            continue
                        
                        # Identify potential headings based on formatting
                        is_bold = font_flags & 2**4  # Bold flag
                        is_italic = font_flags & 2**1  # Italic flag
                        is_large_font = font_size > 12  # Larger than typical body text
                        
                        # Check if it's likely a heading
                        is_heading = (
                            is_bold or 
                            is_large_font or
                            (font_size > 11 and len(text) < 100) or  # Short text with decent size
                            re.match(r'^\d+\.?\s', text) or  # Numbered heading
                            text.isupper() and len(text) < 50  # All caps short text
                        )
                        
                        # Additional heading patterns
                        heading_patterns = [
                            r'^(chapter|section|part)\s+\d+',
                            r'^\d+\.?\d*\.?\s+[A-Z]',
                            r'^[A-Z][A-Z\s]{2,}$',  # All caps
                            r'^[A-Z][a-z\s]+$'  # Title case
                        ]
                        
                        pattern_match = any(re.match(pattern, text, re.IGNORECASE) 
                                          for pattern in heading_patterns)
                        
                        if is_heading or pattern_match:
                            # Get position information
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            y_position = bbox[1] if bbox else 0
                            
                            headings.append({
                                'text': text,
                                'font_size': font_size,
                                'is_bold': is_bold,
                                'is_italic': is_italic,
                                'font_name': font_name,
                                'y_position': y_position,
                                'page': page_num,
                                'confidence': self._calculate_heading_confidence(text, font_size, is_bold, pattern_match)
                            })
        
        except Exception as e:
            print(f"Error extracting headings from page {page_num}: {e}")
        
        # Sort by confidence and y-position
        headings.sort(key=lambda x: (-x['confidence'], x['y_position']))
        return headings
    
    def _calculate_heading_confidence(self, text: str, font_size: float, is_bold: bool, pattern_match: bool) -> float:
        """
        Calculate confidence score for a text being a heading
        
        Returns:
            Confidence score (0-1)
        """
        score = 0.0
        
        # Font size contribution
        if font_size > 16:
            score += 0.4
        elif font_size > 14:
            score += 0.3
        elif font_size > 12:
            score += 0.2
        
        # Bold text
        if is_bold:
            score += 0.3
        
        # Pattern matching
        if pattern_match:
            score += 0.3
        
        # Text characteristics
        if len(text) < 100:  # Short text more likely to be heading
            score += 0.1
        
        if text[0].isupper():  # Starts with capital
            score += 0.1
        
        return min(score, 1.0)
    
    def _extract_text_blocks_from_page(self, text_dict: dict, page_num: int) -> List[Dict]:
        """
        Extract all text blocks with formatting information
        
        Args:
            text_dict: PyMuPDF text dictionary
            page_num: Page number
            
        Returns:
            List of text block dictionaries
        """
        text_blocks = []
        
        try:
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                
                block_text = ""
                avg_font_size = 0
                font_sizes = []
                
                for line in block["lines"]:
                    line_text = ""
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        font_size = span.get("size", 0)
                        line_text += text
                        if font_size > 0:
                            font_sizes.append(font_size)
                    
                    block_text += line_text + "\n"
                
                if font_sizes:
                    avg_font_size = sum(font_sizes) / len(font_sizes)
                
                if block_text.strip():
                    text_blocks.append({
                        'text': block_text.strip(),
                        'avg_font_size': avg_font_size,
                        'page': page_num
                    })
        
        except Exception as e:
            print(f"Error extracting text blocks from page {page_num}: {e}")
        
        return text_blocks
    
    def _find_toc_end_page(self) -> Optional[int]:
        """
        Find the page where Table of Contents ends by analyzing headings
        
        Returns:
            Page number where TOC ends, or None if not found
        """
        toc_keywords = [
            'table of contents',
            'contents',
            'index',
            'list of figures',
            'list of tables'
        ]
        
        # Look for TOC section
        for page_num in range(1, min(20, len(self.page_content_map) + 1)):
            if page_num not in self.page_content_map:
                continue
                
            page_content = self.page_content_map[page_num].lower()
            
            # Check if this page contains TOC
            is_toc_page = any(keyword in page_content for keyword in toc_keywords)
            
            if is_toc_page:
                # Look for the end of TOC by checking subsequent pages
                for next_page in range(page_num + 1, min(page_num + 10, len(self.page_content_map) + 1)):
                    if next_page not in self.page_headings_map:
                        continue
                    
                    # Check if we have significant headings (indicating content start)
                    headings = self.page_headings_map[next_page]
                    high_confidence_headings = [h for h in headings if h['confidence'] > 0.5]
                    
                    if high_confidence_headings:
                        # Check if headings look like content (not TOC entries)
                        content_headings = [h for h in high_confidence_headings 
                                          if not re.search(r'\d+$', h['text'].strip())]  # Not ending with page numbers
                        
                        if content_headings:
                            return next_page - 1
                
                return page_num + 2  # Default fallback
        
        return None
    
    def extract_toc_topics(self) -> List[Dict]:
        """
        Extract topics from TOC with their TOC page numbers and clean names
        
        Returns:
            List of dictionaries with topic info
        """
        if not self.pdf_content:
            return []
        
        try:
            toc_prompt = f"""
Analyze this PDF document and extract ALL topics/sections from the Table of Contents.

Document Content:
{self.pdf_content}

Instructions:
1. Find the Table of Contents section
2. Extract EVERY topic, chapter, section, and subsection listed
3. For each item, provide:
   - The exact topic name as it appears in TOC
   - The page number mentioned in TOC (if any)
4. Include main sections, subsections, appendices, references, etc.
5. Format each line as: "TOPIC_NAME | PAGE_NUMBER" (use | as separator)
6. If no page number is shown in TOC, use "TOPIC_NAME | NO_PAGE"

Example format:
Introduction | 1
Chapter 1: Overview | 5
1.1 Background | 7
1.2 Objectives | 9
Methodology | 15
Results and Discussion | 25
Conclusion | 45
References | 50
Appendix A | 55

Extract all TOC entries:
"""
            
            response = self.model.generate_content(toc_prompt)
            
            topics = []
            if response.text:
                lines = response.text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if '|' in line and len(line) > 3:
                        parts = line.split('|')
                        if len(parts) >= 2:
                            topic_name = parts[0].strip()
                            toc_page = parts[1].strip()
                            
                            # Clean topic name for better matching
                            clean_name = re.sub(r'^\d+\.?\d*\.?\s*', '', topic_name)
                            clean_name = re.sub(r'^(Chapter|Section|Part)\s+\d+:?\s*', '', clean_name, flags=re.IGNORECASE)
                            clean_name = clean_name.strip()
                            
                            if clean_name and len(clean_name) > 2:
                                topics.append({
                                    'name': topic_name,
                                    'toc_page': toc_page,
                                    'clean_name': clean_name
                                })
            
            self.toc_topics = topics
            print(f"Extracted {len(topics)} topics from TOC")
            return topics
            
        except Exception as e:
            print(f"Error extracting TOC: {e}")
            return []
    
    def find_topic_by_heading_format(self, topic_info: Dict) -> Optional[int]:
        """
        Find topic by matching it with actual headings based on formatting
        
        Args:
            topic_info: Dictionary with topic information
            
        Returns:
            First page number where topic appears as a heading after TOC, or None
        """
        topic_name = topic_info['name']
        clean_name = topic_info['clean_name']
        
        # Determine search start page
        start_page = 1
        if self.toc_end_page:
            start_page = self.toc_end_page + 1
        else:
            start_page = 11  # Fallback
        
        # Create multiple search variations
        search_terms = [
            topic_name,
            clean_name,
            topic_name.upper(),
            clean_name.upper(),
            topic_name.lower(),
            clean_name.lower(),
        ]
        
        # Add number variations (e.g., "1. Introduction", "Chapter 1 Introduction")
        for term in [topic_name, clean_name]:
            # Add common prefixes
            search_terms.extend([
                f"1. {term}",
                f"Chapter 1 {term}",
                f"Section 1 {term}",
                f"1 {term}",
                f"I. {term}",
                f"Part I {term}"
            ])
        
        # Search through headings starting from after TOC
        for page_num in sorted(self.page_headings_map.keys()):
            if page_num < start_page:
                continue
            
            headings = self.page_headings_map[page_num]
            
            # Sort headings by confidence
            headings.sort(key=lambda x: -x['confidence'])
            
            for heading in headings:
                heading_text = heading['text']
                
                # Only consider high-confidence headings
                if heading['confidence'] < 0.3:
                    continue
                
                # Try to match with search terms
                for search_term in search_terms:
                    if not search_term or len(search_term) < 3:
                        continue
                    
                    # Exact match
                    if search_term.lower() == heading_text.lower():
                        return page_num
                    
                    # Partial match with high confidence headings
                    if heading['confidence'] > 0.5:
                        # Check if search term is contained in heading
                        if search_term.lower() in heading_text.lower():
                            return page_num
                        
                        # Check if heading is contained in search term
                        if heading_text.lower() in search_term.lower():
                            return page_num
                    
                    # Fuzzy matching for high confidence headings
                    if heading['confidence'] > 0.6:
                        # Remove common words and check similarity
                        clean_heading = re.sub(r'\b(and|or|the|of|in|on|at|to|for|with|by|a|an)\b', '', heading_text, flags=re.IGNORECASE)
                        clean_search = re.sub(r'\b(and|or|the|of|in|on|at|to|for|with|by|a|an)\b', '', search_term, flags=re.IGNORECASE)
                        
                        clean_heading = re.sub(r'\s+', ' ', clean_heading).strip()
                        clean_search = re.sub(r'\s+', ' ', clean_search).strip()
                        
                        if clean_heading and clean_search:
                            if clean_search.lower() in clean_heading.lower() or clean_heading.lower() in clean_search.lower():
                                return page_num
        
        return None
    
    def get_complete_toc_mapping(self) -> str:
        """
        Get comprehensive mapping of TOC topics to their first heading appearance after TOC
        
        Returns:
            Detailed mapping report
        """
        if not self.pdf_content:
            return "No PDF content loaded. Please load a PDF file first."
        
        # Extract TOC topics
        topics = self.extract_toc_topics()
        
        if not topics:
            return "Could not find or extract Table of Contents from the PDF."
        
        print(f"\nSearching for HEADING-BASED occurrences after TOC of {len(topics)} topics...")
        print(f"Search starting from page: {self.toc_end_page + 1 if self.toc_end_page else 11}")
        
        results = []
        found_count = 0
        
        for i, topic_info in enumerate(topics, 1):
            topic_name = topic_info['name']
            toc_page = topic_info['toc_page']
            
            # Find topic by heading format
            heading_page = self.find_topic_by_heading_format(topic_info)
            
            if heading_page:
                # Get the actual heading that was matched
                matched_heading = None
                if heading_page in self.page_headings_map:
                    headings = self.page_headings_map[heading_page]
                    # Find the best matching heading
                    for heading in sorted(headings, key=lambda x: -x['confidence']):
                        if any(term.lower() in heading['text'].lower() or heading['text'].lower() in term.lower() 
                              for term in [topic_info['name'], topic_info['clean_name']] if term):
                            matched_heading = heading
                            break
                
                results.append({
                    'topic': topic_name,
                    'toc_page': toc_page,
                    'heading_page': heading_page,
                    'matched_heading': matched_heading,
                    'found': True
                })
                found_count += 1
                
                heading_info = f" (Heading: '{matched_heading['text']}', Size: {matched_heading['font_size']:.1f}, Bold: {matched_heading['is_bold']})" if matched_heading else ""
                print(f"  ✓ Found: {topic_name[:40]}... → Page {heading_page}{heading_info}")
            else:
                results.append({
                    'topic': topic_name,
                    'toc_page': toc_page,
                    'heading_page': None,
                    'matched_heading': None,
                    'found': False
                })
                print(f"  ✗ Missing: {topic_name[:40]}...")
            
            # Progress update
            if i % 5 == 0:
                print(f"    Progress: {i}/{len(topics)} topics processed")
        
        # Generate report
        report = f"""
TABLE OF CONTENTS → HEADING-BASED MAPPING
========================================

Summary:
- Total topics in TOC: {len(topics)}
- Successfully mapped to headings: {found_count}
- Not found as headings: {len(topics) - found_count}
- Success rate: {(found_count/len(topics)*100):.1f}%
- TOC ends at page: {self.toc_end_page or 'Unknown'}
- Search started from page: {self.toc_end_page + 1 if self.toc_end_page else 11}

DETAILED HEADING MAPPING:
========================

"""
        
        # Sort results by heading page number for found items
        found_results = [r for r in results if r['found']]
        not_found_results = [r for r in results if not r['found']]
        
        found_results.sort(key=lambda x: x['heading_page'])
        
        for result in found_results:
            toc_info = f" (TOC: {result['toc_page']})" if result['toc_page'] != 'NO_PAGE' else ""
            
            heading_details = ""
            if result['matched_heading']:
                h = result['matched_heading']
                heading_details = f"\n    → Matched Heading: '{h['text']}' (Size: {h['font_size']:.1f}, Bold: {h['is_bold']}, Confidence: {h['confidence']:.2f})"
            
            report += f"✓ {result['topic']} → HEADING ON PAGE {result['heading_page']}{toc_info}{heading_details}\n"
        
        if not_found_results:
            report += f"\nNOT FOUND AS HEADINGS AFTER TOC:\n"
            report += "=" * 35 + "\n"
            for result in not_found_results:
                toc_info = f" (TOC: {result['toc_page']})" if result['toc_page'] != 'NO_PAGE' else ""
                report += f"✗ {result['topic']}{toc_info}\n"
        
        return report
    
    def get_heading_analysis(self) -> str:
        """
        Get detailed analysis of headings found in the document
        
        Returns:
            Heading analysis report
        """
        if not self.page_headings_map:
            return "No headings data available"
        
        start_page = self.toc_end_page + 1 if self.toc_end_page else 11
        
        report = f"""
HEADING ANALYSIS (Pages {start_page}+)
===================================

"""
        
        total_headings = 0
        for page_num in sorted(self.page_headings_map.keys()):
            if page_num < start_page:
                continue
                
            headings = self.page_headings_map[page_num]
            if not headings:
                continue
            
            total_headings += len(headings)
            report += f"\nPage {page_num} ({len(headings)} headings):\n"
            report += "-" * 30 + "\n"
            
            for heading in sorted(headings, key=lambda x: -x['confidence']):
                report += f"  • '{heading['text'][:60]}...' " if len(heading['text']) > 60 else f"  • '{heading['text']}' "
                report += f"(Size: {heading['font_size']:.1f}, Bold: {heading['is_bold']}, Conf: {heading['confidence']:.2f})\n"
        
        report += f"\nTotal headings found after TOC: {total_headings}\n"
        return report
    
    def chat(self, query: str) -> str:
        """General chat function"""
        if not self.pdf_content:
            return "No PDF content loaded."
        
        try:
            prompt = f"""
Based on this PDF document content with page markers:

{self.pdf_content}

Question: {query}

Answer based on the document content, and mention actual PDF page numbers when relevant using the ACTUAL_PDF_PAGE_X markers.
"""
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {e}"
    
    def get_pdf_info(self) -> str:
        """Get PDF information"""
        if not self.page_content_map:
            return "No PDF loaded"
        
        total_headings = sum(len(headings) for headings in self.page_headings_map.values())
        toc_info = f", TOC ends at page {self.toc_end_page}" if self.toc_end_page else ""
        return f"PDF loaded: {len(self.page_content_map)} pages{toc_info}, {total_headings} headings extracted"


def main():
    """Main demo function"""
    API_KEY = os.getenv('GEMINI_API_KEY')
    
    if not API_KEY:
        print("ERROR: Please set GEMINI_API_KEY environment variable")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    chatbot = PDFChatbot(API_KEY)
    
    # Load PDF
    pdf_file = input("Press Enter if your PDF is already here. " \
    "If not, then enter path to your PDF file: ").strip()
    if not pdf_file:
        pdf_file = "Sample_PDF.pdf"  # Default file
    
    print(f"\nLoading PDF: {pdf_file}")
    chatbot.load_pdf_from_file(pdf_file)
    
    if not chatbot.pdf_content:
        print("Failed to load PDF. Please check the file path.")
        return
    
    print(f"\n{chatbot.get_pdf_info()}")
    
    print("\n" + "="*60)
    # print(" HEADING ANALYSIS")
    # print("="*60)
    
    # Show heading analysis first
    heading_analysis = chatbot.get_heading_analysis()
    # print(heading_analysis)
    
    # print("\n" + "="*60)
    print(" TOC TOPICS → HEADING-BASED MAPPING")
    print("="*60)
    
    # Get the complete mapping
    mapping = chatbot.get_complete_toc_mapping()
    print(mapping)
    
    # Interactive mode
    print("\n" + "="*60)
    print("Enter 'quit' to exit, or ask questions about the PDF:")
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == 'quit':
            break
        
        answer = chatbot.chat(question)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()