# 📄 PDF Chatbot using Google Gemini API

This project is a Python-based chatbot that leverages **Google Gemini (1.5)** to interact with the contents of a PDF document. A standout feature is its ability to **identify the Table of Contents (TOC)** from a PDF and **map each section to its corresponding PDF page number**. 

You can ask questions like:

> “Identify the entire table of contents from the given PDF and produce a mapping of section start and page numbers.”

Other capabilities include summarizing documents, answering context-based queries, and maintaining a chat history. The app runs in **demo mode** or **interactive terminal mode**.

---

## 🚀 Features

- ✅ **PDF Table of Contents (TOC) Extraction**  
  Automatically detects TOC entries and provides a mapping of section titles to their PDF page numbers.
- ✅ Extract text from `.pdf` or `.txt` files using PyMuPDF (`fitz`)
- ✅ Query the PDF content using Gemini's generative AI
- ✅ Summarize the loaded document
- ✅ Maintain chat history for reference
- ✅ Two modes: Demo and Interactive terminal-based chat


🔑 Setup
Get a Gemini API Key
Sign in at makersuite.google.com and copy your API key.

Create a .env file
In the root of your project, create a .env file with:
GEMINI_API_KEY=your_api_key_here

Place your PDF file
Save your target .pdf file in the project directory or provide the correct path to it when prompted.


🧠 How It Works
Loads and extracts content from PDF using PyMuPDF.

Sends a formatted prompt containing the document content and your query to the Gemini model.

Gemini responds contextually based only on the given PDF content.

Summarization and chat history features are built-in.

🧪 Demo Mode
To run a predefined test using a sample PDF:
python app.py

Select option 1 when prompted.


💬 Interactive Mode
For a conversational session:
python app.py

Select option 2 to:

Load a PDF file or raw text

Ask custom questions

Get document summary

View chat history

Exit anytime


Choose mode:
1. Demo with sample data
2. Interactive chat
Enter choice (1 or 2): 2

Enter your Google AI API key: ****

Load PDF content from (1) direct text input or (2) file? Enter 1 or 2: 2
Enter the path to your text file: Sample_PDF_2.pdf

Your question: Identify the entire table of contents from the given PDF and produce a mapping of section start and page numbers.


FILE STRUCTURE
pdf_project_v3/
│
├── app.py             # Main application file
├── .env               # Your API key (not committed to version control)
├── Sample_PDF_2.pdf   # Your PDF file
└── README.md          # Project documentation
