# Assistants-API-Bot

Requirements
Python 3.6 or higher
Streamlit
OpenAI Python package
Requests library
BeautifulSoup library (for web scraping)
PDFKit library (for PDF conversion)
wkhtmltopdf (PDFKit dependency)
Setup Instructions
Install Dependencies:

pip install streamlit openai requests beautifulsoup4 pdfkit
API Key Configuration:

Securely input your OpenAI API key into the Streamlit app's sidebar to authenticate your API requests.

wkhtmltopdf Installation:

Ensure that wkhtmltopdf is installed and accessible in your system's PATH. This is necessary for PDF conversion with pdfkit.

Usage Guide
Run the Streamlit app:

streamlit run app.py
Input your OpenAI API key in the provided sidebar field.

Optionally, scrape web content and convert it to a PDF for the AI to use as context.

Upload any documents you want the AI to reference during the conversation.

Initiate the chat by clicking "Start Chat" and begin your conversation with the AI.

Application Notes
Always ensure you have the right to scrape content from websites and to use any uploaded documents.
The chat interface activates after the "Start Chat" button is clicked, and the uploaded files are processed.
