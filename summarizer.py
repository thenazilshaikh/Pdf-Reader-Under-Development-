import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
import nltk
import re
from pdf2image import convert_from_path
from f_read.lang import LanguageProcessor
from collections import Counter
from nltk.corpus import stopwords

# Ensure necessary resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Set the Tesseract command path (Adjust this based on your system setup)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"  # Update path if needed

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    :param pdf_path: Path to the PDF file.
    :return: Extracted text as a string.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")  # Extracts the text from each page
        return text
    except Exception as e:
        print(f"Error reading PDF with PyMuPDF: {e}")
        return None

# Function to extract text from image-based PDFs using OCR
def extract_text_from_image_pdf(pdf_path):
    """
    Extracts text from an image-based PDF using Tesseract OCR.
    :param pdf_path: Path to the PDF file.
    :return: Extracted text as a string.
    """
    text = ""
    images = convert_from_path(pdf_path)
    
    for i, image in enumerate(images):
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Optional: Preprocess the image for better OCR results
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # OCR using pytesseract
        page_text = pytesseract.image_to_string(binary)
        text += f"\n--- Page {i + 1} ---\n" + page_text

    return text

# Function to clean up broken words or gaps in text
def clean_text(text):
    """
    Cleans up broken words and unwanted spaces.
    :param text: The raw extracted text.
    :return: Cleaned-up text.
    """
    # Remove unwanted space between words (e.g., "Ar tifical" -> "Artificial")
    cleaned_text = re.sub(r'(\w)\s+(\w)', r'\1 \2', text)
    
    # Remove any extra whitespace at the ends and between words
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text

# Function to highlight important terms (Named Entities, Frequent Words, etc.)
def highlight_important_terms(text):
    """
    Highlight important terms based on named entities, frequent words, and key phrases.
    :param text: The cleaned text from the document.
    :return: Highlighted text with important terms emphasized.
    """
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    
    # Perform Named Entity Recognition (NER)
    named_entities = nltk.chunk.ne_chunk(nltk.pos_tag(words))
    
    # Extract named entities (e.g., PERSON, ORGANIZATION)
    highlighted_text = text
    for chunk in named_entities:
        if isinstance(chunk, nltk.Tree):  # If it's a named entity
            for word, tag in chunk:
                # Ensure words are highlighted with spaces and word boundaries
                highlighted_text = re.sub(r'\b' + re.escape(word) + r'\b', f"**{word.upper()}**", highlighted_text)
    
    # Load stopwords to exclude common unimportant words
    stop_words = set(stopwords.words('english'))
    
    # Count the frequency of words in the document
    word_freq = Counter(words)
    
    # Highlight the most frequent words (excluding stopwords)
    most_common_words = [word for word, freq in word_freq.most_common(10) if word.lower() not in stop_words]
    
    for word in most_common_words:
        # Ensure words are highlighted with spaces and word boundaries
        highlighted_text = re.sub(r'\b' + re.escape(word) + r'\b', f"**{word.upper()}**", highlighted_text)
    
    return highlighted_text

# Function for a brief and direct summary with highlighted important information
def summarize_text(extracted_text, num_sentences=5):
    """
    Provides a concise summary of the extracted text as a single paragraph, highlighting important information.
    :param extracted_text: The extracted text from the PDF.
    :param num_sentences: Number of sentences to include in the summary.
    :return: A clear and connected summary of the content with important points highlighted.
    """
    # Clean the extracted text
    cleaned_text = clean_text(extracted_text)
    
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(cleaned_text)
    
    # Collect the first few sentences for the summary
    summary_sentences = sentences[:num_sentences]
    
    # Generate the summary by joining the sentences
    summary = "Summary of the document: "
    highlighted_summary = " ".join(summary_sentences)
    
    # Highlight important terms dynamically
    highlighted_summary = highlight_important_terms(highlighted_summary)
    
    summary += highlighted_summary
    
    return summary

# Main function to handle PDF processing
def main(pdf_path):
    # First, try extracting text using PyMuPDF
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # If no text is found, fallback to OCR (if the PDF is image-based)
    if not extracted_text or extracted_text.strip() == "":
        print("No text found in the PDF. Attempting OCR on the PDF...")
        extracted_text = extract_text_from_image_pdf(pdf_path)
    
    # Check if extracted text is empty even after OCR
    if not extracted_text or extracted_text.strip() == "":
        print("No valid text found in the PDF, even after OCR.")
        return
    
    # Generate a brief summary with highlighted information
    summarized_text = summarize_text(extracted_text, num_sentences=5)
    
    print("Summary of the document:")
    print(summarized_text)

# Example usage
if __name__ == "__main__":
    pdf_path = input("Enter the path to the PDF: ")  # Ask user for the path
    main(pdf_path)
