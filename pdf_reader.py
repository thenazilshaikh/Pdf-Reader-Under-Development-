import PyPDF2
import sys
import os
import cv2
import numpy as np
import pytesseract
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.tree import Tree
from f_read.lang import LanguageProcessor
from PIL import Image
from pdf2image import convert_from_path

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Ensure necessary resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Function to preprocess text (tokenization and stopword removal)
def preprocess_text(text):
    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    stop_words = set(stopwords.words('english'))
    cleaned_sentences = [
        [word for word in sentence if word.lower() not in stop_words]
        for sentence in tokenized_sentences
    ]
    return cleaned_sentences

# Function to perform POS tagging
def pos_tagging(sentences):
    pos_tagged_sentences = [pos_tag(sentence) for sentence in sentences]
    return pos_tagged_sentences

# Function to extract named entities (like people, locations)
def extract_named_entities(tagged_sentences):
    named_entities = []
    for sentence in tagged_sentences:
        chunked = ne_chunk(sentence)
        named_entities.append([subtree for subtree in chunked if isinstance(subtree, Tree)])
    return named_entities

# Function to extract text from PDF (regular PDFs)
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_number, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    print(f"Warning: No text found on page {page_number}")
        if not text:
            print("No text extracted from the PDF.")
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []

# Function to extract text from image-based PDF using Tesseract OCR
def extract_text_from_image_pdf(pdf_path):
    text = ""
    try:
        images = convert_from_path(pdf_path)  # Convert PDF pages to images
        for i, image in enumerate(images):
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert PIL to OpenCV image
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            page_text = pytesseract.image_to_string(binary)  # OCR processing
            text += f"\n--- Page {i + 1} ---\n" + page_text
        return text
    except Exception as e:
        print(f"Error extracting text from image-based PDF: {e}")
        return None

# After extracting the text, process it for grammar analysis using LanguageProcessor
def analyze_pdf_text(pdf_path):
    # Extract text
    text = extract_text_from_pdf(pdf_path) or extract_text_from_image_pdf(pdf_path)
    
    if text:
        # Initialize the LanguageProcessor
        processor = LanguageProcessor()

        # Process the extracted text to get grammar details
        processed_text = processor.process_text(text)
        
        # Print or use the processed information
        print("Processed Tokens and Grammar Terms:")
        print(processed_text)
        
        # Further analysis or actions can be performed here (e.g., QnA, summarization)
        return processed_text
    else:
        print("Failed to extract text from PDF.")
        return None
    
# Main function to process the PDF (regular or image-based)
def process_pdf(pdf_path):
    # First try regular text extraction
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # If no text found (e.g., scanned PDF), try image-based extraction
    if not extracted_text:
        print("Text extraction failed, attempting image-based OCR...")
        extracted_text = extract_text_from_image_pdf(pdf_path)
    
    if extracted_text:
        print("Preprocessing text...")
        preprocessed_text = preprocess_text(extracted_text)
        print("Preprocessed Text: ", preprocessed_text)
        
        print("POS Tagging...")
        pos_tags = pos_tagging(preprocessed_text)
        print("POS Tagged Sentences: ", pos_tags)
        
        print("Extracting Named Entities...")
        named_entities = extract_named_entities(pos_tags)
        print("Named Entities: ", named_entities)
        
        return extracted_text, preprocessed_text, pos_tags, named_entities
    else:
        print("No text was extracted.")
        return None, None, None, None

# Test the integrated flow with your PDF
if __name__ == "__main__":
    # Use dynamic file selection or ensure the file exists
    pdf_path = input("Enter the path to the PDF: ")
    process_pdf(pdf_path)
