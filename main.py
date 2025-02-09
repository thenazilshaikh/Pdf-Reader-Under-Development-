import os
import cv2
import pytesseract
import numpy as np
from pdf_reader import extract_text_from_pdf
from qa_system import  answer_question  # Corrected typo here if needed
from pdf2image import convert_from_path
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.tree import Tree
from f_read.lang import LanguageProcessor
from transformers import pipeline
from f_read.data_apple import DataProcessor

# Ensure necessary resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Function to extract text from image-based PDFs using OCR
def extract_text_from_image_pdf(pdf_path):
    """Extract text from image-based PDF using OpenCV and Tesseract."""
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
        
        # Debug: Output OCR result for each page
        print(f"OCR Result for Page {i+1}:\n{page_text}\n{'-'*30}")
    
    return text

# Function to check and extract text from PDF
def check_and_extract_text(pdf_path):
    """Check PDF content and extract text or fall back to OCR."""
    # First, attempt to extract text from the PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Handle case where extracted text is a list
    if isinstance(extracted_text, list):
        extracted_text = " ".join([entry.get('text', '') for entry in extracted_text])  # Join list into a single string

    if not extracted_text.strip():  # If no text found, attempt OCR
        print("No text found in the PDF. Attempting OCR...")
        extracted_text = extract_text_from_image_pdf(pdf_path)
    
    # If after OCR, there's still no text, return an error
    if not extracted_text.strip():
        print("No valid text found in the PDF, even after OCR.")
    
    return extracted_text

def main():
    # Ask the user for the PDF file path
    pdf_path = input("Please enter the path to the PDF file: ").strip()
    
    # Create an instance of the LanguageProcessor to use its methods
    processor = LanguageProcessor()
    
    # Create an instance of DataProcessor to use its methods
    data_processor = DataProcessor()

    # Extract text from the PDF (with fallback to OCR)
    extracted_text = check_and_extract_text(pdf_path)
    
    if not extracted_text.strip():
        print("No valid text found in the PDF, even after OCR.")
        return
    
    print(f"Extracted Text (First 500 characters):\n{extracted_text[:500]}...")

    # Preprocess the extracted text
    cleaned_text = processor.preprocess_text(extracted_text)
    
    # POS tagging using the DataProcessor class
    pos_tags = data_processor.pos_tagging(cleaned_text)  # Call from DataProcessor
    print(f"POS Tags: {pos_tags}")  # Optional: To see the POS tags if needed
    
    # Named entity extraction (use raw extracted text, not pos_tags)
    named_entities = processor.extract_named_entities(extracted_text)  # Pass raw text for NER
    print(f"Named Entities: {named_entities}")
    
    # Chunk the extracted text using the chunking method from LanguageProcessor
    chunks = processor.chunk_text(extracted_text)
    print(f"Text Chunks: {chunks}")
    
    # QA model processing
    question = "What is the data?"
    nlp = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", framework="pt", device=-1)
    if extracted_text.strip():
        result = nlp({'question': question, 'context': extracted_text})
        print("Answer:", result['answer'])
    else:
        print("Error: No context found for the question.")
    
    # Generate a summary using the summarize_text method from LanguageProcessor
    summary = processor.summarize_text(extracted_text)
    print(f"Summary: {summary}")

if __name__ == "__main__":
    main()
