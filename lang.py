import fitz  # PyMuPDF
import nltk
import spacy
import re
import os
import string
import sys

nltk.download('stopwords')
sys.path.append('/Users/thenazilshaikh/Desktop/Pdf_read/f_read')
from nltk.corpus import words  # Importing the words corpus from nltk
from data_apple import DataProcessor
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



nltk_data_dir = os.path.expanduser('~/nltk_data')
os.environ['NLTK_DATA'] = nltk_data_dir

# Ensure NLTK resources are downloaded if not already present
def download_nltk_resources():
    resources = ['punkt', 'averaged_perceptron_tagger', 'stopwords', 'maxent_ne_chunker', 'words']

    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}")
            print(f"{resource} already downloaded.")
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, download_dir=nltk_data_dir)

# Download NLTK resources if they aren't already present
download_nltk_resources()

analyzer = SentimentIntensityAnalyzer()
text = "I love this product! It's amazing and works great."

# Get sentiment score
sentiment = analyzer.polarity_scores(text)

print(sentiment)

compound_score = sentiment['compound']
if compound_score >= 0.05:
    print("Positive sentiment")
elif compound_score <= -0.05:
    print("Negative sentiment")
else:
    print("Neutral sentiment")


nlp = spacy.load("en_core_web_sm")

# Predefined lists for various categories
dog_breeds = [
    "Labrador Retriever", "Poodle", "Golden Retriever", "Beagle", "Chihuahua", 
    "Shih Tzu", "Siberian Husky", "Bulldog", "Rottweiler", "German Shepherd",
    "Dachshund", "Boxer", "Cocker Spaniel", "Doberman", "French Bulldog",
    "Yorkshire Terrier", "Border Collie", "Dalmatian", "Shiba Inu", "Bichon Frise",
    "Pug", "Maltese", "Saint Bernard", "Great Dane", "Akita", "Pit Bull"
]

technologies = [
    "Artificial Intelligence", "Machine Learning", "Blockchain", "Quantum Computing", 
    "Cloud Computing", "5G", "IoT", "Cybersecurity", "Data Science", "Big Data", "Augmented Reality", 
    "Virtual Reality", "Neural Networks", "Deep Learning", "Natural Language Processing"
]

religions = [
    "Christianity", "Islam", "Hinduism", "Buddhism", "Judaism", "Sikhism", "Taoism", "Shinto", "Confucianism", 
    "Zoroastrianism", "Bahá'í Faith", "Jainism", "Shamanism"
]

countries = [
    "USA", "Canada", "India", "China", "Brazil", "Germany", "France", "Italy", "Australia", "Russia", 
    "UK", "Japan", "South Korea", "Mexico", "Argentina", "South Africa", "Egypt", "Saudi Arabia", "Japan"
]

class LanguageProcessor:
    
    def __init__(self):
        # Download necessary NLTK data if it's not already available
        try:
            nltk.data.find('corpora/words.zip')
        except LookupError:
            nltk.download('words')

        # Initialize the word list from NLTK corpus, alphabet, and an empty dictionary
        self.data_processor = DataProcessor() 
        self.nlp = spacy.load('en_core_web_sm')
        self.word_list = set(words.words())  # English dictionary from NLTK
        self.alphabet = list(string.ascii_lowercase)  # List of letters (a-z)
        self.dictionary = set()  # Dynamic dictionary for learned words
        self.dynamic_dict = set()
        self.sia = None  # This line ensures that we have a placeholder for SentimentIntensityAnalyzer

        # Initialize SentimentIntensityAnalyzer if it's not None
        if self.sia is None:
            self.sia = SentimentIntensityAnalyzer()
        
    def validate_and_add_word(self, word):
        """Validate if the word is in the dictionary and add it if necessary."""
        if word in self.dynamic_dict:
            print(f"'{word}' is already in the dynamic dictionary.")
        else:
            self.dynamic_dict.add(word)
            print(f"New word '{word}' added to the dictionary!")
        print("Dynamic Dictionary:", self.dynamic_dict)

    def analyze_sentiment(self, word):
        """Perform sentiment analysis on a word."""
        sentiment = self.sia.polarity_scores(word)
        
        if sentiment['compound'] >= 0.05:
            sentiment_label = 'Positive'
        elif sentiment['compound'] <= -0.05:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        
        return sentiment, sentiment_label

    def integrate_sentiment_with_word(self, word):
        """Integrate sentiment analysis with word checking and dictionary management."""
        sentiment, sentiment_label = self.analyze_sentiment(word)
        
        # Check if word is valid and add to dictionary if necessary
        self.validate_and_add_word(word)
        
        print(f"Sentiment analysis for '{word}': {sentiment_label} sentiment")
        print(f"Sentiment scores: {sentiment}")

    def add_to_dictionary(self, word):
        """Add a new word to the dynamic dictionary."""
        if word.lower() not in self.word_list:
            self.dictionary.add(word.lower())  # Store in lowercase for consistency
            print(f"Added '{word}' to dynamic dictionary.")
    
    def is_word_valid(self, word):
        """Check if the word is valid (in NLTK dictionary or dynamic dictionary)."""
        return word.lower() in self.word_list or word.lower() in self.dictionary
    
    def show_dictionary(self):
        """Display the current dynamic dictionary."""
        return self.dictionary

    def tokenize_characters(self, text):
        """
        Tokenize text into individual characters.
        """
        return list(text.lower())
    
    def construct_words_from_characters(self, characters):
        """
        Construct valid words from a sequence of characters.
        """
        word = ""
        words = []
        predictions = []
        
        for char in characters:
            if char in self.alphabet:  # Build word if it's an alphabetic character
                word += char
            else:  # Save word and reset on encountering non-alphabetic characters
                if word:
                    if word in self.word_list:
                        words.append(word)
                    else:
                        predictions += self.predict_word(word)
                    word = ""  # Reset the word
        
        # Add the last word if valid
        if word:
            if word in self.word_list:
                words.append(word)
            else:
                predictions += self.predict_word(word)
        
        return {"constructed_words": words, "predictions": predictions}
    
    def predict_word(self, prefix):
        """
        Predict possible completions for a given prefix.
        """
        return [word for word in self.word_list if word.startswith(prefix)]
    
    def learn_word(self, word):
        """
        Learn a new word and add it to the dynamic dictionary.
        """
        self.dictionary.add(word)
        print(f"New word '{word}' added to the dictionary!")
    
    def chunk_text(self, text, chunk_size=500):
        """
        Breaks the text into chunks of a specified size.
        """
        words = text.split()
        chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
        return [' '.join(chunk) for chunk in chunks]
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file using PyMuPDF (fitz).
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text("text")
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None
    
    def clean_text(self, text):
        """
        Cleans up broken words and unwanted spaces.
        """
        def add_spaces_between_words(text):
            text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Split lowercase-uppercase
            words_in_text = text.split()
            corrected_words = []
            for word in words_in_text:
                if word.lower() in self.word_list:
                    corrected_words.append(word)
                else:
                    corrected_word = re.sub(r'([a-z])([A-Z])', r'\1 \2', word)
                    corrected_words.append(corrected_word)
            return ' '.join(corrected_words)

        cleaned_text = add_spaces_between_words(text)
        return ' '.join(cleaned_text.split())
    
    def summarize_text(self, extracted_text, num_sentences=5):
        """Summarize text after preprocessing and POS tagging"""
        cleaned_text = self.preprocess_text_from_data_processing(extracted_text)
        sentences = nltk.sent_tokenize(cleaned_text)
        pos_tags = self.pos_tagging_from_data_processing(cleaned_text)
        # You can further process pos_tags or other NLP features here
        summary_sentences = sentences[:num_sentences]
        return ' '.join(summary_sentences)
    
    def extract_named_entities(self, text):
        """Extract named entities from the text using spaCy NER and custom category extraction."""
        doc = self.nlp(text)
        named_entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],
            "DATE": [],
            "MONEY": [],
            "PRODUCT": [],
            "NORP": [],
            "LOC": [],
            "FAC": [],
            "EVENT": [],
            "WORK_OF_ART": [],
            "LANGUAGE": [],
            "LAW": [],
            "DOG_BREED": [],
            "TECHNOLOGY": [],
            "RELIGION": [],
            "COUNTRY": [],
            "CUISINE": [],
            "FOOD": [],
            "CAPITAL": []
        }

        seen_entities = set()

        # Standard Named Entity Recognition (NER) from spaCy
        for ent in doc.ents:
            if ent.text not in seen_entities:
                seen_entities.add(ent.text)
                if ent.label_ == "PERSON" and ent.text not in dog_breeds:
                    named_entities["PERSON"].append(ent.text)
                elif ent.label_ == "ORG":
                    named_entities["ORG"].append(ent.text)
                elif ent.label_ == "GPE":
                    named_entities["GPE"].append(ent.text)
                elif ent.label_ == "DATE":
                    named_entities["DATE"].append(ent.text)
                elif ent.label_ == "MONEY":
                    named_entities["MONEY"].append(ent.text)
                elif ent.label_ == "PRODUCT":
                    named_entities["PRODUCT"].append(ent.text)
                elif ent.label_ == "NORP":
                    named_entities["NORP"].append(ent.text)
                elif ent.label_ == "LOC":
                    named_entities["LOC"].append(ent.text)
                elif ent.label_ == "FAC":
                    named_entities["FAC"].append(ent.text)
                elif ent.label_ == "EVENT":
                    named_entities["EVENT"].append(ent.text)
                elif ent.label_ == "WORK_OF_ART":
                    named_entities["WORK_OF_ART"].append(ent.text)
                elif ent.label_ == "LANGUAGE":
                    named_entities["LANGUAGE"].append(ent.text)
                elif ent.label_ == "LAW":
                    named_entities["LAW"].append(ent.text)

                # Detect categories like dog breeds, technology terms, and others
                if ent.text in dog_breeds:
                    named_entities["DOG_BREED"].append(ent.text)
                elif ent.text in technologies:
                    named_entities["TECHNOLOGY"].append(ent.text)
                elif ent.text in religions:
                    named_entities["RELIGION"].append(ent.text)
                elif ent.text in countries:
                    named_entities["COUNTRY"].append(ent.text)

        return named_entities
    
    def preprocess_text(self, text):
        # Example preprocessing: remove unwanted characters, stop words, etc.
        stop_words = set(stopwords.words('english'))
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

        # Remove stopwords
        words = text.split()
        text = ' '.join([word for word in words if word.lower() not in stop_words])

        return text.lower()  # Ensure you call lower() correctly
        
    def process_text(self, text):
        # Process the text with spaCy
        doc = self.nlp(text)
        return doc
    
    def preprocess_text_from_data_processing(self, text):
        """
        Preprocess the text as intended.
        """
        return self.data_processor.process_text(text)

  # Example usage section
if __name__ == "__main__":
    # Create an instance of the LanguageProcessor class
    processor = LanguageProcessor()

    # Add new words to the dynamic dictionary
    processor.add_to_dictionary("innovative")
    processor.add_to_dictionary("pythonic")
    processor.integrate_sentiment_with_word("hello")
    processor.integrate_sentiment_with_word("innovative")
    processor.integrate_sentiment_with_word("nonexistent")
    processor.integrate_sentiment_with_word("neologism")
    
    
    # Check word validity
    print(processor.is_word_valid("hello"))       # True, since "hello" is in the NLTK dictionary
    print(processor.is_word_valid("innovative"))  # True, since "innovative" was added to the dictionary
    print(processor.is_word_valid("nonexistent"))  # False, since "nonexistent" is not in either dictionary
    

    # Display the dynamic dictionary
    print("Dynamic Dictionary:", processor.show_dictionary())
    
    # Learn a new word and add it to the dictionary
    processor.learn_word("neologism")
    print("Dynamic Dictionary after learning a new word:", processor.show_dictionary())