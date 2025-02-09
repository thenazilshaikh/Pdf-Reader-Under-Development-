import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.tree import Tree
import spacy

# Ensure necessary resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Initialize spaCy model for named entity recognition
nlp = spacy.load("en_core_web_sm")

class DataProcessor:
    def __init__(self):
        # Initialize any attributes if needed
        pass

    # Preprocess the text (tokenize and remove stopwords)
    def preprocess_text(self, text):
        """
        Preprocess the text by removing special characters, converting to lowercase, and removing stopwords.
        """
        # Remove special characters
        cleaned_text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        cleaned_text = cleaned_text.lower()  # Convert text to lowercase

        # Tokenize the text and remove stopwords
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(cleaned_text)
        filtered_words = [word for word in words if word not in stop_words]

        # Return the processed text (joined back to a single string)
        return ' '.join(filtered_words)
    
    
      # POS tagging (find parts of speech for each word in each sentence)
   
    def pos_tagging(self, text):
        # Tokenize the text into sentences first
        sentences = sent_tokenize(text)
        
        # Tokenize each sentence into words
        tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
        
        # Apply POS tagging on the tokenized sentences
        return [pos_tag(sentence) for sentence in tokenized_sentences]

         
         
    def extract_named_entities(self, tagged_sentences):
        named_entities = []
        for sentence in tagged_sentences:
            chunked = ne_chunk(sentence)
            named_entities.append([subtree for subtree in chunked if isinstance(subtree, Tree)])
        return named_entities

    # Using spaCy for named entity recognition (alternative)
    def extract_named_entities_spacy(self, text):
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    # Example function to clean text and extract entities
    def process_text(self, text):
        # Step 1: Preprocess the text
        preprocessed_text = self.preprocess_text(text)
        print("Preprocessed Text:", preprocessed_text)

        # Step 2: Tokenize the text into sentences
        sentences = nltk.sent_tokenize(preprocessed_text)
        print("Sentences:", sentences)

        # Step 3: POS tagging
        pos_tags = self.pos_tagging([word_tokenize(sentence) for sentence in sentences])
        print("POS Tagged Sentences:", pos_tags)

        # Step 4: Extract named entities using NLTK's Chunking
        named_entities_nltk = self.extract_named_entities(pos_tags)
        print("Named Entities (NLTK):", named_entities_nltk)

        # Step 5: Extract named entities using spaCy (alternative approach)
        named_entities_spacy = self.extract_named_entities_spacy(text)
        print("Named Entities (spaCy):", named_entities_spacy)

        # Return processed results
        return {
            "preprocessed_text": preprocessed_text,
            "sentences": sentences,
            "pos_tags": pos_tags,
            "named_entities_nltk": named_entities_nltk,
            "named_entities_spacy": named_entities_spacy
        }

# Example of running the functions
if __name__ == "__main__":

    text = "apple looking buying uk startup 1 billion steve jobs visionary"

    # Create an instance of DataProcessor
    processor = DataProcessor()

    # Call pos_tagging on the text
    pos_tags = processor.pos_tagging(text)

    # Output the result
    print(pos_tags)