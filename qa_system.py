from transformers import pipeline
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from f_read.lang import LanguageProcessor

# Sample text to process
text = "The Labrador Retriever is one of the most popular dog breeds in the United States."

# Define or import preprocess_text function here if not already defined
def preprocess_text(text):
    # Your text preprocessing steps here (e.g., lowercasing, removing stopwords, etc.)
    return text.lower()  # Example preprocessing

# Create an instance of LanguageProcessor
language_processor = LanguageProcessor()

# Process the text
processed_text = preprocess_text(text)  # Make sure to preprocess text first


# Initialize the question-answering pipeline
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad",
    framework="pt",
    device=-1  # Use -1 for CPU
)


def process_question(question, context):
    # Preprocess the context to clean it
    processed_text = preprocess_text(context)
    return processed_text

def chunk_text(text, chunk_size=1000):
    """Chunks the text into smaller parts to improve question answering."""
    # Split the text into chunks of `chunk_size` length or less
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

# qna.py
def filter_context_with_entities(context, named_entities):
    # Ensure 'context' is a string and split it into sentences if necessary
    sentences = context.split('.')
    filtered_context = [sentence for sentence in sentences if any(entity in sentence for entity in named_entities)]
    return ' '.join(filtered_context)


def answer_question(context, question):
    """Answer a question based on the provided context and adjust verb tense for proper grammar."""
    try:
        # Use QA pipeline to get the direct answer
        result = qa_pipeline({'context': context, 'question': question})
        answer = result['answer']

        # Detect if the question is asking for a conclusion or summary
        is_conclusion_request = any(keyword in question.lower() for keyword in ['conclusion', 'summary', 'overall', 'wrap up'])
        
        # Detect if the question is asking for an interpretive or reflective response
        is_reflection_request = any(keyword in question.lower() for keyword in ['author', 'lesson', 'message', 'meaning', 'portray', 'theme', 'moral'])

        # Detect if the question is asking about details or specific information
        is_detail_request = any(keyword in question.lower() for keyword in ['what', 'who', 'where', 'how', 'when', 'which'])

        # Detect if the question is asking about the reason or cause (why)
        is_reason_request = 'why' in question.lower()

        # Detect if the question is asking for a fact or opinion
        is_opinion_request = any(keyword in question.lower() for keyword in ['opinion', 'view', 'thought', 'think'])

        # Handle conclusion/summary request
        if is_conclusion_request:
            summary = summarize_text(context)  # You can implement your own summarizer
            return f"To conclude, here's the summary of the document: {summary}"

        # Handle reflective/interpretive question
        elif is_reflection_request:
            reflective_answer = "The author seems to be portraying themes of reflection and learning from the experiences shared in the document."  # You can adjust this
            return f"In my view, the lesson from the document is: {reflective_answer}"

        # Handle detail-based questions (like 'who', 'what', etc.)
        elif is_detail_request:
            return f"The specific information I found is: {answer}"

        # Handle questions asking 'why' (cause or reason)
        elif is_reason_request:
            reason_answer = "The reason seems to be based on the context provided, focusing on aspects like decision-making or outcomes."  # You can adjust this based on the context
            return f"The reason could be: {reason_answer}"

        # Handle opinion-based questions
        elif is_opinion_request:
            opinion_answer = "In my opinion, the document highlights various perspectives, encouraging readers to reflect on the information presented."
            return f"My opinion on this would be: {opinion_answer}"

        # Handle questions related to time or duration
        if 'time' in question.lower() or 'when' in question.lower() or 'period' in question.lower():
            time_answer = "The document covers events that span multiple periods or dates."
            return f"Regarding time, it seems that: {time_answer}"

        # Handle hypothetical or conditional questions
        elif 'if' in question.lower() or 'what if' in question.lower():
            hypothetical_answer = "If the situation were different, the response could change depending on additional context."
            return f"Considering the hypothetical situation, here's a possible response: {hypothetical_answer}"

        # Handle statistical or numeric questions
        elif 'how many' in question.lower() or 'number' in question.lower():
            numerical_answer = "The document references multiple breeds, but exact numbers can be determined from specific sections."
            return f"The number referenced is: {numerical_answer}"

        # Handle cause-effect questions (e.g., 'why did', 'because of')
        elif 'because' in question.lower() or 'due to' in question.lower():
            cause_effect_answer = "The outcome is likely influenced by the factors discussed in the document."
            return f"The reason behind this is: {cause_effect_answer}"

        # Handle process-related questions (e.g., 'how to', 'procedure')
        elif 'how to' in question.lower() or 'process' in question.lower():
            process_answer = "The steps or process involved are outlined in the document, typically starting with an overview."
            return f"The process described seems to be: {process_answer}"

        # Handle questions about location (e.g., 'where', 'in which country')
        elif 'where' in question.lower():
            location_answer = "The document mentions locations in a variety of contexts, but no specific location is emphasized."
            return f"Regarding location, the document suggests: {location_answer}"

        # Handle general questions about the document
        elif 'what is' in question.lower() or 'who is' in question.lower():
            general_answer = f"The answer to your query seems to be: {answer}"
            return general_answer

        # Handle 'compare' questions (e.g., 'compare the breeds', 'difference between')
        elif 'compare' in question.lower() or 'difference' in question.lower():
            comparison_answer = "The document contrasts several breeds, highlighting key differences in behavior and traits."
            return f"To compare, we can say: {comparison_answer}"

        # Handle historical questions (e.g., 'when did', 'history of')
        elif 'history' in question.lower() or 'when did' in question.lower():
            historical_answer = "The document reflects on events that span across historical periods, focusing on breed development over centuries."
            return f"Regarding history, it indicates that: {historical_answer}"

        # Handle future-oriented questions (e.g., 'what will happen', 'future of')
        elif 'future' in question.lower() or 'will' in question.lower():
            future_answer = "The future implications discussed suggest that trends in dog breeding may evolve over time."
            return f"Looking forward, it seems that: {future_answer}"

        # Handle reflective 'why do you think' questions
        elif 'why do you think' in question.lower():
            reflective_thought_answer = "I believe the author chose to focus on this topic to bring awareness to the diversity of dog breeds."
            return f"My thought on this would be: {reflective_thought_answer}"

        # Handle quantifiable results (e.g., 'how much', 'total')
        elif 'how much' in question.lower():
            quantity_answer = "The document provides various quantities related to dog breeds and their traits."
            return f"The total or quantity referred to is: {quantity_answer}"

        # Handle questions involving analysis (e.g., 'analyze', 'evaluate')
        elif 'analyze' in question.lower() or 'evaluate' in question.lower():
            analysis_answer = "The document provides detailed analysis on several dog breeds, comparing their characteristics and traits."
            return f"Analyzing the document, we can conclude: {analysis_answer}"

        # Handle verb tense adjustments for proper grammar:
        verb = "is"  # Default to present tense "is"
        
        # Check for past tense question indicators
        past_tense_keywords = ['did', 'was', 'were', 'had']
        if any(keyword in question.lower() for keyword in past_tense_keywords):
            verb = "was"  # Use past tense for past-related questions
        elif 'have' in question.lower() or 'has' in question.lower():
            verb = "has"  # Use present perfect tense

        # **50+ Styles**: Adding more styles to present the answers in different manners.
        styles = [
            f"Well, the answer {verb} {answer}",
            f"In my opinion, the answer to that would {verb} {answer}",
            f"Here's what I found: {answer}",
            f"After reading through the context, it seems the answer {verb} {answer}",
            f"To summarize, I believe the answer {verb} {answer}",
            f"In conclusion, the answer {verb} {answer}",
            f"Considering all points, we can say the answer {verb} {answer}",
            f"To wrap up, the answer is {answer}",
            f"Summing up the information, the answer is {answer}",
            f"Based on my analysis, I conclude that the answer {verb} {answer}",
            f"Reflecting on the document, it appears that {answer}",
            f"In short, I would say that the answer is {answer}",
            f"Given the context, we can infer that the answer is {answer}",
            f"Taking everything into account, it seems that the answer {verb} {answer}",
            f"Here’s my takeaway: {answer}",
            f"From the information I gathered, I believe the answer {verb} {answer}",
            f"As discussed earlier, the answer is {answer}",
            f"Based on the facts, it’s clear that {answer}",
            f"My conclusion from the document is that the answer {verb} {answer}",
            f"To finalize, the document points to {answer}",
            f"Considering all angles, the answer seems to be {answer}",
            f"Taking a holistic view, the answer {verb} {answer}",
            f"After thorough review, I can say the answer {verb} {answer}",
            f"The final answer is: {answer}"
        ]


    except Exception as e:
        print(f"Error answering question: {e}")
        return "I couldn't find an answer to that."

     
def process_pdf(pdf_path, question):
    """Process the PDF, extract relevant information, and answer the question."""
    # Here we assume that the text extraction part is handled in a separate file (like pdf_reader.py)
    from pdf_reader import extract_text_from_pdf
    from f_read.lang import LanguageProcessor  # Import the class

    # Create an instance of LanguageProcessor
    processor = LanguageProcessor()
    
    # Extract text from the PDF (assuming extract_text_from_pdf is defined in pdf_reader.py)
    text = extract_text_from_pdf(pdf_path)

    # Extract named entities using lang.py
    named_entities = processor.extract_named_entities(text)
    print("Named Entities found:", named_entities)  # Optional for debugging

    # Answer the user's question with the provided context (PDF text)
    answer = answer_question(text, question)
    
    return answer

# Example usage
if __name__ == "__main__":
    # Prompt the user to enter the PDF path
    pdf_path = input("Please enter the path to the PDF file: ")

    print("Feel free to ask a question related to the document.")
    
    # Take user input for the question
    question = input("Enter your question: ")
    
    try:
        # Extract text from the PDF (assuming the pdf_reader.extract_text_from_pdf function exists)
        from pdf_reader import extract_text_from_pdf
        text = extract_text_from_pdf(pdf_path)  # Extract text from the provided PDF
        
        # Preprocess the extracted text
        processed_text = preprocess_text(text)
        
        # Create an instance of LanguageProcessor
        language_processor = LanguageProcessor()
        
        # Example of using the LanguageProcessor methods
        named_entities = language_processor.extract_named_entities(processed_text)
        sentiment = language_processor.analyze_sentiment(processed_text)

        # Output the results from LanguageProcessor
        print("Named Entities:", named_entities)
        print("Sentiment:", sentiment)

        # Run question-answering on the document text using the qa_pipeline
        answer = qa_pipeline({"context": processed_text, "question": question})["answer"]

        if answer:
            print(f"Answer: {answer}")
        else:
            print("No answer found in the document.")
        
    except Exception as e:
        print("An error occurred:", e)