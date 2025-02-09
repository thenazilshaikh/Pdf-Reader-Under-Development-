import tkinter as tk
from tkinter import filedialog, messagebox
from summarizer import summarize_text, extract_text_from_pdf
from qa_system import answer_question, chunk_text  # Ensure chunk_text is implemented correctly
import os


# Global variable to store extracted text for question answering
full_text = ""

def select_file():
    """Open a file dialog to select a PDF file."""
    pdf_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if pdf_path:
        if not os.path.exists(pdf_path):
            messagebox.showerror("File Error", f"The file {pdf_path} does not exist.")
            return
        file_label.config(text=f"Selected file: {pdf_path}")  # Update the label to show the selected file
        print(f"Selected PDF: {pdf_path}")  # Debugging line to print the selected file path
        process_pdf(pdf_path)  # Automatically process the selected PDF
    else:
        messagebox.showerror("File Error", "No PDF file selected.")
        
def generate_summary():
    """Generate a summary of the extracted PDF text."""
    if full_text:
        try:
            print(full_text[:500])  # Debugging line to check the first 500 characters of full_text
            summary = summarize_text(full_text, char_limit=500)  # Generate a summary
            summary_label.config(text="Summary: " + summary)  # Update the label with the summary
        except Exception as e:
            messagebox.showerror("Summary Error", f"Error generating summary: {e}")
    else:
        messagebox.showerror("No PDF Loaded", "Please select a PDF to summarize.")

def clean_extracted_text(extracted_text):
    """Clean up the extracted text by removing unnecessary line breaks."""
    # Join all characters and remove any newlines between characters or words
    cleaned_text = ' '.join(extracted_text.splitlines())
    return cleaned_text

def process_pdf(pdf_path):
    """Extract and clean the text from the selected PDF."""
    print(f"Processing PDF: {pdf_path}")  # Debugging line to check if this function is triggered
    metadata = extract_text_from_pdf(pdf_path)  # Extract text from the selected PDF
    
    if not metadata:
        messagebox.showerror("PDF Error", "Failed to extract text from the PDF.")
        print("Failed to extract text: ", metadata)  # Print the metadata for debugging
        return

    # Clean up the extracted text
    extracted_text = clean_extracted_text("\n".join(metadata))  # Join and clean the text
    
    # Debugging: print the first 500 characters of the cleaned text
    print(f"Cleaned text (first 500 chars): {extracted_text[:500]}")

    # Display the cleaned text in the text box
    text_display.config(state=tk.NORMAL)  # Enable text display box to show text
    text_display.delete(1.0, tk.END)  # Clear previous extracted text
    text_display.insert(tk.END, extracted_text)  # Display cleaned text in the text box
    text_display.config(state=tk.DISABLED)  # Disable text box after showing the content

    # Store cleaned text for later use when answering questions
    global full_text
    full_text = extracted_text
    print(f"Full text updated: {full_text[:500]}")  # Debugging line to check if full_text is updated



def get_answer():
    """Get the answer to the question from the extracted text."""
    question = question_entry.get()  # Get the question from the user input
    if full_text and question:
        try:
            chunks = chunk_text(full_text)  # Split the text into chunks for better processing
            answers = []
            
            for chunk in chunks:
                answer = answer_question(chunk, question)  # Get answer from each chunk
                if answer:
                    answers.append(answer)
            
            if answers:
                result_label.config(text="Answer(s): " + ", ".join(answers))  # Display answers
            else:
                result_label.config(text="Answer not found.")  # Display if no answer found
        except Exception as e:
            result_label.config(text=f"Error getting answer: {e}")
    else:
        result_label.config(text="Please select a PDF and ask a valid question.")  # Error message if question is missing

# Initialize the main window
root = tk.Tk()
root.title("PDF Read by Nazil")

# Create and place GUI elements
select_button = tk.Button(root, text="Select PDF", command=select_file)  # Add Select PDF button
select_button.pack()

file_label = tk.Label(root, text="No file selected.")  # Display selected file path
file_label.pack()

question_label = tk.Label(root, text="Ask a question about the PDF:")
question_label.pack()

question_entry = tk.Entry(root, width=50)
question_entry.pack()

ask_button = tk.Button(root, text="Get Answer", command=get_answer)  # Add "Get Answer" button
ask_button.pack()

summary_button = tk.Button(root, text="Generate Summary", command=generate_summary)  # Add "Generate Summary" button
summary_button.pack()

# Create a text box to display the extracted text from the PDF
text_display_label = tk.Label(root, text="Extracted Text from PDF:")
text_display_label.pack()

text_display = tk.Text(root, height=10, width=50)
text_display.config(state=tk.DISABLED)  # Initially disable text display box
text_display.pack()

# Create a label for the summary
summary_label = tk.Label(root, text="Summary will be displayed here.", wraplength=400, justify="left")
summary_label.pack()

# Result label to display answers
result_label = tk.Label(root, text="Answer will be displayed here.", wraplength=400)
result_label.pack()

root.mainloop()
