import sys
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re
import os

# Model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
offload_folder = './offload_folder'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, offload_folder=offload_folder)

def clean_text(text):
    # Remove headers, footers, page numbers, and common non-informative patterns
    text = re.sub(r'\n\d+\s*|\s*\d+\n', '', text)  # Page numbers
    text = re.sub(r'(?m)^\s*Page \d+\s*$', '', text)  # 'Page x' patterns
    text = text.strip()
    return text

# File loader and preprocessing
def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    final_text = ""
    for text in pages:
        cleaned_text = clean_text(text.page_content.strip())
        final_text += cleaned_text + "\n"
    return final_text

# LLM pipeline with sliding window for large documents
def llm_pipeline(file_path):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        min_length=100,  # Adjusted min length
        max_length=600  # Adjusted max length
    )
    input_text = file_preprocessing(file_path)
    print("Total Words in Input Text:")
    print(len(input_text.split()))

    # Define chunk parameters
    chunk_size = 10000  # Adjusted chunk size
    overlap_size = 7000  # Adjusted overlap size

    def summarize_text(text):
        chunks = []
        start_idx = 0

        # Create overlapping chunks
        while start_idx < len(text):
            end_idx = start_idx + chunk_size
            if end_idx >= len(text):
                chunk = text[start_idx:]
            else:
                chunk = text[start_idx:end_idx]
            chunks.append(chunk)
            start_idx += chunk_size - overlap_size

        # Perform summarization on each chunk
        print("Total chunks:", len(chunks))
        summaries = []
        for i, chunk in enumerate(chunks):
            print('Processing chunk', i)
            result = pipe_sum(chunk)
            summary_text = result[0]['summary_text']
            summaries.append(summary_text)
        
        return '\n'.join(summaries)

    # Single pass summarization
    final_summary = summarize_text(input_text)

    final_summary = summarize_text(final_summary)

    # Post-process to improve readability
    final_summary = final_summary.replace(' .', '.').replace(' ,', ',')
    final_summary_paragraphs = '\n\n'.join(final_summary.split('. '))

    return final_summary_paragraphs

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_pdf>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    try:
        # Perform summarization
        summary = llm_pipeline(file_path)
        print("Summarization Complete")

        # Save the summary to a text file
        output_file_path = file_path.replace('.pdf', '_summary.txt')
        with open(output_file_path, "w") as output_file:
            output_file.write(summary)
        print(f"Summary saved to {output_file_path}")

    except Exception as e:
        print(f"Error summarizing the document: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

