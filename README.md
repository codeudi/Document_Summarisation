# Executive Summary Generator using LaMini-Flan-T5-248M

This project generates an executive summary from a PDF document using the LaMini-Flan-T5-248M model.

## Prerequisites

Ensure you have the following installed:

1. **Python**: This project requires Python 3.8 or higher. You can download it from [python.org](https://www.python.org/downloads/).

2. **pip**: Python's package installer should be installed along with Python. You can check by running:
   ```sh
   pip --version

3. **Install dependencies**

pip install -r requirements.txt

4. **Running the Summarization Script**
Run the summarization script:
python script.py <path_to_pdf>
For example:
python script.py sample_document.pdf

This will generate a summary and save it to a text file with _summary appended to the original file name, in the data folder.