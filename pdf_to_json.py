# import os
# import re
# import json
# from PyPDF2 import PdfReader
# import pdfplumber

# # Directory where PDF files are stored
# PDF_DIR = "Data"  # Replace with your directory containing PDF files
# OUTPUT_FILE = "Consolidated_Annual_Reports.json"

# def clean_text(text):
#     """
#     Cleans text by removing redundant data like page numbers, section headers, and excessive spaces.
#     Args:
#         text (str): Raw text extracted from PDF.
#     Returns:
#         str: Cleaned text.
#     """
#     text = re.sub(r"\bPage \d+\b", "", text, flags=re.IGNORECASE)  # Remove page numbers
#     text = re.sub(r"ITEM\s?\d+.*?:", "", text, flags=re.IGNORECASE)  # Remove ITEM sections
#     text = re.sub(r"PART\s?[IVXLCDM]+\s?:", "", text, flags=re.IGNORECASE)  # Remove PART headers
#     text = re.sub(r"\s+", " ", text)  # Normalize spaces
#     return text.strip()

# def extract_key_sections(text):
#     """
#     Extracts key sections like revenue, profit, and challenges from the text.
#     Args:
#         text (str): Cleaned text content.
#     Returns:
#         dict: Key sections with extracted information.
#     """
#     sections = {
#         "Revenue": None,
#         "Profit": None,
#         "Challenges": None,
#         "Future Plans": None
#     }

#     # Example patterns to extract information (adjust these patterns as needed)
#     # revenue_match = re.search(r"(revenue.*?€?\$?\d+[.,]?\d+\s?(billion|million)?)", text, re.IGNORECASE)
#     # profit_match = re.search(r"(profit.*?€?\$?\d+[.,]?\d+\s?(billion|million)?)", text, re.IGNORECASE)
#     revenue_match = re.search(r"(revenue|turnover).*?\d+[.,]?\d+\s?(billion|million)?", text, re.IGNORECASE)
#     profit_match = re.search(r"(profit|net income).*?\d+[.,]?\d+\s?(billion|million)?", text, re.IGNORECASE)
#     challenges_match = re.search(r"(challenges.*?\.)", text, re.IGNORECASE)
#     future_plans_match = re.search(r"(future plans.*?\.)", text, re.IGNORECASE)

#     if revenue_match:
#         sections["Revenue"] = revenue_match.group(1)
#     if profit_match:
#         sections["Profit"] = profit_match.group(1)
#     if challenges_match:
#         sections["Challenges"] = challenges_match.group(1)
#     if future_plans_match:
#         sections["Future Plans"] = future_plans_match.group(1)

#     return sections

# # def process_pdf(pdf_path):
# #     """
# #     Processes a single PDF file to extract cleaned and structured data.
# #     Args:
# #         pdf_path (str): Path to the PDF file.
# #     Returns:
# #         list: List of structured data for each page.
# #     """
# #     reader = PdfReader(pdf_path)
# #     structured_data = []

# #     for page in reader.pages:
# #         raw_text = page.extract_text()
# #         if raw_text:  # Skip empty pages
# #             cleaned_text = clean_text(raw_text)
# #             key_sections = extract_key_sections(cleaned_text)
# #             structured_data.append({
# #                 "cleaned_text": cleaned_text,
# #                 "key_sections": key_sections
# #             })

# #     return structured_data

# def process_pdf(pdf_path):
#     structured_data = []

#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             raw_text = page.extract_text()
#             tables = page.extract_tables()
            
#             if raw_text:
#                 cleaned_text = clean_text(raw_text)
#                 key_sections = extract_key_sections(cleaned_text)

#                 structured_data.append({
#                     "cleaned_text": cleaned_text,
#                     "key_sections": key_sections,
#                     "tables": tables if tables else None  # Store extracted tables
#                 })

#     return structured_data

# def consolidate_data(pdf_dir):
#     """
#     Processes all PDFs in a directory and consolidates the data into a single JSON file.
#     Args:
#         pdf_dir (str): Directory containing PDF files.
#     Returns:
#         list: Consolidated data for all PDFs.
#     """
#     consolidated_data = []

#     for root, _, files in os.walk(pdf_dir):
#         for file in files:
#             if file.endswith(".pdf"):
#                 pdf_path = os.path.join(root, file)
#                 company_name = file.split("_")[0]  # e.g., BMW, Ford
#                 year = re.search(r"\d{4}", file).group()  # Extract year from filename

#                 print(f"Processing {file}...")
#                 structured_data = process_pdf(pdf_path)

#                 # Append structured data to consolidated list
#                 consolidated_data.append({
#                     "company": company_name,
#                     "year": int(year),
#                     "file_name": file,
#                     "pages": structured_data
#                 })

#     return consolidated_data

# def save_consolidated_data(data, output_file):
#     """
#     Saves the consolidated data to a JSON file.
#     Args:
#         data (list): Consolidated data.
#         output_file (str): Path to the output JSON file.
#     """
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=4, ensure_ascii=False)

# # Main script execution
# if __name__ == "__main__":
#     consolidated_data = consolidate_data(PDF_DIR)
#     save_consolidated_data(consolidated_data, OUTPUT_FILE)
#     print(f"Consolidated data saved to {OUTPUT_FILE}")

##########################################################################################

# import os
# import re
# import json
# from PyPDF2 import PdfReader
# import pdfplumber

# # Directory where PDF files are stored
# PDF_DIR = "Data"  # Replace with your directory containing PDF files
# OUTPUT_FILE = "Consolidated_Annual_Reports.json"

# def clean_text(text):
#     """
#     Cleans text by removing redundant data like page numbers, section headers, and excessive spaces.
#     Args:
#         text (str): Raw text extracted from PDF.
#     Returns:
#         str: Cleaned text.
#     """
#     text = re.sub(r"\bPage \d+\b", "", text, flags=re.IGNORECASE)  # Remove page numbers
#     text = re.sub(r"ITEM\s?\d+.*?:", "", text, flags=re.IGNORECASE)  # Remove ITEM sections
#     text = re.sub(r"PART\s?[IVXLCDM]+\s?:", "", text, flags=re.IGNORECASE)  # Remove PART headers
#     text = re.sub(r"\s+", " ", text)  # Normalize spaces
#     return text.strip()

# def extract_key_sections(text):
#     """
#     Extracts key sections like revenue, profit, challenges, and other relevant sections from the text.
#     Args:
#         text (str): Cleaned text content.
#     Returns:
#         dict: Key sections with extracted information.
#     """
#     sections = {
#         "Revenue": None,
#         "Profit": None,
#         "Challenges": None,
#         "Future Plans": None,
#         "Economic Factors": None,
#         "Products in Development": None,
#         "Comparative Analysis": None,
#         "Growth Trends": None,
#         "Summary": None
#     }

#     # Example patterns to extract information
#     revenue_match = re.search(
#         r"(revenue|turnover|sales|income|earnings|proceeds|total revenue|net revenue|gross revenue)"
#         r".*?\d+[.,]?\d+\s?(billion|million|thousand)?",
#         text, re.IGNORECASE
#     )
#     profit_match = re.search(
#         r"(profit|net income|earnings|net profit|gross profit|operating profit|operating income|"
#         r"income before tax|income after tax|surplus|net earnings|total profit)"
#         r".*?\d+[.,]?\d+\s?(billion|million|thousand)?",
#         text, re.IGNORECASE
#     )
#     challenges_match = re.search(r"(challenges|obstacles|issues|difficulties).*?\.", text, re.IGNORECASE)
#     future_plans_match = re.search(r"(future plans|strategies|roadmap|objectives).*?\.", text, re.IGNORECASE)
#     economic_factors_match = re.search(r"(economic factors|market conditions|external factors).*?\.", text, re.IGNORECASE)
#     products_in_dev_match = re.search(r"(products in development|upcoming products|pipeline products|under development).*?\.", text, re.IGNORECASE)
#     comparative_analysis_match = re.search(r"(comparative analysis|comparison between|versus).*?\.", text, re.IGNORECASE)
#     growth_trends_match = re.search(r"(growth trends|performance trends|financial trends).*?\.", text, re.IGNORECASE)
#     summary_match = re.search(r"(summary|overview|recap|consolidated report).*?\.", text, re.IGNORECASE)

#     # Assign matched content to sections
#     if revenue_match:
#         sections["Revenue"] = revenue_match.group(0)
#     if profit_match:
#         sections["Profit"] = profit_match.group(0)
#     if challenges_match:
#         sections["Challenges"] = challenges_match.group(0)
#     if future_plans_match:
#         sections["Future Plans"] = future_plans_match.group(0)
#     if economic_factors_match:
#         sections["Economic Factors"] = economic_factors_match.group(0)
#     if products_in_dev_match:
#         sections["Products in Development"] = products_in_dev_match.group(0)
#     if comparative_analysis_match:
#         sections["Comparative Analysis"] = comparative_analysis_match.group(0)
#     if growth_trends_match:
#         sections["Growth Trends"] = growth_trends_match.group(0)
#     if summary_match:
#         sections["Summary"] = summary_match.group(0)

#     # Validate key sections and set to None if less than 3 words
#     for key, value in sections.items():
#         if value and len(value.split()) < 3:
#             sections[key] = None

#     return sections


# def process_pdf(pdf_path):
#     structured_data = []

#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             raw_text = page.extract_text()
#             tables = page.extract_tables()
            
#             if raw_text:
#                 cleaned_text = clean_text(raw_text)
#                 key_sections = extract_key_sections(cleaned_text)

#                 structured_data.append({
#                     "cleaned_text": cleaned_text,
#                     "key_sections": key_sections,
#                     "tables": tables if tables else None  # Store extracted tables
#                 })

#     return structured_data

# def consolidate_data(pdf_dir):
#     """
#     Processes all PDFs in a directory and consolidates the data into a single JSON file.
#     Args:
#         pdf_dir (str): Directory containing PDF files.
#     Returns:
#         list: Consolidated data for all PDFs.
#     """
#     consolidated_data = []

#     for root, _, files in os.walk(pdf_dir):
#         for file in files:
#             if file.endswith(".pdf"):
#                 pdf_path = os.path.join(root, file)
#                 company_name = file.split("_")[0]  # e.g., BMW, Ford
#                 year = re.search(r"\d{4}", file).group()  # Extract year from filename

#                 print(f"Processing {file}...")
#                 structured_data = process_pdf(pdf_path)

#                 # Append structured data to consolidated list
#                 consolidated_data.append({
#                     "company": company_name,
#                     "year": int(year),
#                     "file_name": file,
#                     "pages": structured_data
#                 })

#     return consolidated_data

# def save_consolidated_data(data, output_file):
#     """
#     Saves the consolidated data to a JSON file.
#     Args:
#         data (list): Consolidated data.
#         output_file (str): Path to the output JSON file.
#     """
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=4, ensure_ascii=False)

# # Main script execution
# if __name__ == "__main__":
#     consolidated_data = consolidate_data(PDF_DIR)
#     save_consolidated_data(consolidated_data, OUTPUT_FILE)
#     print(f"Consolidated data saved to {OUTPUT_FILE}")


######################################################################

# import os
# import re
# import json
# import pdfplumber

# # Directory where PDF files are stored
# PDF_DIR = "Data"  # Replace with your directory containing PDF files
# OUTPUT_FILE = "Consolidated_Annual_Reports.json"

# def clean_text(text):
#     """
#     Cleans text by removing redundant data like page numbers, section headers, and excessive spaces.
#     Args:
#         text (str): Raw text extracted from PDF.
#     Returns:
#         str: Cleaned text.
#     """
#     text = re.sub(r"\bPage \d+\b", "", text, flags=re.IGNORECASE)  # Remove page numbers
#     text = re.sub(r"ITEM\s?\d+.*?:", "", text, flags=re.IGNORECASE)  # Remove ITEM sections
#     text = re.sub(r"PART\s?[IVXLCDM]+\s?:", "", text, flags=re.IGNORECASE)  # Remove PART headers
#     text = re.sub(r"\s+", " ", text)  # Normalize spaces
#     return text.strip()

# def extract_key_sections(text):
#     """
#     Extracts key sections like revenue, profit, challenges, and other relevant sections from the text.
#     Args:
#         text (str): Cleaned text content.
#     Returns:
#         dict: Key sections with extracted information.
#     """
#     sections = {
#         "Revenue": None,
#         "Profit": None,
#         "Challenges": None,
#         "Future Plans": None,
#         "Economic Factors": None,
#         "Products in Development": None,
#         "Comparative Analysis": None,
#         "Growth Trends": None,
#         "Summary": None
#     }

#     # Example patterns to extract information
#     revenue_match = re.search(
#         r"(revenue|turnover|sales|income|earnings|proceeds|total revenue|net revenue|gross revenue)"
#         r".*?\d+[.,]?\d+\s?(billion|million|thousand)?",
#         text, re.IGNORECASE
#     )
#     profit_match = re.search(
#         r"(profit|net income|earnings|net profit|gross profit|operating profit|operating income|"
#         r"income before tax|income after tax|surplus|net earnings|total profit)"
#         r".*?\d+[.,]?\d+\s?(billion|million|thousand)?",
#         text, re.IGNORECASE
#     )
#     challenges_match = re.search(r"(challenges|obstacles|issues|difficulties).*?\.", text, re.IGNORECASE)
#     future_plans_match = re.search(r"(future plans|strategies|roadmap|objectives).*?\.", text, re.IGNORECASE)
#     economic_factors_match = re.search(r"(economic factors|market conditions|external factors).*?\.", text, re.IGNORECASE)
#     products_in_dev_match = re.search(r"(products in development|upcoming products|pipeline products|under development).*?\.", text, re.IGNORECASE)
#     comparative_analysis_match = re.search(r"(comparative analysis|comparison between|versus).*?\.", text, re.IGNORECASE)
#     growth_trends_match = re.search(r"(growth trends|performance trends|financial trends).*?\.", text, re.IGNORECASE)
#     summary_match = re.search(r"(summary|overview|recap|consolidated report).*?\.", text, re.IGNORECASE)

#     # Assign matched content to sections
#     if revenue_match:
#         sections["Revenue"] = revenue_match.group(0)
#     if profit_match:
#         sections["Profit"] = profit_match.group(0)
#     if challenges_match:
#         sections["Challenges"] = challenges_match.group(0)
#     if future_plans_match:
#         sections["Future Plans"] = future_plans_match.group(0)
#     if economic_factors_match:
#         sections["Economic Factors"] = economic_factors_match.group(0)
#     if products_in_dev_match:
#         sections["Products in Development"] = products_in_dev_match.group(0)
#     if comparative_analysis_match:
#         sections["Comparative Analysis"] = comparative_analysis_match.group(0)
#     if growth_trends_match:
#         sections["Growth Trends"] = growth_trends_match.group(0)
#     if summary_match:
#         sections["Summary"] = summary_match.group(0)

#     # Validate key sections and set to None if less than 3 words
#     for key, value in sections.items():
#         if value and len(value.split()) < 3:
#             sections[key] = None

#     return sections

# def process_pdf(pdf_path, company):
#     structured_data = []

#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             raw_text = page.extract_text()
#             tables = page.extract_tables()

#             if raw_text:
#                 cleaned_text = clean_text(raw_text)
#                 key_sections = extract_key_sections(cleaned_text)

#                 structured_data.append({
#                     "cleaned_text": cleaned_text,
#                     "key_sections": key_sections,
#                     "tables": tables if tables else None  # Store extracted tables
#                 })

#     return {"company": company, "data": structured_data}

# def consolidate_data(pdf_dir):
#     """
#     Processes all PDFs in a directory and consolidates the data into a single JSON file.
#     Args:
#         pdf_dir (str): Directory containing PDF files.
#     Returns:
#         dict: Consolidated data for BMW, Tesla, and Ford.
#     """
#     consolidated_data = {"BMW": [], "Tesla": [], "Ford": []}

#     for root, _, files in os.walk(pdf_dir):
#         for file in files:
#             if file.endswith(".pdf"):
#                 pdf_path = os.path.join(root, file)
#                 company_name = file.split("_")[0]  # e.g., BMW, Tesla, Ford

#                 print(f"Processing {file}...")
#                 structured_data = process_pdf(pdf_path, company_name)

#                 if company_name in consolidated_data:
#                     consolidated_data[company_name].append(structured_data)

#     return consolidated_data

# def save_consolidated_data(data, output_file):
#     """
#     Saves the consolidated data to a JSON file.
#     Args:
#         data (dict): Consolidated data.
#         output_file (str): Path to the output JSON file.
#     """
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=4, ensure_ascii=False)

# # Main script execution
# if __name__ == "__main__":
#     consolidated_data = consolidate_data(PDF_DIR)
#     save_consolidated_data(consolidated_data, OUTPUT_FILE)
#     print(f"Consolidated data saved to {OUTPUT_FILE}")


###################################################################
# import os
# import re
# import json
# import pdfplumber
# import openai

# # Set OpenAI API Key
# openai.api_key = "openai-api"
# print(f"Using OpenAI API Key: {openai.api_key[:5]}********")

# # Directory where PDF files are stored
# PDF_DIR = "Data"  # Replace with your directory containing PDF files
# OUTPUT_FILE = "Consolidated_Annual_Reports.json"

# def clean_text(text):
#     """
#     Cleans text by removing redundant data like page numbers, section headers, and excessive spaces.
#     Args:
#         text (str): Raw text extracted from PDF.
#     Returns:
#         str: Cleaned text.
#     """
#     text = re.sub(r"\bPage \d+\b", "", text, flags=re.IGNORECASE)  # Remove page numbers
#     text = re.sub(r"ITEM\s?\d+.*?:", "", text, flags=re.IGNORECASE)  # Remove ITEM sections
#     text = re.sub(r"PART\s?[IVXLCDM]+\s?:", "", text, flags=re.IGNORECASE)  # Remove PART headers
#     text = re.sub(r"\s+", " ", text)  # Normalize spaces
#     return text.strip()

# def extract_key_sections(text):
#     """
#     Extracts key sections like revenue, profit, challenges, and other relevant sections from the text.
#     Args:
#         text (str): Cleaned text content.
#     Returns:
#         dict: Key sections with extracted information.
#     """
#     sections = {
#         "Revenue": None,
#         "Profit": None,
#         "Challenges": None,
#         "Future Plans": None,
#         "Economic Factors": None,
#         "Products in Development": None,
#         "Comparative Analysis": None,
#         "Growth Trends": None,
#         "Summary": None
#     }

#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "Extract relevant financial and business insights from the following text."},
#                 {"role": "user", "content": text}
#             ]
#         )
#         if response and "choices" in response:
#             gpt_output = response["choices"][0]["message"]["content"]
#             sections = json.loads(gpt_output)
#         else:
#             print("No valid response from OpenAI API.")
#     except json.JSONDecodeError as e:
#         print(f"Error parsing GPT response: {e}")
#         print("Raw response:", response)
#     except Exception as e:
#         print(f"Unexpected error: {e}")

#     return sections

# def extract_table_data(tables):
#     """
#     Process table data into a structured format.
#     Args:
#         tables (list): List of tables extracted from the PDF page.
#     Returns:
#         list: Processed table data.
#     """
#     structured_tables = []
#     for table in tables:
#         structured_table = []
#         for row in table:
#             structured_table.append([cell.strip() if cell else "" for cell in row])
#         structured_tables.append(structured_table)
#     return structured_tables

# def process_pdf(pdf_path, company):
#     structured_data = []

#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             raw_text = page.extract_text()
#             tables = page.extract_tables()

#             page_data = {
#                 "cleaned_text": clean_text(raw_text) if raw_text else None,
#                 "key_sections": {},
#                 "tables": []
#             }

#             if raw_text:
#                 page_data["key_sections"] = extract_key_sections(page_data["cleaned_text"])

#             if tables:
#                 page_data["tables"] = extract_table_data(tables)

#             structured_data.append(page_data)

#     return {"company": company, "data": structured_data}

# def consolidate_data(pdf_dir):
#     """
#     Processes all PDFs in a directory and consolidates the data into a single JSON file.
#     Args:
#         pdf_dir (str): Directory containing PDF files.
#     Returns:
#         dict: Consolidated data for BMW, Tesla, and Ford.
#     """
#     consolidated_data = {"BMW": [], "Tesla": [], "Ford": []}

#     for root, _, files in os.walk(pdf_dir):
#         for file in files:
#             if file.endswith(".pdf"):
#                 pdf_path = os.path.join(root, file)
#                 company_name = file.split("_")[0]  # e.g., BMW, Tesla, Ford

#                 print(f"Processing {file}...")
#                 structured_data = process_pdf(pdf_path, company_name)

#                 if company_name in consolidated_data:
#                     consolidated_data[company_name].append(structured_data)

#     return consolidated_data

# def save_consolidated_data(data, output_file):
#     """
#     Saves the consolidated data to a JSON file.
#     Args:
#         data (dict): Consolidated data.
#         output_file (str): Path to the output JSON file.
#     """
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=4, ensure_ascii=False)

# # Main script execution
# if __name__ == "__main__":
#     consolidated_data = consolidate_data(PDF_DIR)
#     save_consolidated_data(consolidated_data, OUTPUT_FILE)
#     print(f"Consolidated data saved to {OUTPUT_FILE}")

