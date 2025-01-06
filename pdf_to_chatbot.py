# import os
# import json
# import pickle
# import numpy as np
# from flask import Flask, request, jsonify, render_template
# from openparse import Pdf, DocumentParser, processing
# from tqdm import tqdm
# import openai

# # Flask app setup
# app = Flask(__name__)

# # OpenAI API Key
# openai.api_key = "openai-api"
# # File to store processed data
# CACHE_FILE = "processed_docs.pkl"

# # Directory containing PDF files
# PDF_DIR = "./Data"  # Should contain subfolders like BMW, Tesla, Ford

# # Utility function for cosine similarity
# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# # Define a custom ingestion pipeline with necessary transformations
# class CustomIngestionPipeline(processing.IngestionPipeline):
#     def __init__(self):
#         self.transformations = [
#             processing.CombineNodesSpatially(x_error_margin=10, y_error_margin=2, criteria="both_small"),
#             processing.CombineHeadingsWithClosestText(),
#             processing.CombineBullets(),
#             processing.RemoveMetadataElements(),
#             processing.RemoveNodesBelowNTokens(min_tokens=10),
#         ]

# # Use the custom pipeline in the process_pdf function
# def process_pdf(file_path):
#     #doc = Pdf(file=file_path)
#     parser = DocumentParser(processing_pipeline=CustomIngestionPipeline())
#     parsed_content = parser.parse(file_path)

#     # Generate embeddings for nodes
#     embeddings = []
#     for idx, node in enumerate(parsed_content.nodes):
#         try:
#             print(f"Generating embedding for node {idx + 1}/{len(parsed_content.nodes)} in {file_path}")
#             embedding = openai.Embedding.create(input=node.text, model="text-embedding-ada-002")
#             embeddings.append(embedding['data'][0]['embedding'])
#         except Exception as e:
#             print(f"Error generating embedding for node {idx + 1} in {file_path}: {e}")
#             embeddings.append(None)  # Use None to indicate missing embedding

#     return parsed_content, embeddings


# # Load and preprocess PDF documents
# processed_docs = {"BMW": {}, "Tesla": {}, "Ford": {}}

# def load_or_process_documents():
#     global processed_docs
#     if os.path.exists(CACHE_FILE):
#         print("Loading processed documents from cache...")
#         with open(CACHE_FILE, "rb") as f:
#             processed_docs = pickle.load(f)
#     else:
#         print("Processing documents...")
#         for company in tqdm(processed_docs.keys(), desc="Processing companies"):
#             company_dir = os.path.join(PDF_DIR, company)
#             if os.path.exists(company_dir):
#                 for filename in tqdm(os.listdir(company_dir), desc=f"Processing {company} PDFs", leave=False):
#                     if filename.endswith(".pdf"):
#                         file_path = os.path.join(company_dir, filename)
#                         parsed_content, embeddings = process_pdf(file_path)
#                         processed_docs[company][filename] = {
#                             "content": parsed_content,
#                             "embeddings": embeddings,
#                         }
#         with open(CACHE_FILE, "wb") as f:
#             pickle.dump(processed_docs, f)
#         print("Processed documents saved to cache.")

# # Flask routes
# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/query", methods=["POST"])
# def query():
#     data = request.json
#     question = data.get("question", "")

#     if not question:
#         return jsonify({"error": "Question cannot be empty."}), 400

#     try:
#         # Generate embedding for the query
#         query_embedding = openai.Embedding.create(input=question, model="text-embedding-ada-002")['data'][0]['embedding']

#         # Find the most relevant document nodes
#         results = []
#         for company, docs in processed_docs.items():
#             for filename, doc in docs.items():
#                 if "embeddings" not in doc or not doc["embeddings"]:
#                     print(f"No embeddings found for {filename}. Skipping.")
#                     continue

#                 similarities = [
#                     cosine_similarity(query_embedding, emb) if emb is not None else 0
#                     for emb in doc["embeddings"]
#                 ]

#                 if not similarities or max(similarities) == 0:
#                     print(f"No valid similarities for {filename}. Skipping.")
#                     continue

#                 best_match_index = np.argmax(similarities)
#                 best_match_score = similarities[best_match_index]

#                 results.append({
#                     "company": company,
#                     "file": filename,
#                     "score": best_match_score,
#                     "content": doc["content"].nodes[best_match_index].text,
#                 })

#         # Sort results by score
#         results = sorted(results, key=lambda x: x["score"], reverse=True)[:5]

#         if not results:
#             return jsonify({
#                 "answer": "I could not find relevant content in the documents. However, you can try rephrasing the question.",
#                 "results": []
#             })

#         # Generate response using GPT
#         context = "\n".join([result["content"] for result in results])
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": (
#                         "You are an assistant that answers questions based on the provided context. "
#                         "If the context does not directly address the question, try to infer insights or provide "
#                         "related information to help the user."
#                     ),
#                 },
#                 {"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
#             ],
#         )

#         answer = response["choices"][0]["message"]["content"]

#         # Check for repetitive or generic responses and refine them
#         if "context does not provide" in answer.lower() or "does not include" in answer.lower():
#             answer = (
#                 "Based on the provided context, there are some general insight or advice related to your query:\n\n" + answer
#             )

#         return jsonify({"answer": answer, "results": results})

#     except Exception as e:
#         print(f"Error during query processing: {e}")
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     load_or_process_documents()
#     app.run(debug=True)

####### some improvement on the above code ################################

# import os
# import json
# import pickle
# import numpy as np
# from flask import Flask, request, jsonify, render_template
# from openparse import Pdf, DocumentParser, processing
# from tqdm import tqdm
# import openai

# # Flask app setup
# app = Flask(__name__)

# # OpenAI API Key
# openai.api_key = "openai-api"
# # File to store processed data
# CACHE_FILE = "processed_docs.pkl"

# # Directory containing PDF files
# PDF_DIR = "./Data"  # Should contain subfolders like BMW, Tesla, Ford

# # Utility function for cosine similarity
# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# # Define a custom ingestion pipeline with necessary transformations
# class CustomIngestionPipeline(processing.IngestionPipeline):
#     def __init__(self):
#         self.transformations = [
#             processing.CombineNodesSpatially(x_error_margin=10, y_error_margin=2, criteria="both_small"),
#             processing.CombineHeadingsWithClosestText(),
#             processing.CombineBullets(),
#             processing.RemoveMetadataElements(),
#             processing.RemoveNodesBelowNTokens(min_tokens=10),
#         ]

# # Use the custom pipeline in the process_pdf function
# def process_pdf(file_path):
#     parser = DocumentParser(processing_pipeline=CustomIngestionPipeline())
#     parsed_content = parser.parse(file_path)

#     # Generate embeddings for nodes
#     embeddings = []
#     for idx, node in enumerate(parsed_content.nodes):
#         try:
#             print(f"Generating embedding for node {idx + 1}/{len(parsed_content.nodes)} in {file_path}")
#             embedding = openai.Embedding.create(input=node.text, model="text-embedding-ada-002")
#             embeddings.append(embedding['data'][0]['embedding'])
#         except Exception as e:
#             print(f"Error generating embedding for node {idx + 1} in {file_path}: {e}")
#             embeddings.append(None)  # Use None to indicate missing embedding

#     return parsed_content, embeddings

# # Load and preprocess PDF documents
# processed_docs = {"BMW": {}, "Tesla": {}, "Ford": {}}

# def load_or_process_documents():
#     global processed_docs
#     if os.path.exists(CACHE_FILE):
#         print("Loading processed documents from cache...")
#         with open(CACHE_FILE, "rb") as f:
#             processed_docs = pickle.load(f)
#     else:
#         print("Processing documents...")
#         for company in tqdm(processed_docs.keys(), desc="Processing companies"):
#             company_dir = os.path.join(PDF_DIR, company)
#             if os.path.exists(company_dir):
#                 for filename in tqdm(os.listdir(company_dir), desc=f"Processing {company} PDFs", leave=False):
#                     if filename.endswith(".pdf"):
#                         file_path = os.path.join(company_dir, filename)
#                         parsed_content, embeddings = process_pdf(file_path)
#                         processed_docs[company][filename] = {
#                             "content": parsed_content,
#                             "embeddings": embeddings,
#                         }
#         with open(CACHE_FILE, "wb") as f:
#             pickle.dump(processed_docs, f)
#         print("Processed documents saved to cache.")

# # Flask routes
# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/query", methods=["POST"])
# def query():
#     data = request.json
#     question = data.get("question", "")

#     if not question:
#         return jsonify({"error": "Question cannot be empty."}), 400

#     try:
#         # Generate embedding for the query
#         query_embedding = openai.Embedding.create(input=question, model="text-embedding-ada-002")['data'][0]['embedding']

#         # Find the most relevant document nodes
#         results = []
#         for company, docs in processed_docs.items():
#             for filename, doc in docs.items():
#                 if "embeddings" not in doc or not doc["embeddings"]:
#                     continue

#                 similarities = [
#                     cosine_similarity(query_embedding, emb) if emb is not None else 0
#                     for emb in doc["embeddings"]
#                 ]

#                 if not similarities or max(similarities) == 0:
#                     continue

#                 best_match_index = np.argmax(similarities)
#                 best_match_score = similarities[best_match_index]

#                 results.append({
#                     "company": company,
#                     "file": filename,
#                     "score": best_match_score,
#                     "content": doc["content"].nodes[best_match_index].text,
#                 })

#         # Sort results by score
#         results = sorted(results, key=lambda x: x["score"], reverse=True)[:5]

#         if not results:
#             return jsonify({
#                 "answer": "Unfortunately, I couldn't locate specific information in the documents. However, feel free to ask a related question or rephrase your query.",
#                 "results": []
#             })

#         # Generate response using GPT
#         context = "\n".join([result["content"] for result in results])
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": (
#                         "You are an intelligent assistant. Use the provided context to directly answer the user's question. "
#                         "If the context lacks specific details, infer an answer or provide helpful insights. Avoid disclaimers."
#                     ),
#                 },
#                 {"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
#             ],
#         )

#         answer = response["choices"][0]["message"]["content"]

#         # Refine repetitive or generic responses
#         if "does not provide" in answer.lower() or "does not include" in answer.lower():
#             answer = (
#                 f"Although there is no explicit information in the context, here is an inferred response to your query:\n\n{answer}"
#             )

#         return jsonify({"answer": answer, "results": results})

#     except Exception as e:
#         print(f"Error during query processing: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     load_or_process_documents()
#     app.run(debug=True)



###########################################################

# import os
# import json
# import pickle
# import numpy as np
# from flask import Flask, request, jsonify, render_template
# from openparse import Pdf, DocumentParser, processing
# from tqdm import tqdm
# import openai
# import camelot  # For table parsing
# import pytesseract  # For OCR
# from PIL import Image  # For handling images

# # Flask app setup
# app = Flask(__name__)

# # OpenAI API Key
# openai.api_key = "openai-api"

# # File to store processed data
# CACHE_FILE = "processed_docs.pkl"

# # Directory containing PDF files
# PDF_DIR = "./Data"  # Should contain subfolders like BMW, Tesla, Ford

# # Utility function for cosine similarity
# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# # Define a custom ingestion pipeline with necessary transformations
# class CustomIngestionPipeline(processing.IngestionPipeline):
#     def __init__(self):
#         self.transformations = [
#             processing.CombineNodesSpatially(x_error_margin=10, y_error_margin=2, criteria="both_small"),
#             processing.CombineHeadingsWithClosestText(),
#             processing.CombineBullets(),
#             processing.RemoveMetadataElements(),
#             processing.RemoveNodesBelowNTokens(min_tokens=10),
#         ]

# # Extract tables from PDFs
# def extract_tables(file_path):
#     try:
#         tables = camelot.read_pdf(file_path, pages="all", flavor="stream")
#         extracted_tables = [table.df.to_dict(orient="records") for table in tables]
#         return extracted_tables
#     except Exception as e:
#         print(f"Error extracting tables from {file_path}: {e}")
#         return []

# # Extract text from images using OCR
# def extract_images_text(file_path):
#     try:
#         images = Pdf(file=file_path).extract_images()
#         extracted_text = []
#         for image in images:
#             text = pytesseract.image_to_string(Image.open(image))
#             extracted_text.append(text)
#         return "\n".join(extracted_text)
#     except Exception as e:
#         print(f"Error extracting text from images in {file_path}: {e}")
#         return ""

# # Use the custom pipeline in the process_pdf function
# def process_pdf(file_path):
#     parser = DocumentParser(processing_pipeline=CustomIngestionPipeline())
#     parsed_content = parser.parse(file_path)

#     # Extract tables
#     tables = extract_tables(file_path)

#     # Extract images text
#     images_text = extract_images_text(file_path)

#     # Generate embeddings for nodes
#     embeddings = []
#     for idx, node in enumerate(parsed_content.nodes):
#         try:
#             print(f"Generating embedding for node {idx + 1}/{len(parsed_content.nodes)} in {file_path}")
#             embedding = openai.Embedding.create(input=node.text, model="text-embedding-ada-002")
#             embeddings.append(embedding['data'][0]['embedding'])
#         except Exception as e:
#             print(f"Error generating embedding for node {idx + 1} in {file_path}: {e}")
#             embeddings.append(None)  # Use None to indicate missing embedding

#     return {
#         "content": parsed_content,
#         "tables": tables,
#         "images_text": images_text,
#         "embeddings": embeddings,
#     }

# # Load and preprocess PDF documents
# processed_docs = {"BMW": {}, "Tesla": {}, "Ford": {}}

# def load_or_process_documents():
#     global processed_docs
#     if os.path.exists(CACHE_FILE):
#         print("Loading processed documents from cache...")
#         with open(CACHE_FILE, "rb") as f:
#             processed_docs = pickle.load(f)
#     else:
#         print("Processing documents...")
#         for company in tqdm(processed_docs.keys(), desc="Processing companies"):
#             company_dir = os.path.join(PDF_DIR, company)
#             if os.path.exists(company_dir):
#                 for filename in tqdm(os.listdir(company_dir), desc=f"Processing {company} PDFs", leave=False):
#                     if filename.endswith(".pdf"):
#                         file_path = os.path.join(company_dir, filename)
#                         doc_data = process_pdf(file_path)
#                         processed_docs[company][filename] = doc_data
#         with open(CACHE_FILE, "wb") as f:
#             pickle.dump(processed_docs, f)
#         print("Processed documents saved to cache.")

# # Flask routes
# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/query", methods=["POST"])
# def query():
#     data = request.json
#     question = data.get("question", "")

#     if not question:
#         return jsonify({"error": "Question cannot be empty."}), 400

#     try:
#         # Generate embedding for the query
#         query_embedding = openai.Embedding.create(input=question, model="text-embedding-ada-002")['data'][0]['embedding']

#         # Find the most relevant document nodes
#         results = []
#         for company, docs in processed_docs.items():
#             for filename, doc in docs.items():
#                 if "embeddings" not in doc or not doc["embeddings"]:
#                     continue

#                 similarities = [
#                     cosine_similarity(query_embedding, emb) if emb is not None else 0
#                     for emb in doc["embeddings"]
#                 ]

#                 if not similarities or max(similarities) == 0:
#                     continue

#                 best_match_index = np.argmax(similarities)
#                 best_match_score = similarities[best_match_index]

#                 results.append({
#                     "company": company,
#                     "file": filename,
#                     "score": best_match_score,
#                     "content": doc["content"].nodes[best_match_index].text,
#                     "tables": doc["tables"],
#                     "images_text": doc["images_text"],
#                 })

#         # Sort results by score
#         results = sorted(results, key=lambda x: x["score"], reverse=True)[:5]

#         if not results:
#             return jsonify({
#                 "answer": "Unfortunately, I couldn't locate specific information in the documents. However, feel free to ask a related question or rephrase your query.",
#                 "results": []
#             })

#         # Generate response using GPT
#         context = "\n".join([result["content"] for result in results])
#         additional_context = "\n".join([json.dumps(result["tables"], indent=2) for result in results if result["tables"]])
#         context += "\n" + additional_context
#         context += "\n".join([result["images_text"] for result in results if result["images_text"]])

#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": (
#                         "You are an intelligent assistant. Use the provided context, including tables and images, to answer the user's question. "
#                         "If the context lacks specific details, infer an answer or provide helpful insights. Avoid disclaimers or saying the answer is not directly in context and try to provide an answer based the input."
#                     ),
#                 },
#                 {"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
#             ],
#         )

#         answer = response["choices"][0]["message"]["content"]

#         return jsonify({"answer": answer, "results": results})

#     except Exception as e:
#         print(f"Error during query processing: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     load_or_process_documents()
#     app.run(debug=True)


######################################
import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template, session
from openparse import Pdf, DocumentParser, processing
from tqdm import tqdm
import openai
import camelot  # For table parsing
import pytesseract  # For OCR
from PIL import Image  # For handling images
import secrets  # For secure secret key generation

# Flask app setup
app = Flask(__name__)

# Generate a secure secret key
app.secret_key = secrets.token_hex(24)

# OpenAI API Key
openai.api_key = "openai-api"

# File to store processed data
CACHE_FILE = "processed_docs.pkl"

# Directory containing PDF files
PDF_DIR = "./Data"  # Should contain subfolders like BMW, Tesla, Ford

# Utility function for cosine similarity
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Define a custom ingestion pipeline with necessary transformations
class CustomIngestionPipeline(processing.IngestionPipeline):
    def __init__(self):
        self.transformations = [
            processing.CombineNodesSpatially(x_error_margin=10, y_error_margin=2, criteria="both_small"),
            processing.CombineHeadingsWithClosestText(),
            processing.CombineBullets(),
            processing.RemoveMetadataElements(),
            processing.RemoveNodesBelowNTokens(min_tokens=10),
        ]

# Extract tables from PDFs
def extract_tables(file_path):
    try:
        tables = camelot.read_pdf(file_path, pages="all", flavor="stream")
        extracted_tables = [table.df.to_dict(orient="records") for table in tables]
        return extracted_tables
    except Exception as e:
        print(f"Error extracting tables from {file_path}: {e}")
        return []

# Extract text from images using OCR
def extract_images_text(file_path):
    try:
        images = Pdf(file=file_path).extract_images()
        extracted_text = []
        for image in images:
            text = pytesseract.image_to_string(Image.open(image))
            extracted_text.append(text)
        return "\n".join(extracted_text)
    except Exception as e:
        print(f"Error extracting text from images in {file_path}: {e}")
        return ""

# Use the custom pipeline in the process_pdf function
def process_pdf(file_path):
    parser = DocumentParser(processing_pipeline=CustomIngestionPipeline())
    parsed_content = parser.parse(file_path)

    # Extract tables
    tables = extract_tables(file_path)

    # Extract images text
    images_text = extract_images_text(file_path)

    # Generate embeddings for nodes
    embeddings = []
    for idx, node in enumerate(parsed_content.nodes):
        try:
            print(f"Generating embedding for node {idx + 1}/{len(parsed_content.nodes)} in {file_path}")
            embedding = openai.Embedding.create(input=node.text, model="text-embedding-ada-002")
            embeddings.append(embedding['data'][0]['embedding'])
        except Exception as e:
            print(f"Error generating embedding for node {idx + 1} in {file_path}: {e}")
            embeddings.append(None)  # Use None to indicate missing embedding

    return {
        "content": parsed_content,
        "tables": tables,
        "images_text": images_text,
        "embeddings": embeddings,
    }

# Load and preprocess PDF documents
processed_docs = {"BMW": {}, "Tesla": {}, "Ford": {}}

def load_or_process_documents():
    global processed_docs
    if os.path.exists(CACHE_FILE):
        print("Loading processed documents from cache...")
        with open(CACHE_FILE, "rb") as f:
            processed_docs = pickle.load(f)
    else:
        print("Processing documents...")
        for company in tqdm(processed_docs.keys(), desc="Processing companies"):
            company_dir = os.path.join(PDF_DIR, company)
            if os.path.exists(company_dir):
                for filename in tqdm(os.listdir(company_dir), desc=f"Processing {company} PDFs", leave=False):
                    if filename.endswith(".pdf"):
                        file_path = os.path.join(company_dir, filename)
                        doc_data = process_pdf(file_path)
                        processed_docs[company][filename] = doc_data
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(processed_docs, f)
        print("Processed documents saved to cache.")

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        # Generate embedding for the query
        query_embedding = openai.Embedding.create(input=question, model="text-embedding-ada-002")['data'][0]['embedding']

        # Find the most relevant document nodes
        results = []
        for company, docs in processed_docs.items():
            for filename, doc in docs.items():
                if "embeddings" not in doc or not doc["embeddings"]:
                    continue

                similarities = [
                    cosine_similarity(query_embedding, emb) if emb is not None else 0
                    for emb in doc["embeddings"]
                ]

                if not similarities or max(similarities) == 0:
                    continue

                best_match_index = np.argmax(similarities)
                best_match_score = similarities[best_match_index]

                results.append({
                    "company": company,
                    "file": filename,
                    "score": best_match_score,
                    "content": doc["content"].nodes[best_match_index].text,
                    "tables": doc["tables"],
                    "images_text": doc["images_text"],
                })

        # Sort results by score
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:5]

        if not results:
            return jsonify({
                "answer": "Unfortunately, I couldn't locate specific information in the documents. However, feel free to ask a related question or rephrase your query.",
                "results": []
            })

        # Generate response using GPT
        context = "\n".join([result["content"] for result in results])
        additional_context = "\n".join([json.dumps(result["tables"], indent=2) for result in results if result["tables"]])
        context += "\n" + additional_context
        context += "\n".join([result["images_text"] for result in results if result["images_text"]])

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intelligent assistant. Use the provided context, including tables and images, to answer the user's question. "
                        "If the context lacks specific details, infer an answer or provide helpful insights. Format the response using Markdown for clear presentation."
                    ),
                },
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
            ],
        )

        answer = response["choices"][0]["message"]["content"]

        return jsonify({"answer": answer, "results": results})

    except Exception as e:
        print(f"Error during query processing: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_or_process_documents()
    app.run(debug=True)
