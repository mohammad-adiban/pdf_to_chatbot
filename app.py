# import json
# from flask import Flask, request, jsonify, render_template
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Flask app setup
# app = Flask(__name__)

# # File paths
# DATASET_FILE = "Consolidated_Annual_Reports.json"

# # Load SentenceTransformer for embeddings
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Load Hugging Face generative model for answering
# hf_model_name = "google/flan-t5-small"
# hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
# hf_model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
# qa_pipeline = pipeline("text2text-generation", model=hf_model, tokenizer=hf_tokenizer)

# # Load the JSON dataset
# def load_dataset(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         return json.load(f)

# # Preprocess the dataset to generate embeddings
# def preprocess_dataset(data):
#     texts = []
#     metadata = []

#     for record in data:
#         company = record["company"]
#         year = record["year"]

#         for page in record["pages"]:
#             cleaned_text = page["cleaned_text"]
#             texts.append(cleaned_text)
#             metadata.append({"company": company, "year": year, "key_sections": page["key_sections"]})

#     # Generate embeddings for all documents
#     embeddings = embedding_model.encode(texts, show_progress_bar=True)
#     return texts, metadata, embeddings

# # Retrieve relevant documents based on a query
# # def retrieve_documents(query, texts, embeddings, metadata, top_k=5):
# #     query_embedding = embedding_model.encode([query])
# #     similarities = cosine_similarity(query_embedding, embeddings)[0]
# #     top_indices = np.argsort(similarities)[::-1][:top_k]

# #     # Gather top-k relevant documents
# #     results = [
# #         {
# #             "content": texts[i],
# #             "metadata": metadata[i],
# #             "score": float(similarities[i])
# #         }
# #         for i in top_indices
# #     ]
# #     return results

# def retrieve_documents(query, texts, embeddings, metadata, top_k=5):
#     query_embedding = embedding_model.encode([query])
#     similarities = cosine_similarity(query_embedding, embeddings)[0]
#     top_indices = np.argsort(similarities)[::-1]

#     # Filter documents based on query metadata (company and year)
#     filtered_results = []
#     for i in top_indices:
#         if "revenue" in query.lower() and metadata[i]["key_sections"]["Revenue"]:
#             filtered_results.append({
#                 "content": metadata[i]["key_sections"]["Revenue"],
#                 "metadata": metadata[i],
#                 "score": float(similarities[i])
#             })
#         elif "profit" in query.lower() and metadata[i]["key_sections"]["Profit"]:
#             filtered_results.append({
#                 "content": metadata[i]["key_sections"]["Profit"],
#                 "metadata": metadata[i],
#                 "score": float(similarities[i])
#             })
#         # Add more filters for "Challenges", "Future Plans", etc.

#         if len(filtered_results) >= top_k:
#             break

#     # If no results match the key sections, fall back to raw content
#     if not filtered_results:
#         filtered_results = [
#             {
#                 "content": texts[i],
#                 "metadata": metadata[i],
#                 "score": float(similarities[i])
#             }
#             for i in top_indices[:top_k]
#         ]

#     return filtered_results


# # Generate answer using the Hugging Face pipeline
# # def generate_answer(context, question):
# #     prompt = f"Context: {context} Question: {question}"
# #     result = qa_pipeline(prompt)[0]["generated_text"]
# #     return result

# def generate_answer(context, question):
#     if not context.strip():
#         return "No relevant data is available to answer your question."
    
#     prompt = f"Context: {context} Question: {question}"
#     result = qa_pipeline(prompt)[0]["generated_text"]
#     return result


# # Load and preprocess dataset
# print("Loading and preprocessing dataset...")
# dataset = load_dataset(DATASET_FILE)
# texts, metadata, document_embeddings = preprocess_dataset(dataset)
# print("Dataset loaded and embeddings generated.")

# # Maintain conversational context
# context = []

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/query", methods=["POST"])
# def query():
#     global context
#     data = request.json
#     question = data.get("question", "")

#     if not question:
#         return jsonify({"error": "Question cannot be empty."}), 400

#     try:
#         # Retrieve relevant documents
#         retrieved_docs = retrieve_documents(question, texts, document_embeddings, metadata)

#         # Combine context from retrieved documents
#         combined_context = " ".join([doc["content"] for doc in retrieved_docs])

#         # Include recent conversational context
#         full_context = " ".join(context[-3:]) + " " + combined_context

#         # Generate answer
#         answer = generate_answer(full_context, question)

#         # Update conversational context
#         context.append(combined_context)

#         return jsonify({
#             "answer": answer,
#             "relevant_documents": retrieved_docs
#         })
#     except Exception as e:
#         print("Error:", str(e))
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)

####################################################################################

# import json
# from flask import Flask, request, jsonify, render_template
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Flask app setup
# app = Flask(__name__)

# # File paths
# DATASET_FILE = "Consolidated_Annual_Reports.json"

# # Load SentenceTransformer for embeddings
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Load Hugging Face generative model for answering
# hf_model_name = "google/flan-t5-small"
# hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
# hf_model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
# qa_pipeline = pipeline("text2text-generation", model=hf_model, tokenizer=hf_tokenizer)

# # Load the JSON dataset
# def load_dataset(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         return json.load(f)

# # Preprocess the dataset to generate embeddings
# def preprocess_dataset(data):
#     texts = []
#     metadata = []

#     for record in data:
#         company = record["company"]
#         year = record["year"]

#         for page in record["pages"]:
#             cleaned_text = page["cleaned_text"]
#             texts.append(cleaned_text)
#             metadata.append({"company": company, "year": year, "key_sections": page["key_sections"]})

#     # Generate embeddings for all documents
#     embeddings = embedding_model.encode(texts, show_progress_bar=True)
#     return texts, metadata, embeddings

# # Retrieve relevant documents based on a query
# def retrieve_documents(query, texts, embeddings, metadata, top_k=5):
#     query_embedding = embedding_model.encode([query])
#     similarities = cosine_similarity(query_embedding, embeddings)[0]
#     top_indices = np.argsort(similarities)[::-1]

#     # Filter documents based on query metadata (company and year)
#     filtered_results = []
#     for i in top_indices:
#         key_sections = metadata[i]["key_sections"]
#         if "revenue" in query.lower() and key_sections["Revenue"]:
#             filtered_results.append({
#                 "content": key_sections["Revenue"],
#                 "metadata": metadata[i],
#                 "score": float(similarities[i])
#             })
#         elif "profit" in query.lower() and key_sections["Profit"]:
#             filtered_results.append({
#                 "content": key_sections["Profit"],
#                 "metadata": metadata[i],
#                 "score": float(similarities[i])
#             })
#         # Add more filters for "Challenges", "Future Plans", etc.

#         if len(filtered_results) >= top_k:
#             break

#     # If no results match the key sections, fall back to raw content
#     if not filtered_results:
#         filtered_results = [
#             {
#                 "content": texts[i],
#                 "metadata": metadata[i],
#                 "score": float(similarities[i])
#             }
#             for i in top_indices[:top_k]
#         ]

#     return filtered_results

# # Generate answer using the Hugging Face pipeline
# def generate_answer(context, question):
#     if not context.strip():
#         return "I'm sorry, but I couldn't find any relevant data to answer your question."

#     prompt = f"According to the context: {context} and the input Question: {question}, please provide a clear and concise answer."
#     result = qa_pipeline(prompt)[0]["generated_text"]

#     # Enhance the response for better readability
#     improved_answer = f"{result.capitalize()}."
#     return improved_answer

# # Load and preprocess dataset
# print("Loading and preprocessing dataset...")
# dataset = load_dataset(DATASET_FILE)
# texts, metadata, document_embeddings = preprocess_dataset(dataset)
# print("Dataset loaded and embeddings generated.")

# # Maintain conversational context
# context = []

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/query", methods=["POST"])
# def query():
#     global context
#     data = request.json
#     question = data.get("question", "")

#     if not question:
#         return jsonify({"error": "Question cannot be empty."}), 400

#     try:
#         # Retrieve relevant documents
#         retrieved_docs = retrieve_documents(question, texts, document_embeddings, metadata)

#         # Combine context from retrieved documents
#         combined_context = " ".join([doc["content"] for doc in retrieved_docs if doc["content"]])

#         # Include recent conversational context
#         full_context = " ".join(context[-3:]) + " " + combined_context

#         # Generate answer
#         answer = generate_answer(full_context, question)

#         # Update conversational context
#         context.append(combined_context)

#         return jsonify({
#             "answer": answer,
#             "relevant_documents": retrieved_docs
#         })
#     except Exception as e:
#         print("Error:", str(e))
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)


##################################################################

# import json
# from flask import Flask, request, jsonify, render_template
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Flask app setup
# app = Flask(__name__)

# # File paths
# DATASET_FILE = "Consolidated_Annual_Reports.json"

# # Load SentenceTransformer for embeddings
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Load Hugging Face generative model for answering
# hf_model_name = "google/flan-t5-small"
# hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
# hf_model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
# qa_pipeline = pipeline("text2text-generation", model=hf_model, tokenizer=hf_tokenizer)

# # Load the JSON dataset
# def load_dataset(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#         if not isinstance(data, dict):
#             raise ValueError("Expected a dictionary with keys for BMW, Tesla, and Ford.")
#         return data

# # Preprocess the dataset to generate embeddings
# def preprocess_dataset(data):
#     texts = []
#     metadata = []

#     for company, records in data.items():  # Iterate over BMW, Tesla, Ford
#         for record in records:
#             for page in record["data"]:  # Iterate through pages in the record
#                 cleaned_text = page["cleaned_text"]
#                 key_sections = page.get("key_sections", {})

#                 texts.append(cleaned_text)
#                 metadata.append({
#                     "company": company,
#                     "key_sections": key_sections
#                 })

#     # Generate embeddings for all documents
#     embeddings = embedding_model.encode(texts, show_progress_bar=True)
#     return texts, metadata, embeddings

# # Retrieve relevant documents based on a query
# def retrieve_documents(query, texts, embeddings, metadata, top_k=5):
#     query_embedding = embedding_model.encode([query])
#     similarities = cosine_similarity(query_embedding, embeddings)[0]
#     top_indices = np.argsort(similarities)[::-1]

#     filtered_results = []
#     for i in top_indices:
#         key_sections = metadata[i].get("key_sections", {})
#         company = metadata[i]["company"]

#         # Example: Check for revenue or profit in the query
#         if "revenue" in query.lower() and key_sections.get("Revenue"):
#             filtered_results.append({
#                 "content": key_sections["Revenue"],
#                 "metadata": metadata[i],
#                 "score": float(similarities[i])
#             })
#         elif "profit" in query.lower() and key_sections.get("Profit"):
#             filtered_results.append({
#                 "content": key_sections["Profit"],
#                 "metadata": metadata[i],
#                 "score": float(similarities[i])
#             })

#         if len(filtered_results) >= top_k:
#             break

#     # Fall back to raw content if no results match
#     if not filtered_results:
#         filtered_results = [
#             {
#                 "content": texts[i],
#                 "metadata": metadata[i],
#                 "score": float(similarities[i])
#             }
#             for i in top_indices[:top_k]
#         ]

#     return filtered_results

# # Generate answer using the Hugging Face pipeline
# def generate_answer(context, question):
#     if not context.strip():
#         return "I'm sorry, but I couldn't find any relevant data to answer your question."

#     prompt = f"According to the context: {context} and the input Question: {question}, please provide a clear and concise answer."
#     result = qa_pipeline(prompt)[0]["generated_text"]

#     # Enhance the response for better readability
#     improved_answer = f"{result.capitalize()}"
#     return improved_answer

# # Load and preprocess dataset
# print("Loading and preprocessing dataset...")
# dataset = load_dataset(DATASET_FILE)
# texts, metadata, document_embeddings = preprocess_dataset(dataset)
# print("Dataset loaded and embeddings generated.")

# # Maintain conversational context
# context = []

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/query", methods=["POST"])
# def query():
#     global context
#     data = request.json
#     question = data.get("question", "")

#     if not question:
#         return jsonify({"error": "Question cannot be empty."}), 400

#     try:
#         # Retrieve relevant documents
#         retrieved_docs = retrieve_documents(question, texts, document_embeddings, metadata)

#         # Combine context from retrieved documents
#         combined_context = " ".join([doc["content"] for doc in retrieved_docs if doc["content"]])

#         # Include recent conversational context
#         full_context = " ".join(context[-3:]) + " " + combined_context

#         # Generate answer
#         answer = generate_answer(full_context, question)

#         # Update conversational context
#         context.append(combined_context)

#         return jsonify({
#             "answer": answer,
#             "relevant_documents": retrieved_docs
#         })
#     except Exception as e:
#         print("Error:", str(e))
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)


##########################################
## openAI
import os
import json
from flask import Flask, request, jsonify, render_template
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set OpenAI API Key
openai.api_key = "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"
print(f"Using OpenAI API Key: {openai.api_key[:5]}********")

# Flask app setup
app = Flask(__name__)

# File paths
DATASET_FILE = "Consolidated_Annual_Reports.json"

# Load SentenceTransformer for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the JSON dataset
def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Preprocess the dataset to generate embeddings
def preprocess_dataset(data):
    texts = []
    metadata = []

    for company, reports in data.items():
        for report in reports:
            for page in report["data"]:
                cleaned_text = page["cleaned_text"]
                if cleaned_text:
                    texts.append(cleaned_text)
                    metadata.append({"company": company, "key_sections": page["key_sections"], "tables": page["tables"]})

    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    return texts, metadata, embeddings

# Retrieve relevant documents based on a query
def retrieve_documents(query, texts, embeddings, metadata, top_k=5):
    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for i in top_indices:
        results.append({
            "content": texts[i],
            "metadata": metadata[i],
            "score": float(similarities[i])
        })
    return results

# Generate answer using OpenAI API
def generate_answer(context, question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that provides answers based on provided context."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
            ]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "I'm sorry, I couldn't generate an answer at this time."

# Load and preprocess dataset
print("Loading and preprocessing dataset...")
dataset = load_dataset(DATASET_FILE)
texts, metadata, document_embeddings = preprocess_dataset(dataset)
print("Dataset loaded and embeddings generated.")

# Maintain conversational context
context = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    global context
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        # Retrieve relevant documents
        retrieved_docs = retrieve_documents(question, texts, document_embeddings, metadata)

        # Combine context from retrieved documents
        combined_context = " ".join([doc["content"] for doc in retrieved_docs])

        # Include recent conversational context
        full_context = " ".join(context[-3:]) + " " + combined_context

        # Generate answer
        answer = generate_answer(full_context, question)

        # Update conversational context
        context.append(combined_context)

        return jsonify({
            "answer": answer,
            "relevant_documents": retrieved_docs
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
