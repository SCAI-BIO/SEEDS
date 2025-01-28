import argparse
import openai
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
from dotenv import load_dotenv
import os
import pickle

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def load_prompts(file_path):
    with open(file_path, 'r') as file:
        prompts = file.readlines()
    prompts = [preprocess_text(prompt.strip()) for prompt in prompts if prompt.strip()]
    return prompts

def save_embeddings(data, file_path):
    # Ensure the directory exists before saving
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_embeddings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    return None

def generate_embeddings(texts, file_path):
    embeddings = load_embeddings(file_path)
    if embeddings is None:
        embeddings = []
        for text in texts:
            response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
            embeddings.append(response['data'][0]['embedding'])
        save_embeddings(embeddings, file_path)
    return embeddings

def calculate_similarity(embeddings_1, embeddings_2):
    return cosine_similarity(embeddings_1, embeddings_2)

def rank_answers(similarity_scores, data_column, prompts, top_n=5):
    ranked_answers = []
    for i, prompt in enumerate(prompts):
        sorted_indices = similarity_scores[i].argsort()[::-1]
        ranked_answers.append((prompt, [(data_column.iloc[idx], similarity_scores[i][idx]) for idx in sorted_indices[:top_n]]))
    return ranked_answers

def generate_detailed_answers(prompt, top_triples):
    input_text = f"User Prompt: {prompt}\nTop Information:\n" + "\n".join([f"- {triple[0]} (Score: {triple[1]:.4f})" for triple in top_triples])
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a plant and pesticide expert who answers precisely."},
            {"role": "user", "content": input_text}
        ],
        max_tokens=400,
        temperature=0
    )
    return response['choices'][0]['message']['content'].strip()

def save_detailed_answers_to_csv(ranked_text_answers, detailed_text_answers, ranked_node_info_answers, detailed_node_info_answers, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Prompt", "Rank", "Answer", "Similarity Score", "Detailed Answer", "Type"])

        for (prompt, text_answers), (_, detailed_text_answer), (_, node_info_answers), (_, detailed_node_info_answer) in zip(ranked_text_answers, detailed_text_answers, ranked_node_info_answers, detailed_node_info_answers):
            writer.writerow([prompt, "", "", "", detailed_text_answer, "Text"])
            for rank, (answer, score) in enumerate(text_answers, start=1):
                writer.writerow(["", rank, answer, score, "", "Text"])
            writer.writerow(["", "", "", "", detailed_node_info_answer, "NodeInfo"])
            for rank, (answer, score) in enumerate(node_info_answers, start=1):
                writer.writerow(["", rank, answer, score, "", "NodeInfo"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Path Management for GPT QA System")
    parser.add_argument('--prompts', required=True, help="Path to the prompts file")
    parser.add_argument('--triples', required=True, help="Path to the triples CSV file")
    parser.add_argument('--properties', required=True, help="Path to the properties CSV file")
    parser.add_argument('--output', required=True, help="Output path for the results CSV")
    parser.add_argument('--prompt_emb_path', required=True, help="Path to save/load prompt embeddings")
    parser.add_argument('--kg_text_emb_path', required=True, help="Path to save/load KG text embeddings")
    parser.add_argument('--kg_node_emb_path', required=True, help="Path to save/load KG NodeInfo embeddings")

    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load and preprocess prompts
    prompts = load_prompts(args.prompts)

    # Load KG data
    kg_data = pd.read_csv(args.triples)
    kg_data['text'] = kg_data['StartNode'] + ' ' + kg_data['Relationship'] + ' ' + kg_data['EndNode']

    prop2_data = pd.read_csv(args.properties)
    prop2_data['NodeInfo'] = prop2_data['NodeInfo'].str.replace('\n', '')

    df = pd.DataFrame({'text': kg_data['text'], 'NodeInfo': prop2_data['NodeInfo']})

    # Generate or load embeddings
    prompt_embeddings = generate_embeddings(prompts, args.prompt_emb_path)
    kg_text_embeddings = generate_embeddings(df["text"].tolist(), args.kg_text_emb_path)
    kg_node_info_embeddings = generate_embeddings(df["NodeInfo"].tolist(), args.kg_node_emb_path)

    # Convert embeddings to numpy arrays for similarity calculation
    prompt_embeddings = np.array(prompt_embeddings)
    kg_text_embeddings = np.array(kg_text_embeddings)
    kg_node_info_embeddings = np.array(kg_node_info_embeddings)

    # Calculate similarity scores
    text_similarity_scores = calculate_similarity(prompt_embeddings, kg_text_embeddings)
    node_info_similarity_scores = calculate_similarity(prompt_embeddings, kg_node_info_embeddings)

    # Rank answers for text similarity (Top 5)
    ranked_text_answers = rank_answers(text_similarity_scores, df["text"], prompts, top_n=5)

    # Rank answers for NodeInfo similarity (Top 5)
    ranked_node_info_answers = rank_answers(node_info_similarity_scores, df["NodeInfo"], prompts, top_n=5)

    # Generate detailed answers for each prompt for text similarity
    detailed_text_answers = []
    for prompt, top_triples in ranked_text_answers:
        detailed_answer = generate_detailed_answers(prompt, top_triples)
        detailed_text_answers.append((prompt, detailed_answer))

    # Generate detailed answers for each prompt for NodeInfo similarity
    detailed_node_info_answers = []
    for prompt, top_triples in ranked_node_info_answers:
        detailed_answer = generate_detailed_answers(prompt, top_triples)
        detailed_node_info_answers.append((prompt, detailed_answer))

    # Save all ranked answers and detailed answers to one CSV file
    save_detailed_answers_to_csv(ranked_text_answers, detailed_text_answers, ranked_node_info_answers, detailed_node_info_answers, args.output)
    print(f"All ranked answers and detailed answers saved to '{args.output}'")
