import xml.etree.ElementTree as ET
import pandas as pd
import re
import time
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run the ESBEAS-XAI application.")
parser.add_argument("--prompts", required=True, help="Path to the prompts file.")
parser.add_argument("--triples", required=True, help="Path to the triples CSV file.")
parser.add_argument("--properties", required=True, help="Path to the prop2 CSV file.")
parser.add_argument("--mesh_terms", required=True, help="Path to the mesh terms CSV file.")
parser.add_argument("--embeddings", required=True, help="Directory to save/load embeddings.")
args = parser.parse_args()

# Function to preprocess text
def preprocess_text(text):
    #text = re.sub(r'[\^\w\s]', '', text) # Keep question marks and punctuation
    text = re.sub(r'[^\w\s?.!]', '', text)
    #return text.lower()
    return text.strip().lower()  # Ensure stripping of whitespace

# Function to load prompts
def load_prompts(file_path):
    with open(file_path, 'r') as file:
        prompts = file.readlines()
    prompts = [preprocess_text(prompt.strip()) for prompt in prompts if prompt.strip()]
    return prompts

# Load and preprocess prompts
prompts = load_prompts(args.prompts)


# Load KG data
kg_data = pd.read_csv(args.triples)
kg_data['text'] = kg_data['StartNode'] + ' ' + kg_data['Relationship'] + ' ' + kg_data['EndNode']
prop2_data = pd.read_csv(args.properties)
prop2_data['NodeInfo'] = prop2_data['NodeInfo'].str.replace('\n', '')
df = pd.DataFrame({'text': kg_data['text'], 'NodeInfo': prop2_data['NodeInfo']})

# Load MeSH terms and descriptions
mesh_data = pd.read_csv(args.mesh_terms)
mesh_terms = mesh_data['Term'].tolist()
mesh_descriptions = mesh_data['Description'].tolist()

# Combine terms and descriptions for embedding generation
mesh_texts = [f"{term} {desc}" for term, desc in zip(mesh_terms, mesh_descriptions)]

# Function to generate embeddings using OpenAI API with retry logic and batch processing
def generate_embeddings(texts, batch_size=100, max_retries=5):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        retries = 0
        while retries < max_retries:
            try:
                response = openai.Embedding.create(input=batch, model="text-embedding-ada-002")
                embeddings.extend([data['embedding'] for data in response['data']])
                break
            except openai.error.ServiceUnavailableError:
                retries += 1
                time.sleep(2 ** retries)  # Exponential backoff
                if retries == max_retries:
                    raise
    return embeddings

# Function to save/load embeddings
def save_load_embeddings(file_path, texts):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        embeddings = generate_embeddings(texts)
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
        return embeddings

# File paths for embeddings
mesh_embeddings_file = os.path.join(args.embeddings, "mesh_embeddings_all.pkl")
prompt_embeddings_file = os.path.join(args.embeddings, "prompt_embeddings.pkl")
kg_text_embeddings_file = os.path.join(args.embeddings, "kg_text_embeddings.pkl")
kg_node_info_embeddings_file = os.path.join(args.embeddings, "kg_node_info_embeddings.pkl")

# Generate or load embeddings
mesh_embeddings = save_load_embeddings(mesh_embeddings_file, mesh_texts)
prompt_embeddings = save_load_embeddings(prompt_embeddings_file, prompts)
kg_text_embeddings = save_load_embeddings(kg_text_embeddings_file, df["text"].tolist())
kg_node_info_embeddings = save_load_embeddings(kg_node_info_embeddings_file, df["NodeInfo"].tolist())

# Convert embeddings to numpy arrays for similarity calculation
prompt_embeddings = np.array(prompt_embeddings)
kg_text_embeddings = np.array(kg_text_embeddings)
kg_node_info_embeddings = np.array(kg_node_info_embeddings)
mesh_embeddings = np.array(mesh_embeddings)

# Function to calculate similarity
def calculate_similarity(embeddings_1, embeddings_2):
    return cosine_similarity(embeddings_1, embeddings_2)

# Calculate similarity scores between prompts and KG entries
prompt_kg_text_similarity = calculate_similarity(prompt_embeddings, kg_text_embeddings)
prompt_kg_node_info_similarity = calculate_similarity(prompt_embeddings, kg_node_info_embeddings)

# Function to get top N most similar KG entries for a given prompt
def get_top_n_kg_entries(similarity_scores_text, similarity_scores_node, n=5):
    combined_scores = np.hstack((similarity_scores_text, similarity_scores_node))
    top_n_kg_entries = []
    for scores in combined_scores:
        top_indices = scores.argsort()[-n:][::-1]
        top_n_kg_entries.append(top_indices)
    return top_n_kg_entries

# Get top N most similar KG entries for each prompt
top_n_kg_entries = get_top_n_kg_entries(prompt_kg_text_similarity, prompt_kg_node_info_similarity, n=5)

# Function to assign most similar label
def assign_most_similar_label(similarity_scores, labels):
    assigned_labels = []
    for scores in similarity_scores.T:
        assigned_labels.append(labels[np.argmax(scores)])
    return assigned_labels

# Assign MeSH labels to KG text and KG NodeInfo
kg_text_labels = assign_most_similar_label(calculate_similarity(mesh_embeddings, kg_text_embeddings), mesh_terms)
kg_node_info_labels = assign_most_similar_label(calculate_similarity(mesh_embeddings, kg_node_info_embeddings), mesh_terms)

# Combine embeddings and labels for visualization
all_embeddings = np.vstack((kg_text_embeddings, kg_node_info_embeddings))
labels = kg_text_labels + kg_node_info_labels
texts = df["text"].tolist() + df["NodeInfo"].tolist()

# Function to perform clustering and return the cluster assignments
def perform_clustering(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embeddings)
    clusters = kmeans.labels_
    return clusters, kmeans

# Perform clustering
num_clusters = 2 
clusters, kmeans = perform_clustering(all_embeddings, num_clusters)

# Function to assign multiple labels to each cluster
def assign_cluster_labels(cluster_assignments, mesh_terms, num_clusters):
    cluster_labels = {i: [] for i in range(num_clusters)}
    for idx, cluster in enumerate(cluster_assignments):
        cluster_labels[cluster].append(mesh_terms[idx])
    # Get the most common labels for each cluster
    for cluster in cluster_labels:
        labels = cluster_labels[cluster]
        label_counts = pd.Series(labels).value_counts()
        cluster_labels[cluster] = label_counts.index.tolist()
    return cluster_labels

# Assign cluster labels
cluster_labels = assign_cluster_labels(clusters, labels, num_clusters)

# Function to reduce dimensions using t-SNE
def reduce_dimensions_tsne(embeddings, n_components=2, perplexity=30):
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(perplexity, len(embeddings) - 1))
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings

# Function to reduce dimensions using PCA
def reduce_dimensions_pca(embeddings, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

# Reduce dimensions
reduced_embeddings_tsne = reduce_dimensions_tsne(all_embeddings)
reduced_embeddings_pca = reduce_dimensions_pca(all_embeddings)

# Function to filter top KG entries for a specific prompt
def filter_top_kg_entries(prompt_index, top_n_kg_entries, reduced_embeddings):
    top_indices = top_n_kg_entries[prompt_index]
    filtered_embeddings = reduced_embeddings[top_indices]
    filtered_labels = [labels[i] for i in top_indices]
    filtered_texts = [texts[i] for i in top_indices]
    return filtered_embeddings, filtered_labels, filtered_texts

# Function to generate the elbow plot data
def calculate_elbow_data(embeddings, max_clusters=20):
    sse = []
    max_clusters = min(max_clusters, len(embeddings))  # Ensure max_clusters <= number of samples
    for k in range(1, max_clusters + 1):
        #kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
        sse.append(kmeans.inertia_)
    return sse


# Calculate the elbow plot data
elbow_data = calculate_elbow_data(all_embeddings)

# Create a DataFrame for Plotly
df_tsne = pd.DataFrame(reduced_embeddings_tsne, columns=['x', 'y'])
df_tsne['label'] = [", ".join(cluster_labels[cluster]) for cluster in clusters]
df_tsne['text'] = texts

df_pca = pd.DataFrame(reduced_embeddings_pca, columns=['x', 'y'])
df_pca['label'] = [", ".join(cluster_labels[cluster]) for cluster in clusters]
df_pca['text'] = texts

# Plot with MeSH labels and clusters
fig_tsne_labeled = px.scatter(df_tsne, x='x', y='y', color='label', hover_data={'text': True}, title='t-SNE Plot with Clusters and MeSH Labels')
fig_pca_labeled = px.scatter(df_pca, x='x', y='y', color='label', hover_data={'text': True}, title='PCA Plot with Clusters and MeSH Labels')

# Function to explain relevance
def explain_relevance(prompt, top_kg_entries, prompt_embedding, kg_embeddings, kg_texts, kg_labels):
    explanations = []
    for idx in top_kg_entries:
        kg_embedding = kg_embeddings[idx]
        similarity = cosine_similarity([prompt_embedding], [kg_embedding])[0][0]
        mesh_label = kg_labels[idx]
        explanations.append(f"KG entry '{kg_texts[idx]}' was chosen because it has a high similarity score ({similarity:.2f}) and is labeled as '{mesh_label}'.")
    return explanations

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Embedding Visualization and XAI"),
    dcc.Tabs(id="tabs", value='tab-tsne', children=[
        dcc.Tab(label='t-SNE plot for selected prompt', value='tab-tsne'),
        dcc.Tab(label='PCA plot for selected prompt', value='tab-pca'),
        dcc.Tab(label='Explanation', value='tab-explanation'),
        dcc.Tab(label='Overall 2D Embeddings Plot', value='tab-2d-plot'),
        dcc.Tab(label='Elbow Plot', value='tab-elbow')
    ]),
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': f'Prompt {i+1}: {prompts[i]}', 'value': i} for i in range(len(prompts))],
        value=0,
        clearable=False,
        style={'width': '50%', 'margin-top': '20px'}
    ),
    dcc.Slider(
        id='top-n-slider',
        min=1,
        max=10,
        step=1,
        value=5,
        marks={i: str(i) for i in range(1, 11)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    html.Div(id='tabs-content')
], style={'margin-top': '20px'})

# Define callback to update the content based on the selected tab and dropdown
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value'), Input('dropdown', 'value'), Input('top-n-slider', 'value')]
)
def render_content(tab, selected_prompt_index, top_n):
    if tab == 'tab-tsne':
        top_n_kg_entries = get_top_n_kg_entries(prompt_kg_text_similarity, prompt_kg_node_info_similarity, top_n)
        filtered_embeddings, filtered_labels, filtered_texts = filter_top_kg_entries(selected_prompt_index, top_n_kg_entries, reduced_embeddings_tsne)
        fig_filtered_tsne = px.scatter(x=filtered_embeddings[:, 0], y=filtered_embeddings[:, 1], color=filtered_labels, hover_data={'text': filtered_texts}, title='Filtered t-SNE Plot for Selected Prompt')
        return html.Div([
            dcc.Graph(
                id='tsne-plot',
                figure=fig_filtered_tsne
            )
        ])
    elif tab == 'tab-pca':
        top_n_kg_entries = get_top_n_kg_entries(prompt_kg_text_similarity, prompt_kg_node_info_similarity, top_n)
        filtered_embeddings, filtered_labels, filtered_texts = filter_top_kg_entries(selected_prompt_index, top_n_kg_entries, reduced_embeddings_pca)
        fig_filtered_pca = px.scatter(x=filtered_embeddings[:, 0], y=filtered_embeddings[:, 1], color=filtered_labels, hover_data={'text': filtered_texts}, title='Filtered PCA Plot for Selected Prompt')
        return html.Div([
            dcc.Graph(
                id='pca-plot',
                figure=fig_filtered_pca
            )
        ])
    elif tab == 'tab-explanation':
        prompt = prompts[selected_prompt_index]
        top_n_kg_entries = get_top_n_kg_entries(prompt_kg_text_similarity, prompt_kg_node_info_similarity, top_n)
        explanations = explain_relevance(prompt, top_n_kg_entries[selected_prompt_index], prompt_embeddings[selected_prompt_index], np.vstack((kg_text_embeddings, kg_node_info_embeddings)), df["text"].tolist() + df["NodeInfo"].tolist(), kg_text_labels + kg_node_info_labels)
        
        # Calculate key concepts using MeSH labels
        key_concepts = set([kg_text_labels[i] for i in top_n_kg_entries[selected_prompt_index] if i < len(kg_text_labels)] +
                           [kg_node_info_labels[i - len(kg_text_labels)] for i in top_n_kg_entries[selected_prompt_index] if i >= len(kg_text_labels)])
        
        return html.Div([
            html.H2("XAI"),
            html.H3(f"Prompt: {prompt}"),
            html.H3("Top KG Entries:"),
            html.Ul([html.Li(explanation) for explanation in explanations]),
            html.H3("Key Concepts:"),
            html.P(", ".join(key_concepts)),
            html.H3("Cluster Concepts:"),
            html.Ul([html.Li(f"Cluster {i}: {', '.join(labels)}") for i, labels in cluster_labels.items()])
        ])
    elif tab == 'tab-2d-plot':
        return html.Div([
            dcc.Graph(
                id='tsne-plot',
                figure=fig_tsne_labeled
            ),
            dcc.Graph(
                id='pca-plot',
                figure=fig_pca_labeled
            )
        ])
    elif tab == 'tab-elbow':
        fig_elbow = px.line(x=range(1, len(elbow_data) + 1), y=elbow_data, title='Elbow Plot to Determine Optimal k', labels={'x': 'Number of Clusters (k)', 'y': 'Sum of Squared Distances'})
        return html.Div([
            dcc.Graph(
                id='elbow-plot',
                figure=fig_elbow
            )
        ])

if __name__ == '__main__':
    app.run_server(debug=True)




