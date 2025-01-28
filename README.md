# SEEDS: Similarity-based Expert Embedding Decision System

## 📌 Overview

SEEDS (**S**imilarity-based **E**xpert **E**mbedding **D**ecision **S**ystem) is a Retrieval-Augmented Generation (RAG) based agricultural question-answering (QA) system. It is built upon a domain-specific knowledge graph (KG), representing Cedar Apple Rust disease.



🔍 **Core Idea**:

- Uses **Knowledge Graphs (KGs)** with plant defense, pesticide, and fungal infection data.

- Generates **embeddings** for queries and KG data using **OpenAI's `text-embedding-ada-002` model**.

- Computes **cosine similarity** to rank the most relevant KG entries for a given query.

- Provides **detailed responses** using **GPT-4 Turbo** for expert-level responses.

- Incorporates **Explainable AI (XAI)** via **t-SNE, PCA, and clustering visualizations**.


🧪 **Goal**: Assist plant protection experts and agriculturists in making **data-driven decisions** for pest control.

------

## 🚀 Features
- ✅ **Knowledge Graph-based Question Answering**  
- ✅ **Embedding-based Similarity Matching**  
- ✅ **Explainable AI (XAI) Visualizations**  
- ✅ **Support for Pesticide & Plant Pathway Data**  
- ✅ **Interactive Dash Web Application**  
- ✅ **Retrieval-Augmented Generation (RAG)**  

---

## 📂 Project Structure

```
SEEDS/
│── dummy data/  # Knowledge graph (KG) and question datasets
│   ├── dummy_mesh.csv
│   ├── dummy_questions.txt
│   ├── synthetic_properties.csv
│   ├── synthetic_triples.csv
│── src/                   # Source code
│   ├── QA_ada.py           # QA system using embedding similarity & detailed response generation by GPT       
│   ├── XAI.py             # Dash web app for Explainable AI visualizations
│── .gitignore             # Ignore venv, embeddings, results, sensitive files
│── LICENSE # Open-source license information
│── README.md              # Project documentation
│── requirements.txt       # Required dependencies
```

---

## ⚙️ Installation & Setup

### 1️⃣ **Clone the Repository**

```bash
git clone https://github.com/YOUR_USERNAME/SEEDS.git
cd SEEDS
```

### 2️⃣ **Create Virtual Environment** (Recommended)

```bash
python -m venv .seeds
source .seeds/bin/activate  # On Mac/Linux
.seeds\Scripts\activate     # On Windows
```

### 3️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 4️⃣ **Set up ****\`\`**** file**

Create a `.env` file with:

```bash
OPENAI_API_KEY=your-api-key-here
```

---


## 🎯 Usage

### 🔹 **1. Run the QA System (CLI-Based)**

```bash
python src/QA_ada.py --prompts "dummy data/dummy_questions.txt" --triples "dummy data/synthetic_triples.csv" --properties "dummy data/synthetic_properties.csv" --output "results/output.csv" --prompt_emb_path "embeddings/prompt_embeddings.pkl" --kg_text_emb_path "embeddings/kg_text_embeddings.pkl" --kg_node_emb_path "embeddings/kg_node_info_embeddings.pkl"

```


### 🔹 **2. Run the XAI Dashboard (Web App)**

```bash
python src/XAI.py --prompts "dummy data/dummy_questions.txt" --triples "dummy data/synthetic_triples.csv" --properties "dummy data/synthetic_properties.csv" --mesh_terms "dummy data/dummy_mesh.csv" --embeddings "embeddings/"
```



- Open in **browser**: [http://127.0.0.1:8050](http://127.0.0.1:8050/)

---

## 📊 Example Input & Output

### **📝 Sample Query** 


What is the effect of Fungicide X on _Venturia_ Fungi?


### **📌 Top KG Results**


1. Fungicide X inhibits _Venturia_ by disrupting cell walls (Similarity: 0.92)
2. Fungicide X affects fungal spore germination (Similarity: 0.89)
3. _Venturia_ causes apple and pear scab (Similarity: 0.78)


### **🤖 GPT-4 Generated Answer**

Fungicide X effectively inhibits the growth of _Venturia_ Fungi by disrupting its cell wall integrity. The mode of action primarily targets fungal spores, preventing germination and further infection. In field trials, it has shown strong antifungal properties against various _Venturia_ strains. Thus, fungicide X can be used to manage apple and pear scab diseases. 


---

## 📜 Citation

If you use this work, please cite:

@misc{SEEDS2024, \
  author = {Astha Anand and Marc Jacobs}, \
  title = {SEEDS: Similarity-based Expert Embedding Decision System}, \
  year = {2024}, \
  institution = {Fraunhofer Institute for Algorithms and Scientific Computing (SCAI)}, \
  howpublished = {\url{https://github.com/SCAI-BIO/SEEDS}}, \
  note = {GitHub repository}
}

---

## 🔥 Future Work

- **Expand KG** with more crops, pests, weather data, etc.
- **Integrate multimodal features** (e.g., plant disease images)
- **User feedback loop** for iterative model improvement. 


---

## 📬 Contact

- **Author:** Astha Anand
- **Supervisor:** Dr. Marc Jacobs
- **Affiliation:** Fraunhofer Institute for Algorithms and Scientific Computing (SCAI)
- **Email:** [astha.anand@scai.fraunhofer.de](mailto\:astha.anand@scai.fraunhofer.de)

---

🚀 **Happy Coding! Let’s improve agricultural pest control with AI!** 🎯🌿


