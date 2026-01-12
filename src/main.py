import nltk
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')

# Текст статьи
text = """
Graph theory is a branch of mathematics that studies the properties of graphs.
Graphs consist of vertices and edges. Graph algorithms are widely used in data analysis.
"""

# Токенизация
sentences = nltk.sent_tokenize(text)

# TF-IDF для терминов
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(sentences)
terms = vectorizer.get_feature_names_out()

# Нейросетевые эмбеддинги
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(terms, convert_to_tensor=True)

# Построение графа
G = nx.Graph()

for term in terms:
    G.add_node(term)

for i in range(len(terms)):
    for j in range(i + 1, len(terms)):
        sim = util.cos_sim(embeddings[i], embeddings[j])
        if sim > 0.5:
            G.add_edge(terms[i], terms[j], weight=float(sim))

# Визуализация
plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=True, node_size=1000, font_size=8)
plt.show()
