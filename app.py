# app.py
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load model and data
model = SentenceTransformer('all-MiniLM-L6-v2')
with open('packages_with_clusters.json', 'r') as file:
    packages = json.load(file)
embeddings = np.array([pkg['embedding'] for pkg in packages])

def find_similar_packages(query, packages, model, top_k=20):
    query_embedding = model.encode(query)
    similarities = [
        (pkg['title'], cosine_similarity([pkg['embedding']], [query_embedding])[0][0])
        for pkg in packages
    ]
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

def title_split(s):
    return s.split(" ")[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    results = find_similar_packages(query, packages, model)
    
    formatted_results = []
    for package_name, score in results:
        package_details = next(pkg for pkg in packages if pkg["title"] == package_name)
        
        formatted_results.append({
            'name': package_name,
            'description': package_details.get("Description:", "No description available."),
            'repo': package_details.get("Repository:", "Unknown repository"),
            'maintainer': package_details.get("Maintainers:", "Unknown maintainer"),
            'install': f"sudo pacman -S {title_split(package_name)}",
            'score': f"{score:.2f}"
        })
    
    return jsonify({'results': formatted_results, 'query': query})

if __name__ == "__main__":
    app.run(debug=True)
