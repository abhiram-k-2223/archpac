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

def find_similar_packages(query, packages, embeddings, filters, top_k=100):
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    # Pair packages with their similarity scores
    results = [
        (pkg, sim)
        for pkg, sim in zip(packages, similarities)
        if (filters.get('official', False) and pkg.get('Repository:', '').lower() == 'official') or
           (filters.get('aur', False) and pkg.get('Repository:', '').lower() == 'aur') or
           (filters.get('community', False) and pkg.get('Repository:', '').lower() == 'community') or
           not any(filters.values())  # Include all if no filters are active
    ]

    # Sort by similarity score and take top_k
    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    return [(pkg['title'], sim) for pkg, sim in results]

def title_split(s):
    return s.split(" ")[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    filters = data.get('filters', {'official': True, 'aur': False, 'community': False})

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    results = find_similar_packages(query, packages, embeddings, filters)

    formatted_results = []
    for package_name, score in results:
        package_details = next(pkg for pkg in packages if pkg['title'] == package_name)

        formatted_results.append({
            'name': package_name,
            'description': package_details.get('Description:', 'No description available.'),
            'repo': package_details.get('Repository:', 'Unknown repository'),
            'maintainer': package_details.get('Maintainers:', 'Unknown maintainer'),
            'install': f'sudo pacman -S {title_split(package_name)}',
            'score': score,  # Return as float for frontend sorting
            # 'dependencies': package_details.get('Dependencies:', []),  # Adjust based on your data
            # 'optionalDependencies': package_details.get('Optional Dependencies:', []),
            # 'files': package_details.get('Files:', [])
        })

    return jsonify({'results': formatted_results})

@app.route('/api/suggestions', methods=['POST'])
def suggestions():
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({'suggestions': []})

    # Use a lightweight search for suggestions (e.g., prefix matching on titles)
    suggestions = [
        pkg['title']
        for pkg in packages
        if query.lower() in pkg['title'].lower()
    ][:10]

    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    app.run(debug=True)
