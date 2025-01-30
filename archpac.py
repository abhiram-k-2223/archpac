from sentence_transformers import SentenceTransformer
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

with open('packages_with_clusters.json', 'r') as file:
    packages = json.load(file)

embeddings = np.array([pkg['embedding'] for pkg in packages])

def title_split(s):
    return s.split(" ")[0]

def bold(text):
    BOLD = "\033[1;97m"
    RESET = "\033[0m"
    return f"{BOLD}{text}{RESET}"

def find_similar_packages(query, packages, model, top_k=20):
    query_embedding = model.encode(query)

    similarities = [
        (pkg['title'], cosine_similarity([pkg['embedding']], [query_embedding])[0][0])
        for pkg in packages
    ]

    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

def format_results(query, results, packages):
    output = [f"Here are the packages similar to your query: {bold(query)}:\n"]

    for package_name, score in results:
        package_details = next(pkg for pkg in packages if pkg["title"] == package_name)

        description = package_details.get("Description:", "No description available.")
        repo = package_details.get("Repository:", "Unknown repository")
        maintainer = package_details.get("Maintainers:", "Unknown maintainer")

        output.append(
            f"- {bold(package_name)}: {bold(description)} (Similarity Score: {score:.2f})\n"
            f"  {bold('Repository')}: {repo}\n"
            f"  {bold('Maintainer')}: {maintainer}\n"
            f"  {bold('Installation')}: sudo pacman -S {title_split(package_name)}\n"
        )

    return "\n".join(output)



def archpac():
    while True:
        try:
            file_path = "logo"
            with open(file_path, "r") as file:
                ded = file.read()
            print(ded)
            query = input("What are you looking for?\n")
            results = find_similar_packages(query, packages, model)
            formatted_output = format_results(query, results, packages)
            print(formatted_output)
        except KeyboardInterrupt:
            print("\nArchPac Exiting...")
            break

if __name__ == "__main__":
    archpac()
