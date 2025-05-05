# Archpac

Archpac is a **semantic search tool** for discovering packages in the **official Arch Linux repository**. It supports both a **CLI tool** and a **Flask-based web interface**, helping users find packages using natural language queries instead of exact keywords.

## 🚀 Features

- 🔍 **Semantic Search**: Uses all-MiniLM-L6-v2 to find relevant packages based on the meaning of your query.
- 💻 **CLI Tool**: Lightweight, interactive terminal experience.
- 🌐 **Web Interface**: Minimal Flask app for querying via browser.
- 🛆 **Official Packages Only**: Searches only within the Arch Linux official repositories (not AUR).
- 🧠 **Context-Aware**: Finds packages based on purpose or functionality, even if you don’t know the exact name.

## 🛠️ Installation

### Clone the repository
```bash
git clone https://github.com/abhiram-k-2223/archpac.git
cd archpac
```

### Install dependencies
Make sure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

## 📆 CLI Usage

```bash
python archpac.py
```

You’ll be prompted with:

```
What are you looking for?
```

Enter a query like:

```
terminal emulator
```

And Archpac will return a list of the most relevant packages with installation commands and metadata.

## 🌐 Web Interface

To start the Flask web app:

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000/
```

Enter your query in the search bar and get instant results in the browser.

## 📁 Dataset

Archpac uses a preprocessed dataset (`packages_with_clusters.json`) which contains package metadata and their corresponding sentence embeddings.

> **Note:** This tool does not fetch live data or include AUR packages. Only the official Arch Linux repository is supported.

## 🧠 How it Works

- Each package’s description is converted into a sentence embedding using `all-MiniLM-L6-v2` from `sentence-transformers`.
- User queries are also embedded the same way.
- Cosine similarity is computed to find the most relevant packages.

## ⚙️ Example Query

Query:
```
i want a lightweight image viewer
```

Output (CLI or web):
```
[1] geeqie 2.5-1
    Lightweight image viewer
    Installation: sudo pacman -S geeqie
    Similarity: 0.96

[2] gpicview 0.2.5-8
    Lightweight image viewer
    Installation: sudo pacman -S gpicview
    Similarity: 0.96

[3] pqiv 2.13.2-4
    Powerful image viewer with minimal UI
    Installation: sudo pacman -S pqiv
    Similarity: 0.84
```

## 🤝 Contributing

Feel free to fork the repo and submit a pull request!

---

**Made with ❤️ for Arch Linux users.**
