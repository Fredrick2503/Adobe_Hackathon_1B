# Document Summarization System for Persona-Based Queries

A modular system designed to extract, rank, and summarize relevant sections from academic or technical PDFs based on a given persona and a job to be done. It combines layout-based header classification, NLP-driven ranking, and JSON-based structured output.

---

## Tech Stack

| Layer           | Tool / Library              | Purpose                                       |
|----------------|-----------------------------|-----------------------------------------------|
| Backend         | Python                      | Core logic and orchestration                  |
|                | Pandas, NumPy                | Data handling and preprocessing               |
|                | SentenceTransformers         | Semantic ranking of text sections             |
|                | Scikit-learn, LightGBM       | ML-based header classification                |
|                | Argparse, JSON               | Argument parsing and result export            |
| PDF Parsing     | PyMuPDF (fitz)              | PDF layout and text extraction                |
| Model Storage   | Joblib                      | Header model serialization and reuse          |

---

## Functional Pipeline

### 1. Command-line Argument Input

The script accepts the following arguments:
- `--persona`: Defines the perspective (e.g., Data Scientist, Research Scholar).
- `--query`: Defines the objective or task (e.g., Literature Review on Quantum Algorithms).

These parameters are used to customize document relevance.

---

### 2. PDF Loading

- All PDFs are read from the local `./PDFs` directory using `pathlib`.
- Each document is processed individually and independently.

---

### 3. Header Classification

| Module           | Task                                       |
|------------------|--------------------------------------------|
| `HeaderClassifier` | Detects structural headers in the PDF      |
| Feature Inputs   | Font size, indentation, spacing, boldness  |
| Model            | LightGBM, trained on visual and layout features |

The classifier identifies hierarchical headers (H1, H2, H3) using supervised learning.

---

### 4. Query-Persona-Based Ranking

| Component                              | Function                                                                 |
|----------------------------------------|--------------------------------------------------------------------------|
| `parseDoc.rank_headers_by_query_and_persona` | Uses sentence embeddings to compute similarity between section text and the input query-persona pair |
| Ranking Basis                          | Semantic relevance, heading position, section structure                  |

This module ensures extracted sections are aligned with the user intent.

---

### 5. Section and Subsection Aggregation

- Extracted headers and their corresponding page numbers are saved.
- Relevant subsection texts are paired and recorded.
- Combined using Pandas into two final DataFrames.

---

### 6. JSON Output Generation

| Output File     | Content                                                    |
|-----------------|-------------------------------------------------------------|
| `output.json`   | Metadata, ranked section headers, subsection analysis       |

Metadata includes:
- Input documents used
- Persona
- Job description
- Processing timestamp

---

## Requirements

Listed in `requirements.txt`:

pandas==2.3.1
numpy==2.2.6
sentence-transformers==5.0.0
lightgbm==4.6.0
joblib==1.5.1
scikit-learn==1.7.0


---

## Docker Support

The system is Dockerized for reproducible deployment.

### Dockerfile Contents

- Python environment with required dependencies
- Mounted volume support for input PDFs
- Entrypoint configured for argument passing

### Usage


 docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
 docker run -it --rm -v "${PWD}/PDFs:/app/PDFs" adobe-hackathon-1b

Design Principles
Feature	Description
Modularity	Components for header extraction and ranking are decoupled
Reusability	Models stored and reused via joblib
Scalability	Can process multiple documents; suitable for batch runs
Extensibility	Plug-and-play support for more advanced ranking or models

Conclusion
This summarization system leverages a blend of layout-aware classification and semantic similarity to extract and prioritize document sections relevant to a specific persona and task. It is designed for efficient research, content triage, and document intelligence use cases.