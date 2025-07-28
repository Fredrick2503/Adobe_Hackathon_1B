import os
from sentence_transformers import SentenceTransformer, CrossEncoder

# Define model output directory (relative to this script's location)
base_dir = os.path.abspath(os.path.dirname(__file__))
model_dir = os.path.join(base_dir, "models")

# Ensure 'models' directory exists
os.makedirs(model_dir, exist_ok=True)

# Download and save SentenceTransformer model (bi-encoder)
print("Downloading bi_encoder: multi-qa-MiniLM-L6-cos-v1")
bi_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
bi_path = os.path.join(model_dir, "multi-qa-MiniLM-L6-cos-v1")
bi_model.save(bi_path)
print(f"bi_encoder saved to {bi_path}")

# Download and save CrossEncoder model
print("Downloading cross_encoder: cross-encoder/ms-marco-MiniLM-L-6-v2")
cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
cross_path = os.path.join(model_dir, "cross-encoder-ms-marco-MiniLM-L-6-v2")
cross_model.save(cross_path)
print(f"cross_encoder saved to {cross_path}")
