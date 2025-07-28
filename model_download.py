from sentence_transformers import SentenceTransformer, CrossEncoder
import os

model_dir = os.path.join(os.path.dirname(__file__), "models")
from sentence_transformers import SentenceTransformer, CrossEncoder

# Download and save bi_encoder
bi_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
bi_model.save(os.path.join(model_dir,"multi-qa-MiniLM-L6-cos-v1"))

# Download and save cross_encoder
cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
cross_model.save(os.path.join(model_dir,"cross-encoder-ms-marco-MiniLM-L-6-v2"))

