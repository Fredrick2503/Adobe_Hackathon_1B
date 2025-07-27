import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import numpy as np

class parseDoc:
    bi_encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")  # ~80MB, fast
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # ~430MB, accurate

    @staticmethod
    def build_header_sections(df):
        headers = [None]
        res=[]
        section_text=''
        label_masked= {'H1':1,'H2':2 ,'H3':3,'H4':4,'other':5}
        for _, row in df.iterrows():
            label = row['predicted_label']
            text = row['text']
            if label_masked[label]<5:
                if  headers[-1] == None or (label_masked[label]>headers[-1]):
                    headers.append(label_masked[label])
                else:
                    res.append({'level':label,'text':text,'section_text':section_text,"page":row['page'],"doc":row['doc']})
                    section_text=''
                    headers.pop()
            else:
                section_text = section_text + text
    
        return pd.DataFrame(res)
    @staticmethod
    def normalize_scores_to_relevance(ranked_headers,min_score=None,max_score=None):
        scores = ranked_headers["relevance_score"]
        if min_score is None:
            min_score = min(scores)
        if max_score is None:
            max_score = max(scores)
        def map_score(score):
            norm = (score-min_score) / (max_score - min_score)  
            relevance = round(norm * 5) 
            return int(np.clip(relevance, 0, 5))
        ranked_headers['rank']=ranked_headers["relevance_score"].apply(lambda score: map_score(score))
        return ranked_headers

    @staticmethod
    def rank_headers_by_query_and_persona(df, query, persona, top_k=25):
        headers=parseDoc.build_header_sections(df)
        if headers.empty:
            return []
        full_query = f"As a {persona}, I wants to {query}."
        query_embedding = parseDoc.bi_encoder.encode(full_query, convert_to_tensor=True)
        header_embeddings = parseDoc.bi_encoder.encode(headers["text"], convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, header_embeddings)[0]
        top_k = min(top_k, len(cosine_scores))
        top_results = np.argpartition(-cosine_scores, range(top_k))[:top_k]
        cross_inputs = [(full_query, headers.loc[int(idx),"text"]) for idx in top_results]
        relevance_scores = parseDoc.cross_encoder.predict(cross_inputs)

        reranked = sorted(
            zip([headers.loc[int(i),:] for i in top_results], relevance_scores),
            key=lambda x: x[1],
            reverse=True
        )
        reranked_df = pd.DataFrame([row[0] for row in reranked])
        reranked_df["relevance_score"] = [row[1] for row in reranked]
        normalized_rank =parseDoc.normalize_scores_to_relevance(reranked_df)
        return normalized_rank
