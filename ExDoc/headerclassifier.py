import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.utils import resample
# from sentence_transformers import SentenceTransformer
from ExDoc.headerextractor import headerExtractor
import os

class HeaderClassifier:
    def __init__(self, model_output_path="header_classifier_combined_new4.pkl"):
        self.EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
        self.MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "models", model_output_path)
        self.model = None
        self.meta_model = None
        self.font_encoder = None
        self.target_encoder = None
        self.feature_columns = None
        # self.embedding_model = SentenceTransformer(self.EMBEDDING_MODEL)

    def train(self, df):
        try:
            df_majority = df[df["label"] == "other"]
            df_minority = df[df["label"] != "other"]

            df_minority_upsampled = resample(
                df_minority,
                replace=True,
                n_samples=(len(df_majority) * 2),
                random_state=42,
            )
            df_balanced = pd.concat([df_majority, df_minority_upsampled])
            df = df_balanced
            df.dropna(inplace=True)
            if df.empty:
                print("Warning: DataFrame is empty after dropping NaN values.")
                return

            font_encoder = LabelEncoder()
            df["font"] = df["font"].astype(str)
            font_encoder.fit(list(df["font"].unique()) + ["unknown"])
            df["font_name_encoded"] = df["font"].apply(
                lambda x: x if x in font_encoder.classes_ else "unknown"
            )
            df["font_name_encoded"] = font_encoder.transform(df["font_name_encoded"])

            feature_columns = [
                "font_name_encoded",
                "font_size",
                "word_count",
                "isbold",
                "line_indent",
                "line_center_offset",
                "line_width_ratio",
                "line_spacing_above",
                "is_all_caps",
                "is_title_case",
                "line_length_chars",
                "line_density",
                "relative_y0",
                "is_bold_and_large",
            ]

            X = df[feature_columns]
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify=y, random_state=42
            )

            cat_cols = [
                "font_name_encoded",
                "font_size",
                "isbold",
                "is_bold_and_large",
                "is_all_caps",
                "is_title_case",
            ]
            for col in cat_cols:
                df[col] = df[col].astype("category")

            model = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.0001,
                num_leaves=31,
                max_depth=6,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=1.0,
                class_weight="balanced",
                random_state=42,
            )

            target_encoder = LabelEncoder()
            target_encoder.fit(list(y.unique()))
            y_train_encoded = target_encoder.transform(y_train)
            y_test_encoded = target_encoder.transform(y_test)

            model.fit(X_train, y_train)

            model_train = model.predict_proba(X_test)
            meta_model = LogisticRegression(max_iter=3000)
            meta_model.fit(model_train, np.argmax(model_train, axis=1))

            stacked_model = StackingClassifier(
                estimators=[("lgb", model)],
                final_estimator=meta_model,
                passthrough=True,
                n_jobs=-1,
            )

            stacked_model.fit(X_train, y_train)

            joblib.dump(
                {
                    "model": stacked_model,
                    "meta_model": meta_model,
                    "font_encoder": font_encoder,
                    "target_encoder": target_encoder,
                    "feature_columns": feature_columns,
                },
                self.MODEL_OUTPUT_PATH,
            )

            print(f"✅ Model saved as {self.MODEL_OUTPUT_PATH}")
        except Exception as e:
            print(f"❌ Error during training: {e}")
            pass

    def load_model(self):
        model_bundle = joblib.load(self.MODEL_OUTPUT_PATH)
        self.model = model_bundle["model"]
        self.meta_model = model_bundle["meta_model"]
        self.font_encoder = model_bundle["font_encoder"]
        self.target_encoder = model_bundle["target_encoder"]
        self.feature_columns = model_bundle["feature_columns"]

    def predict_headers(self, pdf_path):
        self.load_model()
        print("Extracting Text from PDF")
        df_new = headerExtractor.extract_sentences(pdf_path)

        df_new['font'] = df_new['font'].astype(str)
        df_new['font_name_encoded'] = df_new['font'].apply(
            lambda x: x if x in self.font_encoder.classes_ else 'unknown'
        )
        df_new['font_name_encoded'] = self.font_encoder.transform(df_new['font_name_encoded'])

        X_new = df_new[self.feature_columns]
        print("Analysising PDF layout")
        lgb_probs = self.model.predict_proba(X_new)
        meta_preds = self.meta_model.predict(lgb_probs)
        df_new['predicted_label'] = self.target_encoder.inverse_transform(meta_preds)

        return df_new[['text', 'page', 'predicted_label']]

# if __name__ == '__main__':
#     print("Predicts heading level based on layout features")



