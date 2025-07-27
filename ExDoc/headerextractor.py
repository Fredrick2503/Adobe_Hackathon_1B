import fitz
import pandas as pd
import numpy as np

class headerExtractor:
    BOLD_FONT_KEYWORDS = ["Bold", "Black", "Heavy", "Demi", "SemiBold"]
    @staticmethod    
    def isBold(font):
        for i in headerExtractor.BOLD_FONT_KEYWORDS:
            if i in font:
                return True
        return False
    @staticmethod
    def clean_data(df):
        df.drop_duplicates(subset=["text"], inplace=True)
        return df
    @staticmethod
    def add_layout_features(df, page_width=612, page_height=792):
        df = df.sort_values(by=['page', 'y0'])
        df['line_indent'] = df.groupby('page')['x0'].transform(lambda x: x - x.min())
        df['line_center_offset'] = abs((df['x0'] + df['x1']) / 2 - (page_width / 2))
        df['line_width_ratio'] = (df['x1'] - df['x0']) / page_width
        df['line_spacing_above'] = df.groupby('page')['y0'].diff().fillna(0)
        df["is_all_caps"] = df["text"].apply(lambda x: int(x.upper() == x))
        df["is_title_case"] = df["text"].apply(lambda x: int(x.istitle()))
        df["line_length_chars"] = df["text"].apply(len)
        df["line_density"] = df["line_length_chars"] / (df["x1"] - df["x0"] + 1e-5)
        df["relative_y0"] = df["y0"] / page_height
        df["is_bold_and_large"] = ((df["font_size"] > df["font_size"].mean()) & df["isbold"]).astype(int)
        return df
    @staticmethod
    def extract_sentences(pdf_path):
        import pandas as pd
        doc = fitz.open(pdf_path)
        sentances = []
        for page_num, page in enumerate(doc, start=0):
            page_height = page.rect.height
            page_width = page.rect.width
            for block in page.get_text("dict")["blocks"]:
                sentance= {}
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        span_text = span["text"].strip()
                        if not span_text:
                            continue
                        if not sentance:
                            sentance['text'] = span_text
                            sentance['font'] = span['font']
                            sentance['font_size'] = round(span['size'])
                            sentance['word_count'] = len(span_text.split(" "))
                            sentance['x0'],sentance['y0'],sentance['x1'],sentance['y1']=block["bbox"][0],block["bbox"][1],block["bbox"][2],block["bbox"][3]
                            sentance['isbold']=int(headerExtractor.isBold(span['font']))
                            sentance["page"] = int(page_num)
                        else:
                            sentance['text'] = sentance['text'] + (" " if sentance['text'] else "") + span_text
                            sentance['word_count'] = len(sentance['text'].split(" "))
                if sentance:
                    sentances.append(sentance)
        data = pd.DataFrame(sentances)
        data = headerExtractor.add_layout_features(data, page_width=page_width, page_height=page_height)
        return headerExtractor.clean_data(data)
    @staticmethod
    def label_blocks_from_outline(blocks_df, json_path, threshold=50):
        import json
        from rapidfuzz import fuzz
        with open(json_path, "r", encoding="utf-8") as f:
            f = json.load(f)
            title = f["title"]
            outline = f["outline"]
        def match_label(row):
            matches = [
                (o["text"], o["level"], fuzz.ratio(row["text"], o["text"].strip()))
                for o in outline if o["page"]-1 == row["page"] 
            ]
            if matches:
                best_text, best_level, score = max(matches, key=lambda x: x[2])
                if score >= threshold:
                    return str(best_level)
            return "other"

        blocks_df["label"] = blocks_df.apply(match_label, axis=1)
        return blocks_df
    @staticmethod
    def prepare_training_data(pdf_files):
        import pathlib
        import json 
        all_data = pd.DataFrame()
        for pdf_file in pdf_files:
            try:
                print(f"Processing {pdf_file}...")
                df = headerExtractor.extract_sentences(pdf_file)
                json_path = pathlib.Path(pdf_file).with_suffix(".json")
                if json_path.exists():
                    df = headerExtractor.label_blocks_from_outline(df, json_path)
                    all_data = pd.concat([all_data, df], ignore_index=True)
                else:
                    print(f"Warning: JSON outline {json_path} not found. Skipping labeling.")
            except:
                print("Error Extracting Text from PDF")
                pass
        return headerExtractor.clean_data(all_data)
    

# if __name__ == '__main__':
#     print("Extracts text from PDFs")