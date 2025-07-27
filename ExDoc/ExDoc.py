from ExDoc.headerextractor import headerExtractor
from ExDoc.headerclassifier import HeaderClassifier

class ExDoc:
    def __init__(self,pdfpath):
        self.path = pdfpath
        classifier = HeaderClassifier()
    def ExtractTexts(self):
        print("Reading "+str(self.path))
        predictions_df = ExDoc.classifier.predict_headers(self.path)
        predictions_df.rename(columns={'predicted_label':'level'},inplace=True)
        outline = predictions_df.loc[:,["level","text","page"]].to_dict(orient="records")
        filtered_df = predictions_df[(predictions_df['page'] == 0) &((predictions_df['level'] == 'H1') | (predictions_df['level'] == 'H2'))]
        if not filtered_df.empty:
            title = filtered_df.iloc[0]['text']
        else:
            title = "Untitled"
        data={"title":title,"outline":outline}
        return data
    def ExtractTexts(self):
        print("Reading "+str(self.path))
        predictions_df = ExDoc.classifier.predict_headers(self.path)
        predictions_df.rename(columns={'predicted_label':'level'},inplace=True)
        outline = predictions_df[predictions_df['text']!='other'].loc[:,["level","text","page"]].to_dict(orient="records")
        filtered_df = predictions_df[(predictions_df['page'] == 0) &((predictions_df['level'] == 'H1') | (predictions_df['level'] == 'H2'))]
        if not filtered_df.empty:
            title = filtered_df.iloc[0]['text']
        else:
            title = "Untitled"
        data={"title":title,"outline":outline}
        return data