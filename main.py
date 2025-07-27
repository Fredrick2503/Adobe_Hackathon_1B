import pathlib
from datetime import datetime
import pandas as pd
import json


print("Loading pdfs")
pdfdir = pathlib.Path("./PDFs")
pdfs=[pdf for pdf in pdfdir.glob("*.pdf")]
persona=str(input("Persona: "))
query=str(input("Query: "))
from ExDoc.headerclassifier import HeaderClassifier
Classfier=HeaderClassifier()
final_extracted_section=pd.DataFrame()
final_subsection=pd.DataFrame()
for pdf in pdfs:
    data=Classfier.predict_headers(str(pdf))
    data['doc']=pdf.name 
    from QDoc.parseDoc import parseDoc
    ranked_data=parseDoc.rank_headers_by_query_and_persona(data,query=query,persona=persona)
    extracted_section = ranked_data.loc[:,['doc','text','rank','page']]
    extracted_section.rename(columns={"doc":"document","text":"section_title","rank":"importance_rank","page":"page_number"}, inplace=True)
    subsection_analysis= ranked_data.loc[:,["doc","section_text","page"]]
    subsection_analysis.rename(columns={"doc":"document","section_text":"refined_text","page":"page_number"}, inplace=True)
    final_extracted_section=pd.concat([final_extracted_section,extracted_section],ignore_index=True)
    final_subsection=pd.concat([final_subsection,subsection_analysis],ignore_index=True)
print(final_extracted_section.head())
metadata = {
    "input_documents":final_extracted_section.loc[:,'document'].unique().tolist(),
    "persona":persona,
    "job_to_be_done":query,
    "processing_timestamp":datetime.now().isoformat()
}
final_extracted_section=final_extracted_section.to_dict(orient="records")
final_subsection=final_subsection.to_dict(orient="records")


data = {
    "metadata":metadata,
    "extracted_sections":final_extracted_section,
    "subsection_analysis":final_subsection
}
json_path = pathlib.Path("output.json")
with json_path.open("w", encoding="utf-8") as f:
    json.dump(data,f,indent=2)
print(metadata,final_extracted_section,final_subsection)