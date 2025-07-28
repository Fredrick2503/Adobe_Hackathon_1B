import argparse
import pathlib
from datetime import datetime
import pandas as pd
import json

# ----------------------------
# Parse Command Line Arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Document summarizer for persona-based job queries.")
parser.add_argument("--persona", required=True, help="Persona (e.g., 'PhD Researcher in Biology')")
parser.add_argument("--query", required=True, help="Job to be done (e.g., 'Literature review on X')")
args = parser.parse_args()

persona = args.persona
query = args.query

# ----------------------------
# Load PDFs
# ----------------------------
print("Loading PDFs...")
pdfdir = pathlib.Path("./PDFs")
pdfs = [pdf for pdf in pdfdir.glob("*.pdf")]

# ----------------------------
# Import HeaderClassifier
# ----------------------------
from ExDoc.headerclassifier import HeaderClassifier
Classfier = HeaderClassifier()

final_extracted_section = pd.DataFrame()
final_subsection = pd.DataFrame()

# ----------------------------
# Process Each PDF
# ----------------------------
for pdf in pdfs:
    data = Classfier.predict_headers(str(pdf))
    data['doc'] = pdf.name

    from QDoc.parseDoc import parseDoc
    ranked_data = parseDoc.rank_headers_by_query_and_persona(data, query=query, persona=persona)

    extracted_section = ranked_data.loc[:, ['doc', 'text', 'rank', 'page']]
    extracted_section.rename(columns={
        "doc": "document",
        "text": "section_title",
        "rank": "importance_rank",
        "page": "page_number"
    }, inplace=True)

    subsection_analysis = ranked_data.loc[:, ["doc", "section_text", "page"]]
    subsection_analysis.rename(columns={
        "doc": "document",
        "section_text": "refined_text",
        "page": "page_number"
    }, inplace=True)

    final_extracted_section = pd.concat([final_extracted_section, extracted_section], ignore_index=True)
    final_subsection = pd.concat([final_subsection, subsection_analysis], ignore_index=True)

# ----------------------------
# Metadata & Output
# ----------------------------
metadata = {
    "input_documents": final_extracted_section['document'].unique().tolist(),
    "persona": persona,
    "job_to_be_done": query,
    "processing_timestamp": datetime.now().isoformat()
}

final_output = {
    "metadata": metadata,
    "extracted_sections": final_extracted_section.to_dict(orient="records"),
    "subsection_analysis": final_subsection.to_dict(orient="records")
}

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2)

print("✅ Done. Output saved to output.json")
