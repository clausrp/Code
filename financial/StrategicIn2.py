import ollama
from pypdf import PdfReader
import os

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def extract_strategic_initiatives(report_text, company_name, model): # <- Ajust prompt if needed.
    prompt = f"""
    Du er en analytiker samt IT arkitekt.
    Her er en årsrapport fra en virksomhed:

    {report_text[:18000]}  # begræns input til modellen

    Opgave: Udtræk og list de strategiske IT-initiativer virksomheden nævner.
    Svar kort og punktvis først på engelsk her efter på dansk.
    """
    response = ollama.chat(model=model, messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']

def process_reports(directory, model):
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            company_name = filename.split("_")[0]  # take first part of file name
            file_path = os.path.join(directory, filename)
            
            print(f"\nAnalyzing report for: {company_name}")
            
            try:
                report_text = read_pdf(file_path)
                initiatives = extract_strategic_initiatives(report_text, company_name, model=model)
                print("Strategic initiatives found:\n")
                print(initiatives)
                print("\n" + "-"*60)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    reports_dir = r"D:\tmp\Code\financial\input"    # <-- adjust your directory
    model_to_use = "qwen3:8b"                       # <-- adjust model

    process_reports(reports_dir, model=model_to_use)

