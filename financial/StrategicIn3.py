import ollama
from pypdf import PdfReader
import os

def read_pdf(file_path):
    """Read PDF and extract text from all pages."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=4000):
    """Split text into chunks of roughly chunk_size characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Avoid cutting words in half
        if end < len(text):
            while end > start and text[end] != " ":
                end -= 1
        chunks.append(text[start:end].strip())
        start = end
    return chunks

def extract_strategic_initiatives(report_text_chunk, company_name, model):
    """Use LLM to extract initiatives from a chunk of text."""
    prompt = f"""
    You are an analyst and IT architect.
    Here is a chunk from the annual report of {company_name}:

    {report_text_chunk}

    Task: Extract and list ONLY the strategic IT initiatives mentioned.
    
    ❌ Do NOT explain your reasoning.  
    ❌ Do NOT include commentary, introductions, or meta-thoughts.  
    ✅ ONLY output bullet points.  

    Format requirements:
    - First list bullet points in English.  
    - Then list the same bullet points in Danish.  
    - Use "•" for each bullet.  
    """
    response = ollama.chat(model=model, messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']

def process_reports(directory, model):
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            company_name = filename.split("_")[0]
            file_path = os.path.join(directory, filename)

            print(f"\n📊 Analyzing report for: {company_name}")
            print("-" * 60)

            try:
                report_text = read_pdf(file_path)
                chunks = chunk_text(report_text, chunk_size=4000)

                all_initiatives = []
                for chunk in chunks:
                    initiatives = extract_strategic_initiatives(chunk, company_name, model)
                    # Split by lines to separate bullet points
                    lines = [line.strip("-• ").strip() for line in initiatives.splitlines() if line.strip()]
                    all_initiatives.extend(lines)

                # Deduplicate
                all_initiatives = list(dict.fromkeys(all_initiatives))

                print("Strategic initiatives found:\n")
                for init in all_initiatives:
                    print(f"• {init}")
                print("\n" + "-"*60)

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    reports_dir = r"D:\tmp\Code\financial\input"  # adjust path
    model_to_use = "qwen3:8b"                      # adjust model

    process_reports(reports_dir, model=model_to_use)
