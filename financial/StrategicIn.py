import ollama
from pypdf import PdfReader

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_strategic_initiatives(report_text, model="qwen3:8b"):
    prompt = f"""
    Du er en analytiker. 
    Her er en årsrapport fra en virksomhed:

    {report_text[:8000]}  # begræns input til modellen

    Opgave: Udtræk og list de strategiske IT-initiativer virksomheden nævner.
    Svar kort og punktvis på dansk.
    """
    response = ollama.chat(model=model, messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']

if __name__ == "__main__":
    # pdf_path = "D:/tmp/Code/financial/input/BEC_Aarsrapport_2024.pdf"
    pdf_path = "D:/tmp/Code/financial/input/BEC_Extract_from_Annual_Report_2024.pdf"

    report_text = read_pdf(pdf_path)
    initiatives = extract_strategic_initiatives(report_text)
    print("\nStrategiske initiativer fundet:\n")
    print(initiatives)

