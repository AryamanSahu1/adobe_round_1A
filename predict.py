import json
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
import os

# === CONFIGURATION ===
INPUT_DIR = "input"
OUTPUT_DIR = "output"
MODEL_DIR = "model"
CACHE_DIR = "hf_cache"

# === LOAD MODELS OFFLINE ===
transformer = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=CACHE_DIR)
classifier = joblib.load(os.path.join(MODEL_DIR, 'slm_classifier.pkl'))
label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))


def extract_text_lines(pdf_path):
    doc = fitz.open(pdf_path)
    lines = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                span_texts = [span["text"] for span in line["spans"] if span["text"].strip()]
                span_fonts = [span["size"] for span in line["spans"]]
                if span_texts:
                    text = " ".join(span_texts).strip()
                    max_font_size = max(span_fonts)
                    is_bold = any("Bold" in span.get("font", "") for span in line["spans"])
                    lines.append({
                        "text": text,
                        "page": page_num,
                        "font_size": max_font_size,
                        "is_bold": is_bold,
                        "line_height": block.get("bbox", [0, 0, 0, 0])[1]
                    })
    return lines


def detect_title(lines):
    first_page_lines = [l for l in lines if l["page"] == 1]
    if not first_page_lines:
        return None
    sorted_lines = sorted(first_page_lines, key=lambda x: (-x["font_size"], x["line_height"]))
    for line in sorted_lines:
        if len(line["text"].split()) > 2:
            return line["text"]
    return sorted_lines[0]["text"] if sorted_lines else None


def detect_h1(lines, font_threshold=2.0):
    h1_candidates = []
    pages = sorted(set(l["page"] for l in lines))
    for page in pages:
        page_lines = [l for l in lines if l["page"] == page]
        if not page_lines:
            continue
        median_font = np.median([l["font_size"] for l in page_lines])
        for idx, line in enumerate(page_lines):
            if (
                line["font_size"] > median_font + font_threshold and
                len(line["text"].split()) <= 10 and
                (idx + 1 < len(page_lines) and page_lines[idx + 1]["font_size"] < line["font_size"])
            ):
                h1_candidates.append({
                    "text": line["text"],
                    "level": "H1",
                    "page": line["page"],
                    "confidence": 1.0
                })
    return h1_candidates


def classify_headings(lines, confidence_threshold=0.5):
    h2_h3 = []
    for line in lines:
        embedding = transformer.encode([line["text"]])
        if hasattr(classifier, "predict_proba"):
            probs = classifier.predict_proba(embedding)[0]
            pred_index = np.argmax(probs)
            confidence = probs[pred_index]
        else:
            pred_index = classifier.predict(embedding)[0]
            confidence = 1.0
        label = label_encoder.inverse_transform([pred_index])[0]
        if label in {"H2", "H3"} and confidence >= confidence_threshold:
            h2_h3.append({
                "text": line["text"],
                "level": label,
                "page": line["page"],
                "confidence": confidence
            })
    return h2_h3


def format_output(title, h1s, h2_h3s):
    output = {
        "title": title,
        "outline": []
    }
    all_headings = h1s + h2_h3s
    all_headings.sort(key=lambda x: (x["page"], x.get("line_height", 0)))
    for item in all_headings:
        output["outline"].append({
            "level": item["level"],
            "text": item["text"],
            "page": item["page"]
        })
    return output


def save_output(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def process_pdf(pdf_path, output_path):
    try:
        print(f"\n📄 Processing: {pdf_path}")
        text_lines = extract_text_lines(pdf_path)
        title = detect_title(text_lines)
        h1s = detect_h1(text_lines)
        h2_h3 = classify_headings(text_lines, confidence_threshold=0.5)
        output = format_output(title, h1s, h2_h3)
        save_output(output, output_path)
        print(f"✅ Output saved: {output_path}")
        if title:
            print(f"   Title: {title}")
        else:
            print("   ⚠️ No title detected.")
    except Exception as e:
        print(f"❌ Failed to process {pdf_path}: {e}")


def process_all_pdfs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    for file in pdf_files:
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(file)[0] + ".json")
        process_pdf(input_path, output_path)


if __name__ == "__main__":
    process_all_pdfs()