# Adobe_Hackathon_Round_1A
# SLM Inference - Section Labeling Model (Offline)

This project processes PDF documents to extract and classify section headings (like Title, H1, H2, H3) using a pretrained transformer model (all-MiniLM-L6-v2) and a custom trained classifier. It works completely *offline*.

---

## 📁 Folder Structure
 input/ # Place your input PDF files here
├── output/ # JSON files with title and headings will be saved here
├── model/ # Contains classifier files (label_encoder.pkl, slm_classifier.pkl)
├── hf_cache/ # Hugging Face model cache for offline use
├── predict.py # Main script to run inference
├── requirements.txt # Dependencies
└── README.md


how to run

pip install -r requirements.txt
python main.py
