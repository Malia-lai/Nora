# Nora 🌿

Nora is an AI chatbot that recommends skincare ingredients based on your specific skin issues. Instead of generic advice, Nora personalizes every response to you — using a short survey and a warm, older-sister tone.

## How It Works

1. **Enter your name** — When you first open the app, Nora asks for your name or a nickname.
2. **Fill out a short survey** — A quick set of questions about your skin type and concerns, used to personalize your experience.
3. **Chat with Nora** — Once the survey is done, you land on the chat page. You can either:
   - Type your own question, or
   - Choose from a list of sample questions.
4. **Get personalized answers** — Nora responds based on what you shared in the survey, recommending ingredients suited to your skin — all in a warm, caring tone.

## Tech Stack

- **PyTorch** — Powers the underlying model that processes and generates Nora's responses.
- **RAG (Retrieval-Augmented Generation)** — Before answering, Nora retrieves relevant skincare information rather than relying purely on generated guesses, making her answers more grounded and accurate.
- **Gradio** — Used to build the interface, providing a clean and functional UI without building a frontend from scratch.

## Project Structure

<a href="https://huggingface.co/spaces/kode-with-klossy/3.4-groupD1-capstone">Canva</a> link for the presentation of our team and the chatbot as a capstone project for our participation to Kode With Klossy.

```
nora/
├── app.py              # Main Gradio app
│── knowledge_base/  # Source documents used for retrieval
├── requirements.txt
└── README.md
```


## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Gradio

### Installation

```bash
git clone <https://github.com/Malia-lai/Nora>
cd nora
pip install -r requirements.txt
```

### Running the App

Launch the app using this <a href="https://huggingface.co/spaces/kode-with-klossy/3.4-groupD1-capstone">link!</a> 

## Future Improvements

- [ ] Add follow-up question support
- [ ] Save and revisit past recommendations
- [ ] Expand the knowledge base for more skin concerns
