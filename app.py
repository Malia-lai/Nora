import gradio as gr
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch
import base64
import mimetypes

with open("knowledge.txt", "r", encoding="utf-8") as file:
    data_text = file.read()

def preprocess_text(text):
    cleaned_text = text.strip()
    chunks = cleaned_text.split("\n")
    cleaned_chunks = []

    for chunk in chunks:
        stripped_chunk = chunk.strip()
        if len(stripped_chunk) > 0:
            cleaned_chunks.append(stripped_chunk)

    print(cleaned_chunks)
    print(len(cleaned_chunks))

    return cleaned_chunks

cleaned_chunks = preprocess_text(data_text)

client = InferenceClient(
    "Qwen/Qwen2.5-VL-72B-Instruct",
    bill_to="Kode-with-klossy"
)

model = SentenceTransformer("all-MiniLM-L6-v2")

def image_to_data_url(image_path):
    """Convert the uploaded image into a format the model can read."""

    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_image}"
    #turn into numbers

def create_embeddings(text_chunks):
  # Convert each text chunk into a vector embedding and store as a tensor
  chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True) 
  return chunk_embeddings

chunk_embeddings = create_embeddings(cleaned_chunks)

def get_top_chunks(query, chunk_embeddings, text_chunks):
  query_embedding = model.encode(query, convert_to_tensor=True) # Complete this line
  query_embedding_normalized = query_embedding / query_embedding.norm()
  chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)

  similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized)
  print(similarities)

  # Find the indices of the 3 chunks with highest similarity scores
  top_indices = torch.topk(similarities, k=3).indices

  top_chunks = []

  for i in top_indices:
    chunk = text_chunks[i]
    top_chunks.append(chunk)

  return top_chunks

def respond(message, history, profile):
    if isinstance(message,dict):
        text_message = message.get("text","")
        uploaded_files = message.get("files",[])
    else:
        text_message = message
        uploaded_files = []
    top_results = get_top_chunks(text_message, chunk_embeddings, cleaned_chunks)
    context = "Here is some context: " + " ".join(top_results)

    user_info = ""

    if profile:
        user_info = f"""
The user's skincare profile:
Name: {profile.get('name')}
Skin type: {profile.get('skin_type')}
Age range: {profile.get('age')}
Sensitivity: {profile.get('sensitivity')}
Use this information to personalize your advice.
"""
    context += user_info

    messages = [{"role": "system", "content": """Nora - Your Skincare Big Sister
You are Nora, a friendly skincare assistant who explains skincare like a caring older sister.
Rules:
Be kind and never shame the user.
Give evidence-based advice. 
Recommend patch testing.
Introduce one new active at a time.
Daily SPF 30+ is essential.
Always answer either based of the knowledge.txt file or certified sources that you should mention anytime.
An exemple of a simple interaction with Nora:
Nora : Heyyy! How can I help you today?🩵
User : I have a breakout on my chin and I have a big event tomorrow :((( How can I remove the redness as QUICK as possible?
Nora : Hmm…that seems like inflamed acne. I recommend using a cold compress to calm it down, then treat it with salicylic acid. Would you like some product recommendations?"""}]
    
    user_content = []
    
    if uploaded_files:
        image_file = uploaded_files[0]

        # Get the image path
        if isinstance(image_file, dict):
            image_path = image_file.get("path")
        elif hasattr(image_file, "path"):
            image_path = image_file.path
        else:
            image_path = str(image_file)

        image_url = image_to_data_url(image_path)

        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        })
    

    messages.append({"role": "user", "content": user_content})

    prompt = f"{context}\n\nUser Question: {text_message}"
    user_content.append({
        "type": "text",
        "text": prompt
    })
    
    if history:
        messages.extend(history)


    response = client.chat_completion(
        messages,
        max_tokens=600,
        temperature=.8
    )

    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
:root {
    --nora-light-blue:#bfd9ef;
    --nora-blue: #a2bee3;
    --nora-navy: #173250;
    --nora-cream:#fffbf3;
}
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;600;700&family=Nunito:wght@400;600;700&display=swap');
html, body {
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100%;
    overflow-x: hidden;
}
.gradio-container {
    font-family: 'Nunito', sans-serif !important;
    margin: 0 auto !important;
    min-height: 100dvh !important;
    width: 100% !important;
    max-width: 100% !important;
    padding: 0 !important;
}
.nora-screen {
    background: linear-gradient(180deg, var(--nora-cream) 0%, var(--nora-cream) 35%, var(--nora-blue) 100%);
    border-radius: 28px;
    min-height: 100dvh;
    height: auto;
    width: 100%;
    box-sizing: border-box;
    padding: clamp(72px, 15vw, 144px) clamp(16px, 4vw, 32px);
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    overflow-x: hidden;
}
.nora-screen > * {
    width: 100%;
    max-width: 600px;
}
.nora-avatar {
    width: 92px;
    height: 92px;
    border-radius: 50%;
    background: #c9d9ee;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    margin: 0 auto 20px;
    box-shadow: inset 0 0 0 3px rgba(37, 60, 120, 0.08);
    padding: 40px 20px;
}
.nora-avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.nora-heading {
    font-family: 'Baloo 2', sans-serif !important;
    color: var(--nora-navy) !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    margin-bottom: 24px !important;
}

.nora-name-input,
.nora-name-input * {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.nora-name-input textarea,
.nora-name-input input {
    background: var(--nora-light-blue) !important;
    color: white !important;
    border: none !important;
    text-align: center !important;
    font-size: 16px !important;
    padding: 14px 20px !important;
    box-sizing: border-box;
    border-radius: 999px !important;
}
.nora-name-input textarea::placeholder,
.nora-name-input input::placeholder {
    color: rgba(255, 255, 255, 0.85) !important;
}
.nora-survey-card {
    background: var(--nora-blue) !important;
    border-radius: 22px !important;
    padding: 14px 18px !important;
    width: 100%;
    box-sizing: border-box;
}
.nora-survey-card .nora-card-title {
    font-family: 'Baloo 2', sans-serif !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    margin-bottom: 10px !important;
}
.nora-survey-card label,
.nora-survey-card label * {
    color: var(--nora-navy) !important;
}
.nora-survey-card input[type="radio"] {
    accent-color: var(--nora-navy) !important;
}
.nora-cta {
    background: var(--nora-navy) !important;
    color: white !important;
    border: none !important;
    border-radius: 999px !important;
    width: auto !important;
    padding: 10px 28px !important;
    font-family: 'Baloo 2', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    margin: 24px auto 0 !important;
    display: block;
    box-shadow: 0 4px 12px rgba(37, 60, 120, 0.25);
}
.nora-cta:hover {
    opacity: 0.92;
}
.nora-logo-img {
    background: var(--nora-light-blue) !important;
    padding: 14px 0 !important;
    border-radius: 20px 20px 0 0;
    width: 100% !important;
}
.nora-logo-img img {
    width: 160px !important;
    height: auto !important;
    display: block;
    margin: 0 auto;
}
.gr-chatbot,
.chatbot,
.bubble-wrap {
    background: var(--nora-cream) !important;
}
textarea {
    background: var(--nora-cream) !important;
    border-radius: 999px !important;
    border: none !important;
    padding: 12px 20px !important;
    color: var(--nora-navy) !important;
}
footer {
    display: none !important;
}
.examples-container .icon,
.examples-container svg,
div[class*="example"] span:has(svg),
.example-btn span:first-child {
    display: none !important;
}
.examples-container button,
div[class*="example"] button {
    background-color: var(--nora-cream) !important;
    color: var(--nora-navy) !important;
    border-radius: 16px !important;
    border: 1px solid var(--nora-blue) !important;
    text-align: center !important;
    justify-content: center !important;
    align-items: center !important;
    display: flex !important;
    padding: 12px 16px !important;
    transition: all 0.2s ease-in-out !important;
}

.examples-container button *,
div[class*="example"] button * {
    text-align: center !important;
    width: 100% !important;
}

.examples-container button:hover,
div[class*="example"] button:hover {
    background-color: var(--nora-blue) !important;
    color: var(--nora-navy) !important;
    border-color: var(--nora-blue) !important;
}

.example-icon,
.text-icon-aa {
    display: none !important;

@media (max-width: 600px) {
    .nora-screen {
        border-radius: 20px;
        padding: 24px 16px;
    }
    .nora-avatar {
        width: 76px;
        height: 76px;
    }
    .nora-heading {
        font-size: 22px !important;
    }
    .nora-survey-card {
        padding: 18px 16px !important;
    }
}
"""
with gr.Blocks(css=CSS, title="Nora - Skincare Big Sister") as onboarding_demo:
    user_name = gr.State("")
    user_profile = gr.State({})
    with gr.Column(visible=True, elem_classes=["nora-screen"]) as screen_name:
        gr.Image(
            value="nora.png",
            show_label=False,
            interactive=False,
            container=False,
            elem_classes=["nora-avatar"]
        )
        gr.HTML('<div class="nora-heading">What do I call you?</div>')
        name_input = gr.Textbox(
            placeholder="Just call me . . .",
            show_label=False,
            container=False,
            elem_classes=["nora-name-input"]
        )
        name_btn = gr.Button("Let's go!", elem_classes=["nora-cta"])
    with gr.Column(visible=False, elem_classes=["nora-screen"]) as screen_survey:
        gr.Image(
            value="nora.png",
            show_label=False,
            interactive=False,
            container=False,
            elem_classes=["nora-avatar"]
        )
        gr.HTML('<div class="nora-heading">Quick survey!!</div>')
        with gr.Column(elem_classes=["nora-survey-card"]):
            gr.HTML('<div class="nora-card-title">My Skin Type</div>')
            skin_input = gr.Radio(
                choices=["Oily", "Dry", "Combination"],
                show_label=False,
                container=False
            )
            gr.HTML('<div class="nora-card-title">My Age Range</div>')
            age_input = gr.Radio(
                choices=["12–17", "18–24", "25–29", "30–39", "40–49", "50-59", "60+"],
                show_label=False,
                container=False
            )
            gr.HTML('<div class="nora-card-title">My Sensitivity</div>')
            sensitivity_input = gr.Radio(
                choices=["Sensitive", "Not sensitive"],
                show_label=False,
                container=False
            )
        survey_btn = gr.Button("Let's go!", elem_classes=["nora-cta"])
    with gr.Column(visible=False) as screen_chat:
        gr.Image(
            value="heading.png",
            show_label=False,
            interactive=False,
            container=False,
            elem_classes=["nora-logo-img"]
        )
        chatbot = gr.ChatInterface(
            fn=respond,
            additional_inputs=[user_profile],
            multimodal=True,
            textbox=gr.MultimodalTextbox(
                file_types=["image"],
                file_count="single",
                placeholder="What seems to be the problem, sis?"
            ),
            examples=[
                ["How do I deal with clogged pores?"], ["Give me a skincare routine based on my concerns."], ["How do I know if my skin barrier is damaged?"]
            ]
        )
    def go_to_survey(name):
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            name
        )
    def go_to_chat(name, skin_type, age, sensitivity):
        profile = {
            "name": name,
            "skin_type": skin_type,
            "age": age,
            "sensitivity": sensitivity
        }
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            profile
        )
    name_btn.click(
        go_to_survey,
        inputs=[name_input],
        outputs=[screen_name, screen_survey, user_name]
    )
    survey_btn.click(
        go_to_chat,
        inputs=[
            user_name,
            skin_input,
            age_input,
            sensitivity_input
        ],
        outputs=[
            screen_survey,
            screen_chat,
            user_profile
        ]
    )

onboarding_demo.launch()

# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!