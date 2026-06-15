import os
import gradio as gr
import numpy as np
import faiss

from groq import Groq
from sentence_transformers import SentenceTransformer

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

with open("knowledge.txt", "r", encoding="utf-8") as f:
    knowledge = f.read()

chunks = [c.strip() for c in knowledge.split("\n\n") if c.strip()]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


def retrieve(query, k=3):

    q_embedding = model.encode([query])
    q_embedding = np.array(q_embedding).astype("float32")

    distances, indices = index.search(
        q_embedding,
        k
    )

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return "\n\n".join(results)


def lbs_chat(message, history):

    context = retrieve(message)

    prompt = f"""
You are LBS Connect.
Answer ONLY using the context below.
If the answer is not found, reply exactly:
Sorry, I don't have that information.
Context:
{context}
Question:
{message}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate"
    ),
    css="""
    footer{
        display:none!important;
    }
    body{
        background:#020817!important;
    }
    .gradio-container{
        max-width:900px!important;
        margin:auto!important;
    }
    h1{
        text-align:center!important;
        font-size:56px!important;
        font-weight:800!important;
        color:white!important;
        margin-bottom:0!important;
    }
    .subtitle{
        text-align:center;
        color:#94a3b8;
        font-size:18px;
        margin-bottom:20px;
    }
    .chatbox{
        border-radius:20px!important;
        border:1px solid rgba(255,255,255,.08)!important;
        background:#08122e!important;
    }
    textarea{
        border-radius:14px!important;
    }
    button{
        border-radius:14px!important;
    }
    .faq-card{
        width:100%!important;
        height:58px!important;
        margin-bottom:12px!important;
        font-size:15px!important;
        font-weight:600!important;
        border-radius:14px!important;
    }
    """
) as demo:

    gr.Markdown("""
    # LBS Connect
    <div class="subtitle">
    College Information Portal
    </div>
    """)

    chatbot = gr.Chatbot(
        [
            {
                "role":"assistant",
                "content":"👋 Welcome to LBS Connect\n\nAsk me about courses, admissions, facilities and campus information."
            }
        ],
        height=180,
        autoscroll=True,
        elem_classes="chatbox"
    )

    faq_group = gr.Group(visible=True)

    with faq_group:

        q1 = gr.Button(
            "What courses are offered at LBSCEK?",
            elem_classes="faq-card"
        )

        q2 = gr.Button(
            "How can I get admission to LBSCEK?",
            elem_classes="faq-card"
        )

        q3 = gr.Button(
             "What facilities are available on campus?",
            elem_classes="faq-card"
        )

    with gr.Row():

        msg = gr.Textbox(
            placeholder="Ask anything...",
            show_label=False,
            container=False,
            scale=8
        )

        send = gr.Button(
            "➜",
            variant="primary",
            scale=2
        )

    def respond(message, history):

        if not message.strip():

            return (
                "",
                history,
                gr.update(),
                gr.update()
            )

        answer = lbs_chat(message, history)

        history.append(
            {
                "role":"user",
                "content":message
            }
        )

        history.append(
            {
                "role":"assistant",
                "content":answer
            }
        )

        return (
            "",
            history,
            gr.update(visible=False),
            gr.update(height=420)
        )

    send.click(
        respond,
        [msg, chatbot],
        [msg, chatbot, faq_group, chatbot]
    )

    msg.submit(
        respond,
        [msg, chatbot],
        [msg, chatbot, faq_group, chatbot]
    )

    def faq_click(question, history):

        answer = lbs_chat(question, history)

        history.append(
            {
                "role":"user",
                "content":question
            }
        )

        history.append(
            {
                "role":"assistant",
                "content":answer
            }
        )

        return (
            history,
            gr.update(visible=False),
            gr.update(height=420)
        )

    q1.click(
        lambda h: faq_click(
            "What courses are offered at LBSCEK?",
            h
        ),
        inputs=[chatbot],
        outputs=[chatbot, faq_group, chatbot]
    )

    q2.click(
        lambda h: faq_click(
            "How can I get admission to LBSCEK?",
            h
        ),
        inputs=[chatbot],
        outputs=[chatbot, faq_group, chatbot]
    )

    q3.click(
        lambda h: faq_click(
            "What facilities are available on campus?",
            h
        ),
        inputs=[chatbot],
        outputs=[chatbot, faq_group, chatbot]
    )

    gr.Markdown("""
    ---
    ### LBS College of Engineering Kasaragod
    🌐 https://lbscek.ac.in
    📞 04994 250290
    🎓 APJ Abdul Kalam Technological University
    """)

demo.queue()
demo.launch()
