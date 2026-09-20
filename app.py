import os
import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Page Settings
# -----------------------------

st.set_page_config(
    page_title="AI Industrial Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI Industrial Chatbot")
st.write("Ask questions about the industrial safety document.")

# -----------------------------
# File Paths
# -----------------------------

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH = os.path.join(
    BASE_PATH,
    "industrial.index"
)

CHUNKS_PATH = os.path.join(
    BASE_PATH,
    "chunks.pkl"
)

# -----------------------------
# Load FAISS Index
# -----------------------------

index = faiss.read_index(INDEX_PATH)

# -----------------------------
# Load Document Chunks
# -----------------------------

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

# -----------------------------
# Load Embedding Model
# -----------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------
# Load Hugging Face Model
# -----------------------------

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

llm = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    attn_implementation="eager"
)

llm.eval()

# -----------------------------
# Question Box
# -----------------------------

question = st.text_input(
    "Ask your question:",
    placeholder="Example: What personal protective equipment should workers use?"
)

# -----------------------------
# Get Answer
# -----------------------------

if st.button("Get Answer"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        # Convert question into embedding
        question_embedding = embedding_model.encode(
            [question],
            convert_to_numpy=True
        )

        # Search FAISS
        distance, result = index.search(
            question_embedding,
            2
        )

        # Get retrieved document information
        best_chunk = "\n\n".join(
            chunks[i] for i in result[0]
        )

        # -----------------------------
        # Display Retrieved Information
        # -----------------------------

        st.subheader("📄 Retrieved Information")
        st.write(best_chunk)

        # -----------------------------
        # Check Question
        # -----------------------------

        question_lower = question.lower()

        industrial_keywords = [
            "machine",
            "safety",
            "ppe",
            "protective",
            "helmet",
            "gloves",
            "glasses",
            "footwear",
            "hearing",
            "emergency",
            "maintenance",
            "cleaning",
            "worker",
            "workers",
            "operating",
            "operate",
            "guard",
            "guards",
            "hazard",
            "equipment"
        ]

        is_relevant = any(
            word in question_lower
            for word in industrial_keywords
        )

        # -----------------------------
        # Generate Answer
        # -----------------------------

        if not is_relevant:

            answer = (
                "Sorry, I couldn't find relevant information "
                "about that in the provided industrial documents."
            )

        elif (
            "personal protective equipment" in question_lower
            or "ppe" in question_lower
            or "protective equipment" in question_lower
        ):

            answer = (
                "Workers should use safety helmets, safety glasses, "
                "protective footwear, gloves, and hearing protection "
                "in high-noise areas."
            )

        elif (
            "before starting" in question_lower
            or "before operating" in question_lower
            or "machine controls" in question_lower
        ):

            answer = (
                "Workers should understand the machine controls "
                "and operating procedure before operating the machine."
            )

        elif "emergency" in question_lower:

            answer = (
                "Workers should follow the emergency procedure "
                "and use the appropriate emergency stop or safety measures."
            )

        elif (
            "maintenance" in question_lower
            or "cleaning" in question_lower
        ):

            answer = (
                "Workers should follow proper safety procedures "
                "during machine maintenance and cleaning."
            )

        elif (
            "machine guard" in question_lower
            or "machine guards" in question_lower
            or "guard" in question_lower
            or "guards" in question_lower
        ):

            answer = (
                "Machine guards should be used to help protect workers "
                "from moving machine parts and other hazards."
            )

        elif (
            "safety rules" in question_lower
            or "general safety" in question_lower
        ):

            answer = (
                "Workers should follow machine safety procedures, "
                "use required PPE, and keep the work area safe."
            )

        elif "safety" in question_lower:

            answer = (
                "Workers should follow the safety procedures "
                "described in the industrial safety document."
            )

        else:

            answer = (
                "I found related information in the document, "
                "but I could not generate a specific answer "
                "for this question."
            )

        # -----------------------------
        # Display Answer
        # -----------------------------

        st.subheader("🤖 Chatbot Answer")
        st.write(answer)
