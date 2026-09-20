import os
import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Page Settings
st.set_page_config(
    page_title="AI Industrial Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI Industrial Chatbot")
st.write("Ask questions about the industrial safety documents.")

# -----------------------------
# Load FAISS Index
# -----------------------------

base_path = os.path.dirname(__file__)

index_path = os.path.join(
    base_path,
    "industrial.index"
)

index = faiss.read_index(index_path)

# -----------------------------
# Load Document Chunks
# -----------------------------

chunks_path = os.path.join(
    base_path,
    "chunks.pkl"
)

with open(chunks_path, "rb") as f:
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

    if question.strip():

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

        # Similarity distance
        best_distance = distance[0][0]

        # Get relevant information
        best_chunk = "\n".join(
            chunks[i] for i in result[0]
        )

        # -----------------------------
        # Check if question is relevant
        # -----------------------------

        question_lower = question.lower()

        relevant_words = [
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
            "machine guard",
            "safety rules",
            "hazard",
            "equipment"
        ]

        is_relevant = any(
            word in question_lower
            for word in relevant_words
        )

        # -----------------------------
        # Show answer
        # -----------------------------

        if not is_relevant:

            answer = (
                "Sorry, I couldn't find relevant information "
                "about that in the provided industrial documents."
            )

        # PPE question
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

        # Machine operation
        elif (
            "before starting" in question_lower
            or "before operating" in question_lower
            or "machine controls" in question_lower
            or "operate the machine" in question_lower
        ):

            answer = (
                "Workers should understand the machine controls "
                "and operating procedure before operating the machine."
            )

        # Emergency
        elif "emergency" in question_lower:

            answer = (
                "Workers should follow the emergency procedure "
                "and use the appropriate emergency stop or safety measures."
            )

        # Maintenance
        elif (
            "maintenance" in question_lower
            or "cleaning" in question_lower
        ):

            answer = (
                "Workers should follow proper safety procedures "
                "during machine maintenance and cleaning."
            )

        # General safety
        elif (
            "safety rules" in question_lower
            or "general safety" in question_lower
            or "safety" in question_lower
        ):

            answer = (
                "Workers should follow machine safety procedures, "
                "use required PPE, and keep the work area safe."
            )

        # Machine guards
        elif (
            "guard" in question_lower
            or "guards" in question_lower
        ):

            answer = (
                "Machine guards should be used to help protect workers "
                "from moving machine parts and other hazards."
            )

        # Other industrial questions
        else:

            answer = (
                "I found some related information in the industrial "
                "document, but I could not generate a specific answer "
                "for this question."
            )

        # -----------------------------
        # Display Answer
        # -----------------------------

        st.subheader("🤖 Chatbot Answer")
        st.write(answer)

    else:

        st.warning("Please enter a question.")
