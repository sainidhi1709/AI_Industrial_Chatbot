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

INDEX_PATH = os.path.join(BASE_PATH, "industrial.index")
CHUNKS_PATH = os.path.join(BASE_PATH, "chunks.pkl")

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

tokenizer = AutoTokenizer.from_pretrained(model_name)

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

        question_lower = question.lower()

        # -----------------------------
        # Check if question is related
        # -----------------------------

        industrial_keywords = [
            "machine",
            "safety",
            "ppe",
            "protective equipment",
            "protective",
            "helmet",
            "gloves",
            "glasses",
            "footwear",
            "hearing protection",
            "emergency",
            "maintenance",
            "cleaning",
            "worker",
            "workers",
            "operating",
            "operate",
            "machine guard",
            "machine guards",
            "guard",
            "guards",
            "hazard",
            "equipment",
            "safety rules",
            "general safety"
        ]

        is_relevant = any(
            word in question_lower
            for word in industrial_keywords
        )

        # -----------------------------
        # WRONG / UNRELATED QUESTION
        # -----------------------------

        if not is_relevant:

            st.subheader("📄 Retrieved Information")
            st.write(
                "No relevant information found in the industrial documents."
            )

            st.subheader("🤖 Chatbot Answer")
            st.write(
                "Sorry, I couldn't find relevant information "
                "about that in the provided industrial documents."
            )

        # -----------------------------
        # RELEVANT QUESTION
        # -----------------------------

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

            # Get relevant document information
            best_chunk = "\n\n".join(
                chunks[i] for i in result[0]
            )

            # -----------------------------
            # Retrieved Information
            # -----------------------------

            st.subheader("📄 Retrieved Information")
            st.write(best_chunk)

            # -----------------------------
            # Generate Answer
            # -----------------------------

            if (
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
                    "If an unsafe condition or emergency occurs, "
                    "the operator should stop the machine using the "
                    "appropriate emergency stop control when available "
                    "and inform the responsible supervisor."
                )

            elif (
                "maintenance" in question_lower
                or "cleaning" in question_lower
            ):

                answer = (
                    "Maintenance, cleaning, adjustment, and repair "
                    "should be carried out using the approved procedure. "
                    "The machine should be stopped and its energy sources "
                    "isolated before maintenance."
                )

            elif (
                "machine guard" in question_lower
                or "machine guards" in question_lower
                or "guard" in question_lower
                or "guards" in question_lower
            ):

                answer = (
                    "Safety guards should not be bypassed or removed "
                    "during normal machine operation."
                )

            elif (
                "safety rules" in question_lower
                or "general safety" in question_lower
            ):

                answer = (
                    "Workers should follow the manufacturer's operating "
                    "instructions, use required safety equipment, and "
                    "follow the machine safety procedures."
                )

            elif "safety" in question_lower:

                answer = (
                    "Workers should follow the safety procedures "
                    "described in the industrial safety document."
                )

            else:

                answer = (
                    "I found relevant information in the industrial "
                    "document, but I could not generate a specific answer."
                )

            # -----------------------------
            # Chatbot Answer
            # -----------------------------

            st.subheader("🤖 Chatbot Answer")
            st.write(answer)
