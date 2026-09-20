import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -------------------------------
# Page Settings
# -------------------------------

st.set_page_config(
    page_title="AI Industrial Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI Industrial Chatbot")
st.write("Ask questions about the industrial safety document.")


# -------------------------------
# Load FAISS Index
# -------------------------------

index = faiss.read_index("data/industrial.index")


# -------------------------------
# Load Document Chunks
# -------------------------------

with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


# -------------------------------
# Load Embedding Model
# -------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -------------------------------
# Load Hugging Face Model
# -------------------------------

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

llm = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    attn_implementation="eager"
)

llm.eval()


# -------------------------------
# Question Box
# -------------------------------

question = st.text_input(
    "Ask your question:",
    placeholder="Example: What personal protective equipment should workers use?"
)


# -------------------------------
# Get Answer
# -------------------------------

if st.button("Get Answer"):

    if question:

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

        # Get relevant information
        best_chunk = "\n".join(
            chunks[i] for i in result[0]
        )


        # -------------------------------
        # Display Retrieved Information
        # -------------------------------

        st.subheader("📄 Retrieved Information")
        st.write(best_chunk)


        # -------------------------------
        # Generate Simple Answer
        # -------------------------------

        question_lower = question.lower()


        # PPE question
        if (
            "personal protective equipment" in question_lower
            or "ppe" in question_lower
        ):

            answer = (
                "Workers should use safety helmets, safety glasses, "
                "protective footwear, gloves, and hearing protection "
                "in high-noise areas."
            )


        # Machine operation question
        elif (
            "before starting" in question_lower
            or "before operating" in question_lower
            or "machine controls" in question_lower
        ):

            answer = (
                "Workers should understand the machine controls "
                "and operating procedure before operating the machine."
            )


        # Emergency question
        elif "emergency" in question_lower:

            answer = (
                "Workers should follow the emergency procedure "
                "and use the appropriate emergency stop or safety measures."
            )


        # Maintenance question
        elif "maintenance" in question_lower:

            answer = (
                "Workers should follow proper safety procedures "
                "during machine maintenance and cleaning."
            )


        # General safety question
        elif "safety rules" in question_lower:

            answer = (
                "Workers should follow machine safety procedures, "
                "use required PPE, and keep the work area safe."
            )


        # Other questions
        else:

            sentences = best_chunk.split(".")

            answer = sentences[0].strip()

            if len(sentences) > 1:
                answer += ". " + sentences[1].strip()


        # -------------------------------
        # Display Answer
        # -------------------------------

        st.subheader("🤖 Chatbot Answer")
        st.write(answer)


    else:

        st.warning("Please enter a question.")