import streamlit as st
import requests

# FastAPI Backend URL
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG AI Assistant", layout="centered")
st.title("🤖 RAG Document Assistant")

# Sidebar: File Upload
with st.sidebar:
    st.header("Ingest Documents")
    uploaded_file = st.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"])
    if st.button("Upload & Ingest"):
        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            with st.spinner("Processing..."):
                response = requests.post(f"{BACKEND_URL}/ingest", files=files)
                if response.status_code == 200:
                    st.success("Ingestion successful!")
                else:
                    st.error(f"Error: {response.json().get('detail')}")
        else:
            st.warning("Please select a file first.")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your docs:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(f"{BACKEND_URL}/chat", json={"question": prompt})
                data = response.json()
                answer = data.get("answer", "No response received.")
                st.markdown(answer)
                
                # Display sources if available
                if "sources" in data:
                    with st.expander("View Sources"):
                        st.write(data["sources"])
                        
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("Backend not reachable. Ensure FastAPI is running.")