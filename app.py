import streamlit as st
import requests
import os

# Page config
st.set_page_config(page_title="RAG Document Chatbot", layout="wide")

# Sidebar for API configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input(
        "API URL",
        value="http://localhost:8000",
        help="FastAPI backend URL"
    )
    st.markdown("---")
    
    if st.button("🔄 Ingest Documents", use_container_width=True):
        try:
            with st.spinner("Ingesting documents..."):
                response = requests.post(f"{api_url}/ingest", timeout=60)
                response.raise_for_status()
                data = response.json()
                st.success(f"✅ {data['status']}")
                st.info(f"📊 Total chunks in database: {data['chunks_found']}")
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Cannot connect to API at {api_url}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error: {str(e)}")

# Main content
st.title("📚 RAG Document Chatbot")
st.markdown("Ask questions about your documents. The AI will search for relevant context and answer based on it.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
query = st.chat_input("Ask a question about your documents...")

if query:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Get response from FastAPI
    try:
        with st.spinner("🤔 Thinking..."):
            response = requests.post(
                f"{api_url}/chat",
                json={"question": query},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            answer = data["answer"]
            sources = data.get("sources", [])
            
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                # Show sources
                if sources:
                    with st.expander("📖 Sources"):
                        for source in sources:
                            st.caption(source)
    
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {api_url}. Make sure FastAPI is running.")
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. The API took too long to respond.")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {str(e)}")
    except KeyError as e:
        st.error(f"❌ Unexpected response format from API: {str(e)}")