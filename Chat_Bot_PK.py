import streamlit as st
import os
import time
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

# Set page configuration for improved UI
st.set_page_config(page_title="Chatbot By Pk", layout="wide")

# --- Credentials and Client Initialization ---
AZURE_API_KEY = "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"  # Replace with your actual key
AZURE_ENDPOINT = "https://models.inference.ai.azure.com"

client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_API_KEY)
)

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- UI Styling ---
st.markdown("""
    <style>
    .user-message {
        background-color: #e1f5fe;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .assistant-message {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Main Chat Interface ---
st.title("Chatbot By Pk")

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message"><strong>You:</strong> {msg["content"]}</div>', 
                       unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):
            st.markdown(f'<div class="assistant-message"><strong>Pk:</strong> {msg["content"]}</div>', 
                       unsafe_allow_html=True)

# --- Chat Input ---
user_input = st.chat_input("Ask me anything!")

if user_input:
    # Immediately display the user's question
    with st.chat_message("user"):
        st.markdown(f'<div class="user-message"><strong>You:</strong> {user_input}</div>', 
                   unsafe_allow_html=True)
    
    # Add user message to history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Display assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        retries = 3
        delay = 1  # seconds
        success = False
        
        # Retry loop to handle rate limit errors
        for attempt in range(retries):
            try:
                # Using streaming if supported
                response = client.complete(
                    messages=[
                        SystemMessage(
                            "You are Pk, a friendly and knowledgeable mechanical engineer. "
                            "You speak in a helpful, casual tone, like you're chatting with a colleague. "
                            "You explain things clearly using real-world mechanical examples where possible."
                        ),
                        UserMessage(user_input)
                    ],
                    model="DeepSeek-V3",  # Adjust if needed
                    temperature=0.8,
                    max_tokens=512,
                    top_p=0.1,
                    stream=True
                )
                
                for chunk in response:
                    if chunk.choices:
                        delta = getattr(chunk.choices[0], "delta", None)
                        part = delta.get("content", "") if delta else ""
                        full_response += part
                        placeholder.markdown(f'<div class="assistant-message"><strong>Pk:</strong> {full_response}</div>', 
                                          unsafe_allow_html=True)
                
                success = True
                break  # Exit loop on success
            
            except HttpResponseError as e:
                if "Rate limit" in str(e):
                    st.warning(f"Rate limit exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                    time.sleep(delay)
                    delay *= 2
                else:
                    st.error(f"An error occurred: {e}")
                    break
        
        if not success:
            st.error("Failed to get a response after several retries.")
        
        # Add final response to history
        st.session_state["messages"].append({"role": "assistant", "content": full_response})
