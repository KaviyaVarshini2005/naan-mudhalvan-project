from transformers import pipeline
import gradio as gr

# Load a pre-trained transformer model for question answering
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")

# Function to generate chatbot responses
def respond_to_input(user_input, history=[]):
    from transformers import ConversationalPipeline, Conversation
    conversation = Conversation(user_input)
    chatbot(conversation)
    return str(conversation.generated_responses[-1])

# Gradio interface
iface = gr.Interface(
    fn=respond_to_input,
    inputs="text",
    outputs="text",
    title="AI Customer Support Chatbot",
    description="Ask a question and get automated intelligent responses.",
    theme="default"
)

# Launch the chatbot
iface.launch()
