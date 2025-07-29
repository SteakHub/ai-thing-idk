from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = Flask(__name__)

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Simple chat history buffer (in memory, not per-user)
chat_history = []

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history
    data = request.get_json()
    message = data.get("message")

    if not message:
        return jsonify({"error": "Missing 'message' field"}), 400

    # Encode current user input
    new_input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt').to(device)

    # Combine with previous chat history
    input_ids = new_input_ids if not chat_history else torch.cat([chat_history, new_input_ids], dim=-1)

    # Generate response
    output_ids = model.generate(
        input_ids,
        max_new_tokens=40,
        pad_token_id=tokenizer.eos_token_id
    )

    # Update chat history
    chat_history = output_ids

    # Decode and return response
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({"response": response.strip()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
