from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import threading

device = "cpu"
print(f"Using device: {device}")

# Tải tokenizer
tokenizer = AutoTokenizer.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)

app = Flask(__name__)

# API để sinh văn bản
@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_length = data.get('max_length', 5000)
        temperature = data.get('temperature', 0.7)

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        inputs = tokenizer.encode(prompt, return_tensors="pt")        
        # Sinh văn bản
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95,
            top_k=4,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1  # Trả về 1 chuỗi đầu ra
        )        
        # Giải mã kết quả đầu ra
        return tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except Exception as e:
        print(f"Error generating text: {e}")
        return jsonify({"error": "An error occurred while generating text"}), 500

# Chạy Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))  # Port mặc định 5001
    app.run(host='0.0.0.0', port=port)
