from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Tải tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint, token=auth_token)

# Hàm tải mô hình
def load_model():
    global model
    if model is None:  # Kiểm tra tránh tải lại
        print("Loading model with quantization...")
        model = torch.quantization.quantize_dynamic(
            model=AutoModelForCausalLM.from_pretrained(checkpoint, token=auth_token),
            qconfig_spec={torch.nn.Linear},
            dtype=torch.qint8
        )
        print("Model loaded successfully.")

model = None
# Tải mô hình trong luồng riêng
threading.Thread(target=load_model).start()

app = Flask(__name__)

# API để sinh văn bản
@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_length = data.get('max_length', 500)
        temperature = data.get('temperature', 0.7)

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"generated_text": generated_text})
    except Exception as e:
        print(f"Error generating text: {e}")
        return jsonify({"error": "An error occurred while generating text"}), 500

# Chạy Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))  # Port mặc định 5001
    app.run(host='0.0.0.0', port=port)
