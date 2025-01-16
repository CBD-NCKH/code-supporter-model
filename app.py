from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Khởi tạo Flask app
app = Flask(__name__)

# Cấu hình thiết bị và mô hình
device = "cpu"
checkpoint = "bigcode/starcoder"
auth_token = os.getenv("MODEL_KEY")

# Tải Tokenizer và Mô hình
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, token=auth_token)

print("Loading model...")
model = torch.quantization.quantize_dynamic(
    model=AutoModelForCausalLM.from_pretrained(checkpoint, token=auth_token),
    qconfig_spec={torch.nn.Linear},
    dtype=torch.qint8
)
model.to(device)
print("Model loaded successfully.")

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
