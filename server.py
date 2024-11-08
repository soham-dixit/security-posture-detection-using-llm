from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

model_path = "../model/security_posture_gpt2/"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

few_shot_examples = [
    {
        "input": "id: 1, dur: 0.121478, proto: tcp, service: -, state: FIN, spkts: 6, dpkts: 4, sbytes: 258, dbytes: 172, rate: 74.087490, ct_dst_sport_ltm: 1, ct_dst_src_ltm: 1, is_ftp_login: 0, ct_ftp_cmd: 0, ct_flw_http_mthd: 0, ct_src_ltm: 1, ct_srv_dst: 1, is_sm_ips_ports: 0, attack_cat: Normal, label: 0",
        "output": "This log shows a brief TCP connection with a FIN state, indicating normal termination of communication between endpoints. It has a low data transfer rate, typical for non-attack behavior."
    },
]

def format_prompt(log_entry):
    few_shots = "\n".join([
        f"Input: {ex['input']}\nTarget: {ex['output']}" for ex in few_shot_examples
    ])
    return f"{few_shots}\nInput: {log_entry}\nOutput:"

def generate_analysis(log_entry):
    prompt = format_prompt(log_entry)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

    # Generate the model's response
    output_ids = model.generate(
        inputs["input_ids"],
        max_length=512,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True, 
        temperature=0.5,
        top_p=0.9, 
    )

    # Decode and format output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_response = output_text.split("Output:")[-1].strip().split("Input:")[0]
    return generated_response.strip()

@app.route('/api/threats', methods=['POST'])
def receive_threat():
    # data = request.json
    # # Process the threat data with the LLM
    # # Example: Analyze data['rule']['description'] with the LLM
    # print(data)
    # response = analyze_threat_with_llm(data)
    # print(response)
    # return response

    data = request.json
    if "rule" not in data or "description" not in data["rule"]:
        return jsonify({"error": "Invalid input format"}), 400

    log_entry = data["rule"]["description"]
    response = generate_analysis(log_entry)
    print(response)
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
