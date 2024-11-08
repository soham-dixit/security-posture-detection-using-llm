import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the trained model and tokenizer
model_path = "./security_posture_gpt2"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Move the model to the appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Few-shot examples
few_shot_examples = [
    {
        "input": "id: 1, dur: 0.121478, proto: tcp, service: -, state: FIN, spkts: 6, dpkts: 4, sbytes: 258, dbytes: 172, rate: 74.087490, ct_dst_sport_ltm: 1, ct_dst_src_ltm: 1, is_ftp_login: 0, ct_ftp_cmd: 0, ct_flw_http_mthd: 0, ct_src_ltm: 1, ct_srv_dst: 1, is_sm_ips_ports: 0, attack_cat: Normal, label: 0",
        "output": "This log shows a brief TCP connection with a FIN state, indicating normal termination of communication between endpoints. It has a low data transfer rate, typical for non-attack behavior."
    },
]

# Function to format the prompt with few-shot examples
def format_prompt(log_entry):
    few_shots = "\n".join([
        f"Input: {ex['input']}\nTarget: {ex['output']}" for ex in few_shot_examples
    ])
    return f"Input: {log_entry}\nTarget:"

# Test function
def generate_analysis(log_entry):
    prompt = format_prompt(log_entry)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

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
    generated_response = output_text.split("Target:")[-1].strip().split("Input:")[0]
    return generated_response.strip()

# Example custom input log entry
custom_log_entry = "id: 49000, dur: 0.000009, proto: encap, service: -, state: INT, spkts: 2, dpkts: 0, sbytes: 200, dbytes: 0, rate: 111111.1072, ct_dst_sport_ltm: 1, ct_dst_src_ltm: 2, is_ftp_login: 0, ct_ftp_cmd: 0, ct_flw_http_mthd: 0, ct_src_ltm: 2, ct_srv_dst: 2, is_sm_ips_ports: 0, attack_cat: DoS, label: 1"

# Generate and print the analysis for the custom log entry
analysis = generate_analysis(custom_log_entry)
print("Generated Analysis:", analysis)
