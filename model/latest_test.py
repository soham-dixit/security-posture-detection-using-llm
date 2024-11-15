import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the trained model and tokenizer
model_path = "./security_posture_llm"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ensure pad_token is defined to match model's padding
tokenizer.pad_token = tokenizer.eos_token

few_shot_examples = [
    {
        "input": "dur: 0.121478, proto: tcp, state: FIN, spkts: 6, dpkts: 4, sbytes: 258, dbytes: 172, attack_cat: Normal, label: 0",
        "output": "This log shows a brief, normal TCP connection with low data transfer, ending in a FIN state, indicating routine communication."
    },
    {
        "input": "dur: 3.210123, proto: tcp, service: ssh, spkts: 20, dpkts: 15, sbytes: 5000, dbytes: 3000, attack_cat: Backdoor, label: 1",
        "output": "Potential backdoor activity via SSH with notable data exchange, suggesting unauthorized system access."
    },
    {
        "input": "proto: tcp, dur: 1.821563, spkts: 14, dpkts: 14, attack_cat: Analysis, label: 1",
        "output": "Suspicious network behavior possibly indicative of reconnaissance or probing activity in preparation for an attack."
    },
    {
        "input": "dur: 0.254819, proto: udp, state: CON, spkts: 9, dpkts: 9, sbytes: 1000, dbytes: 1200, attack_cat: Fuzzers, label: 1",
        "output": "Fuzzing activity detected, likely aiming to test for weaknesses by injecting unexpected data."
    },
    {
        "input": "proto: tcp, dur: 0.501287, spkts: 5, dpkts: 3, sbytes: 700, dbytes: 800, attack_cat: Shellcode, label: 1",
        "output": "Possible shellcode activity, indicating an attempt to execute arbitrary code within the target system."
    },
    {
        "input": "proto: icmp, dur: 0.647293, spkts: 15, dpkts: 10, attack_cat: Reconnaissance, label: 1",
        "output": "Reconnaissance attempt through ICMP, potentially scanning for open ports or network vulnerabilities."
    },
    {
        "input": "dur: 0.778914, proto: tcp, service: http, spkts: 7, dpkts: 12, sbytes: 1500, dbytes: 2500, attack_cat: Exploits, label: 1",
        "output": "Exploitation activity via HTTP, where known vulnerabilities are likely targeted for system access."
    },
    {
        "input": "proto: tcp, dur: 2.123456, sbytes: 50000, dbytes: 48000, attack_cat: DoS, label: 1",
        "output": "Denial-of-Service (DoS) attempt indicated by high data volume, aimed at overwhelming the target system."
    },
    {
        "input": "proto: tcp, dur: 3.789012, spkts: 15, dpkts: 20, sbytes: 2500, dbytes: 3000, attack_cat: Worms, label: 1",
        "output": "Potential worm activity, shown by self-replicating behavior typical of network-spreading malware."
    },
    {
        "input": "proto: udp, dur: 1.092384, service: dns, spkts: 4, dpkts: 5, attack_cat: Generic, label: 1",
        "output": "Generic attack pattern observed, possibly using DNS to probe for general vulnerabilities."
    }
]

# Format a custom log entry for testing
def format_test_prompt(log_entry, few_shot_examples):
    few_shots = "\n".join([
        f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in few_shot_examples
    ])
    return f"{few_shots}\nInput: {log_entry}\nOutput:"

# Define the custom log entry
custom_log_entry = "proto: tcp, dur: 0.501287, spkts: 5, dpkts: 3, sbytes: 700, dbytes: 800, attack_cat: Shellcode, label: 1"

# Format the input prompt for the custom log entry
input_prompt = format_test_prompt(custom_log_entry, few_shot_examples)

# Tokenize and generate the response
inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, num_return_sequences=1, temperature=0.7, do_sample=True)

# Decode and print the generated output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(custom_log_entry)
print(generated_text[len(input_prompt):].strip())
