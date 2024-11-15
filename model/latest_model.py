import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
from huggingface_hub import login

login(token = "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")

# Load the sample CSV data
data_path = './UNSW_NB15_training-set.csv'
data = pd.read_csv(data_path)

# Define few-shot examples (prompt-template based on provided example)
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

relevant_columns = {
    "Normal": ["dur", "proto", "state", "spkts", "dpkts", "sbytes", "dbytes"],
    "Backdoor": ["dur", "proto", "service", "spkts", "dpkts", "sbytes", "dbytes"],
    "Analysis": ["proto", "dur", "spkts", "dpkts"],
    "Fuzzers": ["dur", "proto", "state", "spkts", "dpkts", "sbytes", "dbytes"],
    "Shellcode": ["proto", "dur", "spkts", "dpkts", "sbytes", "dbytes"],
    "Reconnaissance": ["proto", "dur", "spkts", "dpkts"],
    "Exploits": ["dur", "proto", "service", "spkts", "dpkts", "sbytes", "dbytes"],
    "DoS": ["proto", "dur", "sbytes", "dbytes"],
    "Worms": ["proto", "dur", "spkts", "dpkts", "sbytes", "dbytes"],
    "Generic": ["proto", "dur", "service", "spkts", "dpkts"],
}

analysis_templates = {
    "Normal": "This log entry indicates normal, non-suspicious activity with no signs of threats.",
    "Fuzzers": "This log suggests a potential fuzzing attack, where random data is sent to the system to find vulnerabilities.",
    "Reconnaissance": "This log shows reconnaissance activity, likely involving scanning or probing to gather system information.",
    "Shellcode": "This entry indicates possible shellcode execution, which may allow an attacker to take control of the system.",
    "Analysis": "This log suggests suspicious analysis activity, possibly indicating unauthorized system monitoring.",
    "Backdoors": "This log entry shows backdoor activity, which may allow remote access to the system without authorization.",
    "DoS": "This log suggests a Denial of Service (DoS) attack, with high request rates aimed at overwhelming the system.",
    "Exploits": "This entry indicates exploit activity, suggesting attempts to use vulnerabilities in the system for unauthorized access.",
    "Generic": "This log shows activity categorized as 'Generic', likely indicating generic malware behavior.",
    "Worms": "This entry suggests worm activity, which may indicate self-replicating malware attempting to spread across the network."
}

model_name = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Function to format data with few-shot examples
def format_prompt(log_entry):
    few_shots = "\n".join([
        f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in few_shot_examples
    ])
    return f"{few_shots}\nInput: {log_entry}\nOutput:"

def generate_analysis(row):
    return analysis_templates.get(row['attack_cat'], "This log entry indicates suspicious activity that does not clearly match known attack patterns.")

# Prepare the dataset for training
class LogDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        log_entry = format_prompt(self.data.iloc[idx]['log_entry'])
        labels = self.data.iloc[idx]['analysis']  # Ground truth target text for the log

        # Tokenize the input and output together as a single sequence
        encoding = self.tokenizer(log_entry + labels, truncation=True, max_length=self.max_length, padding="max_length")
        encoding["labels"] = encoding["input_ids"]
        return {key: torch.tensor(val) for key, val in encoding.items()}

# Prepare data
data['log_entry'] = data.apply(
    lambda row: ", ".join(
        f"{col}: {row[col]}" for col in relevant_columns.get(row['attack_cat'], [])
    ) + f", attack_cat: {row['attack_cat']}, label: {row['label']}",
    axis=1
)

data['analysis'] = data.apply(generate_analysis, axis=1)

# Convert to Dataset
dataset = LogDataset(data, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results_new",
    eval_strategy="no",
    learning_rate=3e-5,
    weight_decay=0.01,
    per_device_train_batch_size=4,
    num_train_epochs=2,
    save_steps=10000,
    save_total_limit=1,
    logging_dir='./logs',
    fp16=True,
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Start training
trainer.train()

# Save model
model.save_pretrained("./security_posture_llm")
tokenizer.save_pretrained("./security_posture_llm")

print("Few-shot training completed.")

def test_model(input_log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prompt = f"Input: {input_log}\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = model.generate(**inputs, max_length=1024, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

custom_input = "proto: udp, dur: 1.092384, service: dns, spkts: 4, dpkts: 5, attack_cat: Generic, label: 1"

# Run the test
output = test_model(custom_input)
print("Generated Output:\n", output)


