import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch

# Load the sample CSV data
data_path = './UNSW_NB15_training-set.csv'
data = pd.read_csv(data_path)

# Define few-shot examples (prompt-template based on provided example)
few_shot_examples = [
    {
        "input": "id: 1, dur: 0.121478, proto: tcp, service: -, state: FIN, spkts: 6, dpkts: 4, sbytes: 258, dbytes: 172, rate: 74.087490, ct_dst_sport_ltm: 1, ct_dst_src_ltm: 1, is_ftp_login: 0, ct_ftp_cmd: 0, ct_flw_http_mthd: 0, ct_src_ltm: 1, ct_srv_dst: 1, is_sm_ips_ports: 0, attack_cat: Normal, label: 0",
        "output": "This log shows a brief TCP connection with a FIN state, indicating normal termination of communication between endpoints. It has a low data transfer rate, typical for non-attack behavior."
    },
    {
        "input": "id: 2, dur: 0.649902, proto: tcp, service: -, state: FIN, spkts: 14, dpkts: 38, sbytes: 734, dbytes: 42014, rate: 78.473372, ct_dst_sport_ltm: 1, ct_dst_src_ltm: 2, is_ftp_login: 0, ct_ftp_cmd: 0, ct_flw_http_mthd: 0, ct_src_ltm: 1, ct_srv_dst: 6, is_sm_ips_ports: 0, attack_cat: Normal, label: 0",
        "output": "This log entry shows a TCP connection with a FIN state and a high volume of data received, possibly indicating a large data transfer. However, the normal status suggests no indication of malicious activity."
    },
    {
        "input": "id: 3, dur: 1.623129, proto: tcp, service: -, state: FIN, spkts: 8, dpkts: 16, sbytes: 364, dbytes: 13186, rate: 14.170161, ct_dst_sport_ltm: 1, ct_dst_src_ltm: 3, is_ftp_login: 0, ct_ftp_cmd: 0, ct_flw_http_mthd: 0, ct_src_ltm: 2, ct_srv_dst: 6, is_sm_ips_ports: 0, attack_cat: Normal, label: 0",
        "output": "This entry shows a prolonged TCP session with modest data transfer, terminated normally. The low rate and data volume indicate routine communication with no threat detected."
    },
    {
        "input": "id: 4, dur: 1.681642, proto: tcp, service: ftp, state: FIN, spkts: 12, dpkts: 12, sbytes: 628, dbytes: 770, rate: 13.677108, ct_dst_sport_ltm: 1, ct_dst_src_ltm: 3, is_ftp_login: 1, ct_ftp_cmd: 1, ct_flw_http_mthd: 0, ct_src_ltm: 2, ct_srv_dst: 1, is_sm_ips_ports: 0, attack_cat: Normal, label: 0",
        "output": "This log indicates an FTP session, as shown by the ftp service and the is_ftp_login flag set to 1. The session had moderate data transfer and terminated normally, with no abnormal behavior flagged."
    },
    {
        "input": "id: 5, dur: 0.449454, proto: tcp, service: -, state: FIN, spkts: 10, dpkts: 6, sbytes: 534, dbytes: 268, rate: 33.373826, ct_dst_sport_ltm: 1, ct_dst_src_ltm: 40, is_ftp_login: 0, ct_ftp_cmd: 0, ct_flw_http_mthd: 0, ct_src_ltm: 2, ct_srv_dst: 39, is_sm_ips_ports: 0, attack_cat: Normal, label: 0",
        "output": "This log shows a short TCP session with typical data exchange, ending with a FIN state. Despite high ct_dst_src_ltm, it remains labeled as normal without indications of threat activity."
    }
]

# Load model and tokenizer (use a lightweight model like `distilgpt2` or LLaMA if you have it)
model_name = "distilgpt2" # Replace with lightweight LLaMA or similar if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Function to format data with few-shot examples
def format_prompt(log_entry):
    few_shots = "\n".join([
        f"Input: {ex['input']}\nTarget: {ex['output']}" for ex in few_shot_examples
    ])
    return f"{few_shots}\nInput: {log_entry}\nTarget:"

# Prepare the dataset for training
class LogDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512):
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
data['log_entry'] = data.apply(lambda row: f"id: {row['id']}, dur: {row['dur']}, proto: {row['proto']}, service: {row['service']}, state: {row['state']}, spkts: {row['spkts']}, dpkts: {row['dpkts']}, sbytes: {row['sbytes']}, dbytes: {row['dbytes']}, rate: {row['rate']}, ct_dst_sport_ltm: {row['ct_dst_sport_ltm']}, ct_dst_src_ltm: {row['ct_dst_src_ltm']}, is_ftp_login: {row['is_ftp_login']}, ct_ftp_cmd: {row['ct_ftp_cmd']}, ct_flw_http_mthd: {row['ct_flw_http_mthd']}, ct_src_ltm: {row['ct_src_ltm']}, ct_srv_dst: {row['ct_srv_dst']}, is_sm_ips_ports: {row['is_sm_ips_ports']}, attack_cat: {row['attack_cat']}, label: {row['label']}", axis=1)
data['analysis'] = "Normal"  # Placeholder analysis for this script

# Convert to Dataset
dataset = LogDataset(data, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
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
    prompt = f"Input: {input_log}\nTarget:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = model.generate(**inputs, max_length=512, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

custom_input = "id: 99, dur: 0.300000, proto: udp, service: DNS, state: INT, spkts: 10, dpkts: 5, sbytes: 1200, dbytes: 300, rate: 10.000, ct_dst_sport_ltm: 2, ct_dst_src_ltm: 2, is_ftp_login: 0, ct_ftp_cmd: 0, ct_flw_http_mthd: 0, ct_src_ltm: 2, ct_srv_dst: 2, is_sm_ips_ports: 1, attack_cat: Attempted Admin, label: 1"

# Run the test
output = test_model(custom_input)
print("Generated Output:\n", output)


