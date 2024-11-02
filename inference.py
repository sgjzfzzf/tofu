from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to the directory where model files are saved
model_path = "real/ft_epoch5_lr2e-05_qwen_full_wd0.01"  # replace with the actual path

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_path, ignore_mismatched_sizes=True)

# Make sure the model is in evaluation mode
model.eval()

# Input text
input_text = "What is a common theme in Anara Yusifova's work?"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate response
outputs = model.generate(**inputs)

# Decode and print the result
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
