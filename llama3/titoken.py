from transformers import AutoTokenizer

# Initialize the tokenizer
model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Basic Tokenization
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# Encoding Text
encoded = tokenizer.encode(text)
print("Encoded:", encoded)

# Decoding Tokens
decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)

# Encoding with Attention Masks and Padding
texts = ["Hello, how are you?", "I am fine, thank you!"]
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print("Encoded inputs:", encoded_inputs)

# Special Tokens
print("Special Tokens:", tokenizer.special_tokens_map)
print("All Special Tokens:", tokenizer.all_special_tokens)
print("All Special Token IDs:", tokenizer.all_special_ids)

# Adding New Tokens
new_tokens = ["[NEW_TOKEN]"]
tokenizer.add_tokens(new_tokens)
print("Added new token IDs:", tokenizer.encode("[NEW_TOKEN]"))

# Using New Tokens in Text
text_with_new_token = "Hello [NEW_TOKEN]!"
encoded_with_new_token = tokenizer.encode(text_with_new_token)
print("Encoded with new token:", encoded_with_new_token)
decoded_with_new_token = tokenizer.decode(encoded_with_new_token)
print("Decoded with new token:", decoded_with_new_token)

# Batch Encoding
texts = ["Hello, how are you?", "I am fine, thank you!"]
batch_encoded = tokenizer.batch_encode_plus(
    texts, padding=True, truncation=True, return_tensors="pt"
)
print("Batch Encoded:", batch_encoded)

# Handling Long Texts
long_text = "This is a very long text." * 100
encoded_long_text = tokenizer.encode_plus(
    long_text, max_length=128, truncation=True, return_tensors="pt"
)
print("Encoded long text:", encoded_long_text)
