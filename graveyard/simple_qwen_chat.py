import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

# Load model and tokenizer
model_id = "Qwen/Qwen1.5-14B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")


def chat():
    print("Simple Qwen Chat (type 'q' to exit)")

    while True:
        # Get user input
        user_input = input("You: ")

        # Exit condition
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        # Prepare the input
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ]

        # Tokenize the input
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Generate response
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Generate
        generate_ids = model.generate(inputs.input_ids, max_new_tokens=512, do_sample=True, temperature=0.7)

        # Decode the response
        response = tokenizer.decode(generate_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)

        # Print response
        print("AI:", response)


if __name__ == "__main__":
    chat()
