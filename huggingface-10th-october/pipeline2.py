from transformers import pipeline

# Text generation
generator = pipeline("text-generation", model="gpt2")
print(generator("Once upon a time", max_new_tokens=30)[0]["generated_text"])

# Summarization (transformers v5+): use a chat model with text-generation.
# The old "summarization" pipeline was removed in v5.
summarizer = pipeline("text-generation", model="Qwen/Qwen3-4B-Instruct-2507")
text = ("Hugging Face is a platform where the machine learning community "
        "shares models, datasets, and applications. It provides open-source "
        "libraries that make it easy to use state-of-the-art models.")
messages = [{"role": "user", "content": f"Summarize the following text in one sentence:\n\n{text}"}]
print(summarizer(messages, max_new_tokens=60)[0]["generated_text"][-1]["content"])

# Zero-shot classification
zs = pipeline("zero-shot-classification")
print(zs("I need to reset my password",
         candidate_labels=["billing", "technical support", "sales"]))
