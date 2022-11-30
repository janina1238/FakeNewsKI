from transformers import pipeline, GPTJForCausalLM, AutoTokenizer
import torch

# analyses if a text has a good or bad mood
classifier = pipeline("sentiment-analysis")
print(classifier("It is a bad and good day!"))

# exception: No module named 'keras.saving.hdf5_format'
# because of an older keras version
# /home/mirevi/janina/gpt_j_test_app/venv/lib/python3.10/site-packages/transformers/modeling_tf_utils.py 
# comment out: #from keras.saving.hdf5_format import save_attributes_to_hdf5_group

# classification in a text with selected labels
classifier = pipeline("zero-shot-classification")
print(classifier("We are going to the zoo today.",
        candidate_labels=["animals", "politics", "kids"]))

model = GPTJForCausalLM.from_pretrained(

    "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# generates a text to a given text
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(generator("It was a beautiful day and"))