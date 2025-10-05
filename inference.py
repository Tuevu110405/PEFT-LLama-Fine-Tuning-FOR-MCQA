from datasets import load_dataset
from transformers import pipeline
from unsloth import FastLanguageModel
PEFT_MODEL = "tientuevu/Llama-3.2-1B-bnb-4bit-MedMCQA"

ds = load_dataset("openlifescienceai/medmcqa")
del ds['test']

# Preprocessing
SYSTEM_PROMPT = "You are a helpful medical assistant specialized in answering multiple-choice questions. Provide the correct answer by only stating the option letter ."

id2label = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}

def formatting_prompt(examples):
    questions = examples["question"]
    opas = examples["opa"]
    opbs = examples["opb"]
    opcs = examples["opc"]
    opds = examples["opd"]
    cops = examples["cop"]

    texts = []
    for idx in range(len(questions)):
        question = questions[idx]
        opa = opas[idx]
        opb = opbs[idx]
        opc = opcs[idx]
        opd = opds[idx]
        answer = id2label[cops[idx]]

        if answer == "A":
            answer = answer
        elif answer == "B":
            answer = answer
        elif answer == "C":
            answer = answer
        elif answer == "D":
            answer = answer


        choices = f"A. {opa}. B. {opb}. C.{opc}. D. {opd}."
        #Build user message content
        user_message_content = f"""

### Question:
{question}

### Choice:
{choices}

"""
        full_conversation = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n{user_message_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n{answer}<|eot_id|>"

        )


        texts.append(full_conversation)
    return {"text": texts,}

process_ds = ds.map(formatting_prompt, batched=True)

#option 1
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = PEFT_MODEL,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    dtype= None,
)

model = FastLanguageModel.for_inference(model)
model.to("cuda")

input_text = process_ds["validation"][0]["text"]

inputs = tokenizer(input_text, return_tensors = "pt").to("cuda")

output = model.generate(input["input_ids"], max_length = 128)

generated_text = tokenizer.decode(output[0], skip_special_tokens = True)
print(f"Answer: {generated_text}")