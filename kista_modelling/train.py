# dataset_setup.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_dataset():
    # Load dataset
    dataset = load_dataset("aaditya/alpaca_subset_1")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    tokenizer.pad_token = tokenizer.eos_token
    
    def format_instruction(example):
        template = f"""### Instruction: {example['instruction']}
### Input: {example['input']}
### Response: {example['output']}"""
        return {"text": template}
    
    # Format dataset
    formatted_dataset = dataset.map(format_instruction)
    
    # Tokenize function
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    # Process dataset
    tokenized_dataset = formatted_dataset.map(
        tokenize,
        remove_columns=formatted_dataset["train"].column_names
    )
    
    return tokenized_dataset, tokenizer

# training_setup.py
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def setup_training(tokenized_dataset, tokenizer):
    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="llama3-alpaca-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        fp16=True
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        args=training_args,
        tokenizer=tokenizer,
        peft_config=peft_config
    )
    
    return trainer

# main.py
def main():
    # Prepare dataset
    tokenized_dataset, tokenizer = prepare_dataset()
    
    # Setup and start training
    trainer = setup_training(tokenized_dataset, tokenizer)
    trainer.train()
    
    # Save model
    trainer.save_model("llama3-alpaca-finetuned-final")

if __name__ == "__main__":
    main()
