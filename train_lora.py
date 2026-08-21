from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template
import os

def main():
    # 1. Configuration
    max_seq_length = 2048 # Generous window for JSON payloads
    model_name = "unsloth/gemma-2b-bnb-4bit" # Replace with exact Gemma-4-2B string if needed
    
    print("Loading 4-bit Base Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    print("Configuring LoRA Adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 2. Dataset Preparation
    print("Loading dataset.jsonl...")
    dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

    # Standardize ChatML format to match ShareGPT format in dataset.jsonl
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "chatml",
        mapping = {"role": "from", "content": "value", "user": "user", "assistant": "model"}
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts }

    dataset = dataset.map(formatting_prompts_func, batched = True)

    # 3. Training Loop
    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # Effective batch size 8
            warmup_steps = 5,
            num_train_epochs = 3, # Standard for JSON enforcement
            learning_rate = 2e-4,
            fp16 = not FastLanguageModel.is_bfloat16_supported(),
            bf16 = FastLanguageModel.is_bfloat16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "lora_outputs",
        ),
    )

    print("Beginning Training! Watch VRAM usage closely.")
    trainer.train()

    # 4. Save the finalized LoRA adapter
    print("Training Complete. Saving adapters to /skyrim_json_lora...")
    model.save_pretrained("skyrim_json_lora")
    tokenizer.save_pretrained("skyrim_json_lora")
    
if __name__ == "__main__":
    main()
