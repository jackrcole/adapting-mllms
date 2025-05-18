import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from unsloth import FastVisionModel
from trl import SFTTrainer
import wandb
from PIL import Image
import requests

# --- Configuration ---
WANDB_PROJECT_NAME = "adapting-mllms"
MODEL_NAME = "unsloth/gemma-3-4b-it" 

OUTPUT_DIR = "../results/model_output"
LOGGING_DIR = "../results/logs"

# --- Helper Functions ---

def initialize_wandb(model_name, dataset_name="custom_dataset"):
    """Initializes Weights & Biases for experiment tracking."""
    try:
        wandb.login() # Attempts to login, will use existing login if available
    except Exception as e:
        print(f"Wandb login failed. Please ensure you are logged in. You can run 'wandb login' in your terminal. Error: {e}")
        print("Proceeding without wandb logging for now, but monitoring will be unavailable.")
        return None
    
    run = wandb.init(
        project=WANDB_PROJECT_NAME,
        config={
            "model_name": model_name,
            "dataset_name": dataset_name,
            "output_dir": OUTPUT_DIR,
            "logging_dir": LOGGING_DIR,
        }
    )
    print("Wandb initialized successfully.")
    return run

def load_model_and_tokenizer_hf(model_name, quantization_config=None):
    """Loads a model and tokenizer from Hugging Face, optionally with quantization."""
    print(f"Loading Hugging Face model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        # device_map="auto", # Usually good for multi-GPU, but manage explicitly for Unsloth
        torch_dtype=torch.float16, # Or bfloat16 if available and preferred
    )
    print("Hugging Face model and tokenizer loaded.")
    return model, tokenizer

def load_model_and_tokenizer_unsloth(model_name, max_seq_length=2048, use_gradient_checkpointing=True):
    """
    Loads a model and tokenizer using Unsloth for potentially faster training.
    Note: Unsloth is primarily optimized for LLMs. VLM support might be indirect
    (e.g., fine-tuning the language component of a VLM).
    This function assumes we are loading a base LLM that Unsloth supports.
    """
    print(f"Loading model with Unsloth: {model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Unsloth handles dtype
        load_in_4bit=True, # Example: Use 4-bit quantization
    )
    
    # Optional: Apply LoRA configuration if planning to fine-tune with LoRA
    model = FastVisionModel.get_peft_model(
        model,
        r=16, # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"], # Adjust for your model
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=42,
        # use_rslora=False,  # RoPE Scaling LoRA
        # loftq_config=None, # LoftQ
    )
    print("Unsloth model and tokenizer loaded and prepared for PEFT.")
    return model, tokenizer

def prepare_dataset_placeholder(tokenizer, dataset_name="dummy_vlm_dataset", split="train"):
    """
    Placeholder for dataset loading and preprocessing for a VLM.
    Actual implementation will depend heavily on the dataset format and task.
    For VLMs, this involves handling both images and text.
    Example: Image-captioning dataset might have image paths/URLs and corresponding captions.
    """
    print(f"Preparing placeholder dataset: {dataset_name}")
    # This is a very basic example. Real VLM datasets are more complex.
    # E.g., {'image': [PIL.Image.open(...), ...], 'text': ["caption1", "caption2", ...]}
    # Or {'image_url': ["url1", "url2"], 'text': ["caption1", "caption2"]}

    # For LLM fine-tuning (if using Unsloth on the language part):
    # data = load_dataset("Abirate/english_quotes", split="train")
    # def formatting_prompts_func(examples):
    #     texts = []
    #     for quote in examples["quote"]:
    #         texts.append(f"Quote: {quote} Who said this?") # Example prompt
    #     return { "text" : texts, }
    # dataset = data.map(formatting_prompts_func, batched = True,)
    # return dataset
    
    # Placeholder for VLM data
    # For a real scenario, you would load images and pair them with text
    # For example, load a dataset like COCO, Flickr30k, or a custom one.
    # The dataset should return image tensors and tokenized text.
    print("Using a dummy dataset for demonstration.")
    # Create a dummy dataset that SFTTrainer can consume
    # This is more like an LLM dataset, adapt for actual VLM data
    raw_dataset = [
        {"image_path": "dummy_image1.jpg", "text": "This is a <image> of a cat."},
        {"image_path": "dummy_image2.png", "text": "Describe this <image>: a dog playing fetch."}
    ]
    # In a real VLM, you'd process images into tensors and text into input_ids
    # SFTTrainer expects a 'text' column or a dataset formatted by its data collator.
    # For VLMs, the 'text' might include special tokens indicating image presence.
    
    # For now, let's simulate a text-based dataset for SFTTrainer compatibility as a placeholder
    # if Unsloth is to be used for LLM part.
    # If using a full VLM fine-tuning approach, this part needs significant changes.
    dummy_text_dataset = load_dataset("imdb", split="train[:1%]") # Load a small text dataset
    
    def preprocess_function(examples):
        # This is a generic text preprocessing function.
        # For VLMs, you would handle image inputs and combine them with text.
        # For example, convert images to pixel values and tokenize text.
        # The tokenizer for VLMs (like BLIP) handles multimodal inputs.
        # inputs = tokenizer(images=examples['image'], text=examples['text'], padding="max_length", truncation=True)
        # For this placeholder, we'll just use the text part.
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    # Processed_dataset = dummy_text_dataset.map(preprocess_function, batched=True)
    # SFTTrainer usually works with a 'text' field directly or a pre-tokenized dataset.
    # For simplicity with SFTTrainer in this placeholder:
    def format_example(example): # SFTTrainer expects a text column
        # For a VLM, this would involve formatting the text prompt around an image
        # e.g., "USER: <image> Describe this image. ASSISTANT: This is an image of..."
        return {"text": example["text"]}

    formatted_dataset = dummy_text_dataset.map(format_example)
    print("Placeholder dataset prepared.")
    return formatted_dataset


def fine_tune_model(model, tokenizer, train_dataset, eval_dataset=None):
    """Fine-tunes the model using SFTTrainer (suitable for LLMs/language part of VLMs)."""
    print("Starting fine-tuning process...")
    
    # For true VLM fine-tuning with Hugging Face, you'd use the Trainer class
    # and a custom data collator for images and text.
    # Unsloth's SFTTrainer is primarily for instruction fine-tuning of LLMs.
    # If 'model' is an Unsloth-prepared LLM:
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",  # Or the field containing your formatted text
        max_seq_length=getattr(tokenizer, 'model_max_length', 2048), # Get from tokenizer or set manually
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            # max_steps=60, # Enable for quick test, comment out for full training
            num_train_epochs=1, # Adjust as needed
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(), # Use bf16 if available, else fp16
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit", # Unsloth recommends adamw_8bit
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=OUTPUT_DIR,
            report_to="wandb" if wandb.run else "none", # Report to wandb if initialized
            logging_dir=LOGGING_DIR,
        ),
    )
    
    print("Trainer initialized. Starting training...")
    trainer_stats = trainer.train()
    print("Training complete.")
    print(f"Trainer stats: {trainer_stats}")

    # Save the fine-tuned model (adapter if LoRA was used)
    # For Unsloth LoRA models:
    # For Unsloth LoRA models:
    if hasattr(model, 'save_pretrained') and "unsloth" in str(type(model)).lower(): # Check for Unsloth model attribute and type string
         model.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora_adapter")) # Saves LoRA adapter
         # tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora_adapter")) # Tokenizer might not change with LoRA
         print(f"LoRA adapter saved to {os.path.join(OUTPUT_DIR, 'final_lora_adapter')}")
    elif hasattr(trainer, 'save_model'): # Standard Hugging Face Trainer
        trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
        # tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model")) # Save tokenizer if needed
        print(f"Full model saved to {os.path.join(OUTPUT_DIR, 'final_model')}")
    else:
        print("Model saving skipped as it's not a recognized Unsloth LoRA model or HF Trainer model.")

    return model # Return the trained model (or adapter-merged model)

def run_vlm_inference(model, tokenizer, image_path_or_url, prompt_text="Describe the image."):
    """
    Runs inference with a Vision-Language Model.
    This function needs to be adapted based on the specific VLM's architecture.
    For example, BLIP models have `generate` method that takes `pixel_values` and `input_ids`.
    """
    print(f"Running VLM inference for image: {image_path_or_url}")
    
    # Load the image
    try:
        if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
            image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_path_or_url).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # This is a generic placeholder. Specific VLMs (e.g., BLIP, LLaVA) have their own processors/tokenizers.
    # For Salesforce/blip-image-captioning-base:
    # It uses BlipProcessor which combines BlipImageProcessor and AutoTokenizer
    # from transformers import BlipProcessor
    # processor = BlipProcessor.from_pretrained(MODEL_NAME) # Or from where model was loaded
    # inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(model.device)
    # generated_ids = model.generate(**inputs)
    # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # Simplified inference for a model where text prompt guides generation based on image context
    # This part is HIGHLY model-dependent.
    # If using Unsloth for an LLM part, this might just be text generation.
    # For now, this is a conceptual placeholder.
    
    # If the model is an Unsloth fine-tuned LLM (not a true VLM), inference would be text-based:
    if "unsloth" in str(type(model)).lower(): # Check if it's an Unsloth model
        print("Running inference with Unsloth LLM (text-based).")
        inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=50)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    elif hasattr(model, 'generate') and hasattr(tokenizer, 'decode'): # General HuggingFace model
        print(f"Running inference with Hugging Face model ({model.config.model_type}).")
        # This is a generic text-based generation.
        # For actual VLMs, image processing is needed here.
        # For models like BLIP, you'd use a processor that handles images.
        # E.g. from transformers import BlipProcessor; processor = BlipProcessor.from_pretrained(hf_model_name)
        # inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(model.device)

        # Since we loaded a generic AutoModelForCausalLM initially, we'll simulate text generation
        # This assumes the VLM can take text prompts for generation.
        # If `MODEL_NAME` was a BLIP model, `AutoModelForCausalLM` might not be the right class.
        # `AutoModelForVision2Seq` or specific classes like `BlipForConditionalGeneration` are common.
        
        # Let's assume a multimodal processor and model for this example (conceptual)
        # This part would typically use a specific processor for the VLM
        try:
            from transformers import BlipProcessor # Attempt to use a BLIP processor as an example
            processor = BlipProcessor.from_pretrained(MODEL_NAME)
            inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(model.device)
            pixel_values = inputs.get('pixel_values')
            input_ids = inputs.get('input_ids')

            if pixel_values is None or input_ids is None:
                print("Warning: Could not get pixel_values or input_ids from processor. Falling back to text-only.")
                # Fallback to text-only if image processing fails or isn't set up for the current model
                text_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
                generated_ids = model.generate(**text_inputs, max_new_tokens=100)
            else:
                 # Ensure all inputs are on the same device as the model
                pixel_values = pixel_values.to(model.device)
                input_ids = input_ids.to(model.device)
                # Note: Some models might not expect input_ids if an image is provided, or vice-versa
                # Adjust based on the specific VLM
                if model.config.model_type == "blip": # Specific handling for BLIP
                     generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_new_tokens=100)
                else: # Generic attempt
                    generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=100, num_beams=4)


            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            print(f"Generated text: {generated_text}")
        except ImportError:
            print("BlipProcessor not found. VLM-specific inference might not work correctly.")
            print("Falling back to simple text generation if model supports it.")
            text_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            outputs = model.generate(**text_inputs, max_new_tokens=50)
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    else:
        generated_text = "Inference for this model type is not fully implemented in this script."

    print(f"Inference result: {generated_text}")

    if wandb.run:
        try:
            wandb_image = wandb.Image(image, caption=f"Input: {prompt_text}")
            wandb.log({"inference_image": wandb_image, "inference_prompt": prompt_text, "inference_output": generated_text})
        except Exception as e:
            print(f"Failed to log inference to wandb: {e}")
            # Fallback: log image path if PIL object fails
            wandb.log({"inference_image_path": image_path_or_url, "inference_prompt": prompt_text, "inference_output": generated_text})


    return generated_text

def main():
    """Main function to run the VLM pipeline."""
    
    # --- 0. Initialize WandB ---
    # Set WANDB_API_KEY in your environment variables, or run `wandb login`
    # os.environ["WANDB_DISABLED"] = "true" # Uncomment to disable wandb
    wandb_run = initialize_wandb(model_name=MODEL_NAME)

    # --- 1. Load Model and Tokenizer ---
    # Choose one: Hugging Face generic or Unsloth optimized
    
    # Option A: Hugging Face (e.g., for a standard VLM like BLIP)
    # For many VLMs, you'd use specific classes like BlipForConditionalGeneration, BlipProcessor
    # from transformers import BlipForConditionalGeneration, BlipProcessor
    # model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    # tokenizer = BlipProcessor.from_pretrained(MODEL_NAME) # Processor often includes tokenizer & image_processor
    
    # Using AutoModel for flexibility, but might need specific model class for full VLM features.
    # The MODEL_NAME Salesforce/blip-image-captioning-base is actually a BlipForConditionalGeneration model.
    from transformers import BlipForConditionalGeneration, BlipProcessor
    try:
        print(f"Attempting to load VLM: {MODEL_NAME} with specific classes.")
        model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
        # Tokenizer for BLIP is usually part of the BlipProcessor
        # For inference/finetuning, the processor handles image and text preparation.
        # For SFTTrainer with text only, we might just need the text tokenizer part.
        # For now, we'll load a general tokenizer, and the inference fn will try to load BlipProcessor
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) # For text part
        print(f"Successfully loaded {MODEL_NAME} and its tokenizer.")
    except Exception as e:
        print(f"Could not load {MODEL_NAME} with BlipForConditionalGeneration. Error: {e}")
        print("Falling back to AutoModelForCausalLM. Full VLM capabilities might be limited.")
        # Fallback or for LLM-focused approach
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if torch.cuda.is_available() else None
        model, tokenizer = load_model_and_tokenizer_hf(MODEL_NAME, quantization_config=quant_config)
        if torch.cuda.is_available():
            model = model.to("cuda")


    # Option B: Unsloth (primarily for LLMs or language component of VLMs)
    # unsloth_model_name = "unsloth/llama-3-8b-bnb-4bit" # Example Unsloth model
    # model, tokenizer = load_model_and_tokenizer_unsloth(unsloth_model_name)
    # Note: If using Unsloth, ensure the dataset and fine-tuning are compatible.
    # The current `prepare_dataset_placeholder` and `fine_tune_model` use SFTTrainer,
    # which is text-based. For full VLM fine-tuning, this would differ.

    # --- 2. Prepare Dataset ---
    # This is a placeholder. For actual VLM fine-tuning, you need a dataset
    # with images and corresponding texts, and a processor to convert them into model inputs.
    # The SFTTrainer used in `fine_tune_model` expects a text-based dataset.
    # If fine-tuning a full VLM, you'd likely use Hugging Face's `Trainer` with a custom DataCollator.
    print("Preparing dataset (using placeholder)...")
    # For SFTTrainer, we need a text dataset. The tokenizer should match the model.
    train_dataset = prepare_dataset_placeholder(tokenizer, dataset_name="placeholder_text_for_sfttrainer")
    # eval_dataset = prepare_dataset_placeholder(tokenizer, dataset_name="dummy_vlm_dataset", split="validation")


    # --- 3. Fine-tune Model (Optional) ---
    # Ensure your model and dataset are correctly prepared for the chosen fine-tuning method.
    # If using Unsloth for LLM part, model should be from load_model_and_tokenizer_unsloth.
    # If using HF Trainer for VLM, setup is different.
    # The current setup leans towards SFTTrainer for text-based fine-tuning.
    
    # To demonstrate fine-tuning with SFTTrainer (assuming a text-based task or LLM component):
    # This will only work if the model is an LLM or can be fine-tuned via its language modeling head
    # and the dataset is text-based as prepared by prepare_dataset_placeholder.
    # For a true VLM like BLIP, SFTTrainer is not the direct tool for end-to-end VLM finetuning.
    # You would use `Trainer` with custom data collators.
    # For now, we will skip fine-tuning if the loaded model is not easily adaptable by SFTTrainer
    # or if it's primarily a VLM better suited for other training loops.
    
    # Let's simulate a fine-tuning step if we're using an Unsloth-like setup for an LLM
    # Or if we want to fine-tune the language part of the VLM using text data.
    # We need to be careful here, as fine-tuning Salesforce/blip-image-captioning-base
    # with SFTTrainer directly on text data might not be standard.

    # For demonstration, let's assume `model` is compatible with SFTTrainer (e.g., if it were an Unsloth LLM)
    # Or if we are fine-tuning only the text decoder of a VLM.
    # Given MODEL_NAME is BLIP, SFTTrainer is not ideal for its standard VLM fine-tuning.
    # We will comment this out to avoid errors with the BLIP model unless explicitly adapted.

    # print("Skipping fine-tuning for BLIP model with SFTTrainer in this generic setup.")
    # print("To fine-tune a VLM like BLIP, use HuggingFace Trainer with a VLM-specific data collator.")
    
    # If you were to use Unsloth for an LLM:
    # model_for_finetuning_unsloth, tokenizer_for_finetuning_unsloth = load_model_and_tokenizer_unsloth("unsloth/llama-2-7b-bnb-4bit")
    # text_dataset_for_unsloth = prepare_dataset_placeholder(tokenizer_for_finetuning_unsloth)
    # fine_tune_model(model_for_finetuning_unsloth, tokenizer_for_finetuning_unsloth, text_dataset_for_unsloth)
    # model = model_for_finetuning_unsloth # Update model to the fine-tuned one
    # tokenizer = tokenizer_for_finetuning_unsloth


    # --- 4. Run Inference ---
    # Example image URL for testing
    # Make sure to create dummy_image.jpg or use a valid path/URL
    # Example: Create a dummy image file for testing if not using URLs
    try:
        from PIL import Image
        dummy_img = Image.new('RGB', (60, 30), color = 'red')
        if not os.path.exists("dummy_image.jpg"):
            dummy_img.save("dummy_image.jpg")
        print("Created/verified dummy_image.jpg for inference testing.")
    except ImportError:
        print("Pillow not installed, cannot create dummy image. Please install Pillow or provide image paths.")
    except Exception as e:
        print(f"Could not create dummy image: {e}")


    image_url = "http://images.cocodataset.org/val2017/00000003gv  9769.jpg" # A cat image
    local_image_path = "dummy_image.jpg" # Use the dummy image created

    # Use the loaded model (either original VLM or potentially fine-tuned LLM part)
    print(f"\n--- Running Inference on VLM ({MODEL_NAME}) ---")
    run_vlm_inference(model, tokenizer, image_path_or_url=image_url, prompt_text="A picture of a cat on a couch.")
    run_vlm_inference(model, tokenizer, image_path_or_url=local_image_path, prompt_text="Describe this image")

    # --- 5. Clean up ---
    if wandb.run:
        wandb.finish()
    print("\nPipeline finished.")

if __name__ == "__main__":
    # Set environment variables for CUDA if necessary, Unsloth might also manage this.
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    main()
