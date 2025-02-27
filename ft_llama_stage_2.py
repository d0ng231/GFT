import torch
# FastVisionModel is the main class for handling vision+language tasks
from unsloth import FastVisionModel

# 4bit pre-quantized models supported to reduce OOM issues and download time.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",  # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit",  # Fits on 80GB GPU
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",
    "unsloth/Pixtral-12B-2409-bnb-4bit",               # Fits on 16GB
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",
    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",           # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",
    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",       # Llava variants
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
]

# Instantiate the model in 4-bit mode and enable gradient checkpointing.
model, tokenizer = FastVisionModel.from_pretrained(
    "/home/cl2733/unsloth/checkpoints/llama_3.2_11b_class_only_split_2_128",
    load_in_4bit=False,              # Use 4bit to reduce memory usage
    use_gradient_checkpointing="unsloth",  # "True" or "unsloth" for long context
)

# Example: Enable LoRA adapters for efficient finetuning
def enable_peft(model):
    """
    Example function to add LoRA adapters for parameter-efficient finetuning.
    Finetune only the language layers and optionally set vision finetuning flags.
    """
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=False,
        finetune_mlp_modules=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model

def prepare_dataset(jsonl_path="/home/cl2733/unsloth/data/Finetune_v18_aug.json",
                    images_dir="/home/cl2733/LLaVA/playground/data/DCP_images"):
    import json
    from PIL import Image
    
    data_list = []
    with open(jsonl_path, "r") as f:
        dataset = json.load(f)

    supplement_info = ( "Here's the explanation with specific visual characteristics observed on OCTA DCP images for healthy, NPDR, and PDR retinas:\n"

        "Healthy Retina \n"
        "* Vascular Structure: The DCP displays a dense and continuous network of small capillaries forming an intricate lattice-like pattern. The vessels are well-distributed and show no interruptions.\n"
        "* Foveal Avascular Zone (FAZ): This central region is a clear, well-defined circular or oval-shaped area devoid of blood vessels. The edges of the FAZ are smooth and regular.\n"
        "* Flow Signals: Blood flow in vessels is continuous and homogeneous, with no gaps or flow voids. The choriocapillaris beneath the retina also shows a dense and uniform flow pattern.\n"

        "Non-Proliferative Diabetic Retinopathy (NPDR)\n"
        "* Bright, rounded dots scattered across the DCP network. These microaneurysms are localized areas of vessel wall weakening or dilation and can be one of the earliest detectable abnormalities in NPDR. \n"
        "* Capillary Dropout: Seen as dark, irregular patches where the capillary network is disrupted and blood flow is absent.\n"
        "* FAZ Enlargement: The FAZ in the DCP becomes noticeably larger and less circular, with jagged and uneven borders. This enlargement is an indicator of early ischemic damage.\n"
        "* Vessel Tortuosity and Dilation: Blood vessels become more tortuous and uneven in thickness. Some vessels may appear dilated and malformed.\n"
        "* Reduced Vascular Density (VD): The DCP shows thinning and reduced capillary density, particularly in the parafoveal region, resulting in a patchy appearance\n"

        "Proliferative Diabetic Retinopathy (PDR)\n"
        "* Neovascularization: Though less common in the DCP, abnormal clusters of fine, disorganized vessels can sometimes be detected extending deeper into this layer. These vessels appear chaotic and lack the uniformity of healthy capillary network\n"
        "* Capillary Non-Perfusion: Large, well-defined regions of absent flow dominate the DCP in advanced PDR. These ischemic zones appear as extensive dark patches, often surrounding the enlarged FAZ.\n"
        "* FAZ Expansion and Irregularity: The FAZ becomes significantly larger and more distorted, often losing its defined circular or oval shape entirely.\n"
        "* Flow Voids: Beyond the ischemic zones, smaller scattered flow voids may appear throughout the DCP, disrupting the capillary continuity and contributing to the overall reduction in VD.\n"

        "Distinguishing Features Between NPDR and PDR in DCP \n"
        "Neovascularization:\n"
        "NPDR: Lacks neovascularization in the DCP.\n"
        "PDR: Displays disorganized, abnormal vessel clusters extending beyond the usual vascular layers.\n"
        "Capillary Non-Perfusion:\n"
        "NPDR: Smaller and patchy non-perfused regions in the DCP.\n"
        "PDR: Larger, confluent areas of capillary dropout, especially in the peripheral retina.\n"
        "FAZ Changes:\n"
        "NPDR: Mild to moderate enlargement with jagged borders.\n"
        "PDR: Severe FAZ enlargement with fragmentation and irregularities in shape.\n"
        "Vascular Density (VD):\n"
        "NPDR: Generalized reduction in VD, primarily in the parafoveal area.\n"
        "PDR: Severe VD reduction across wider areas, often extending beyond the central retina\n")
    
    for item in dataset:
        conversation = []
        user_content = []
        img_path = f"{images_dir}/{item['id']+'.png'}"
        img_obj = Image.open(img_path)
        user_content.append({
            "type": "image",
            "image": img_obj   
        })
        # user_content.append({
        #     "type": "text",
        #     "text": supplement_info
        # })
        conversation.append({
            "role": "user",
            "content": user_content
        })
        for c in item["conversations"]:
            if c["from"] == "human":
                user_content = []
                text_value = c["value"]
                user_content.append({
                    "type": "text",
                    "text": text_value
                })
                conversation.append({
                    "role": "user",
                    "content": user_content
                })
            elif c["from"] == "gpt":
                conversation.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": c["value"]
                    }]
                })
        data_list.append({"messages": conversation})
    print(data_list[0])
    return data_list

# Example: training using TRL's SFTTrainer
def train_model(model, tokenizer, converted_dataset):
    """
    Demonstrates how to fine-tune the model using vision data via TRL SFTTrainer.
    """
    from trl import SFTTrainer, SFTConfig
    from unsloth.trainer import UnslothVisionDataCollator
    from unsloth import is_bf16_supported

    # Switch model to training mode
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),  # Required data collator
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs = 1,
            learning_rate=2e-4,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=2048,
        ),
    )
    trainer.train()

    model.save_pretrained("/home/cl2733/unsloth/checkpoints/llama_3.2_11b_split_2_128_stage2")
    tokenizer.save_pretrained("/home/cl2733/unsloth/checkpoints/llama_3.2_11b_split_2_128_stage2")


# Example usage in this script
if __name__ == "__main__":
    # Prepare data
    data = prepare_dataset()

    # model = enable_peft(model)

    # Train for a few steps
    train_model(model, tokenizer, data)

