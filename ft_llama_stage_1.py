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
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=False,              # Use 4bit to reduce memory usage
    use_gradient_checkpointing="unsloth",  # "True" or "unsloth" for long context
)

# Example: Enable LoRA adapters for efficient finetuning
def enable_peft(model):
    """
    Add LoRA adapters for parameter-efficient finetuning.
    Finetune only the language layers and optionally set vision finetuning flags.
    """
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=64,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model

def prepare_dataset(images_dir, csv_dir, key="True Class"):
    """
    根据给定的 images_dir 和 csv_dir 构造数据集。
    这里假设每个图像的 CSV 文件中包含 'predicted_label_name' 字段，
    标签包括 "healthy", "npdr", 和 "pdr"（不区分大小写）。
    """
    import pandas as pd
    from PIL import Image
    from pathlib import Path
    
    data_list = []
    image_files = list(Path(images_dir).glob("*.png"))  # 如有需要可调整文件扩展名

    for img_path in image_files:
        csv_path = Path(csv_dir) / f"{img_path.stem}.csv"
        if not csv_path.exists():
            continue
        
        df = pd.read_csv(csv_path, header=None, nrows=2)
        if len(df) < 2:
            continue
        
        

        # 提取第二行第二列的标签（行索引1，列索引1）
        true_label = df.iloc[1, 1]
        predicted_label_sentence = str(true_label).strip()
        
        # 简化后的 prompt：直接要求模型分类输出三个选项之一
        conversation = []
        img_obj = Image.open(img_path)
        user_content = [
            {
                "type": "text",
                "text": "Classify the OCTA DCP image and answer with one word only: Healthy, NPDR, or PDR."
            },
            {
                "type": "image",
                "image": img_obj
            }
        ]
        conversation.append({
            "role": "user",
            "content": user_content
        })
        conversation.append({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": predicted_label_sentence
            }]
        })
        
        data_list.append({"messages": conversation})
    
    print(f"Processed {len(data_list)} image-label pairs from {csv_dir}")
    if data_list:
        print("First item example:", data_list[0])
    return data_list

# Example: training using TRL's SFTTrainer
def train_model(model, tokenizer, train_data, eval_data=None):
    """
    Fine-tune the model using vision data via TRL SFTTrainer.
    如果提供 eval_data，则在训练过程中进行验证评估。
    """
    from trl import SFTTrainer, SFTConfig
    from unsloth.trainer import UnslothVisionDataCollator
    from unsloth import is_bf16_supported

    # Switch model to training mode.
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_data,
        eval_dataset=eval_data,  # 加载验证集数据
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=4, 
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
            evaluation_strategy="steps",  # 可设置为 "steps" 后每隔一定步数评估
            eval_steps=30,
        ),
    )
    trainer.train()
    model.save_pretrained("/home/cl2733/unsloth/checkpoints/llama_3.2_11b_class_only_split_1_64")
    tokenizer.save_pretrained("/home/cl2733/unsloth/checkpoints/llama_3.2_11b_class_only_split_1_64")

if __name__ == "__main__":
    # 根据实际情况修改以下路径：
    # 假设训练集图像在一个文件夹，CSV 标签在另一个文件
    train_images_dir = "/home/cl2733/LLaVA/playground/data/DCP_images"
    train_csv_dir = "/home/cl2733/unsloth/data/sub_dataset_1/train"
    # 验证集使用相同的图像读取方式，但 CSV 文件存放在不同的文件夹
    val_images_dir = "/home/cl2733/LLaVA/playground/data/DCP_images"
    val_csv_dir = "/home/cl2733/unsloth/data/sub_dataset_1/val"
    
    train_data = prepare_dataset(train_images_dir, train_csv_dir)
    val_data = prepare_dataset(val_images_dir, val_csv_dir)
    
    model = enable_peft(model)
    train_model(model, tokenizer, train_data, eval_data=val_data)
