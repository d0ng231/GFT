import argparse, json
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from unsloth import FastVisionModel as FVM

def run(args):
    model, tok = FastVisionModel.from_pretrained(args.model, load_in_4bit=False, use_gradient_checkpointing="unsloth")
    FVM.for_training(model)
    if args.data.endswith(".json"):
        with open(args.data) as f: data = json.load(f)
    else:
        with open(args.data) as f: data = [json.loads(l) for l in f]
    conf = SFTConfig(output_dir=args.out, per_device_train_batch_size=1, num_train_epochs=args.epochs, save_steps=200)
    tr = SFTTrainer(model=model, tokenizer=tok, train_dataset=data, data_collator=UnslothVisionDataCollator(tok), args=conf)
    tr.train(); tr.save_model(args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="checkpoints/stage2")
    ap.add_argument("--epochs", type=int, default=1)
    run(ap.parse_args())
