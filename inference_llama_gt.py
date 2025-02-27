import os
import csv
import torch
from PIL import Image
from unsloth import FastVisionModel
from datetime import datetime
import re

def load_local_model(model_path):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=False
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer

def clean_model_output(output):

    output_lower = output.lower()
    pattern = r'\b(healthy|npdr|pdr|non[-\s]?proliferative diabetic retinopathy|proliferative diabetic retinopathy)\b'
    match = re.search(pattern, output_lower, re.IGNORECASE)
    if match:
        found = match.group(1)
        if found.startswith("healthy"):
            return "Healthy"
        elif found.startswith("npdr") or found.startswith("non"):
            return "NPDR"
        elif found.startswith("pdr") or found.startswith("proliferative"):
            return "PDR"
    return None

def run_inference(model, tokenizer, image_path, max_retries=5):
    structured_prompt = (
        "Analyze this deep capillary plexus (DCP) image imaged by optical coherence tomography angiography (OCTA). First state your prediction exactly as one of the three classes: "
        "'Healthy', 'NPDR', or 'PDR'. "
        "Then provide your reasoning."

        "Here's the explanation with specific visual characteristics observed on OCTA DCP images for healthy, NPDR, and PDR retinas:\n"

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
        "PDR: Severe VD reduction across wider areas, often extending beyond the central retina\n"
    )

    simple_prompt = (
        "Classify the OCTA DCP image and answer with one word only: Healthy, NPDR, or PDR"
    )
    
    simple_prompt_stage2 = (
        "Classify the OCTA DCP image as: Healthy, NPDR, or PDR and explain why."
    )

    Exp_q1 = (
        "Answer in a paragraph. Is this OCTA DCP image suggesting a healthy, PDR or NPDR condition? Explain your prediction."
    )

    Exp_q2 = (
        "Answer in a three-sentence paragraph. Try to locate any regions indicating capillary dropout and neovascularization, respectively."
    )
    
    Exp_q3=(
        "Answer in a three-sentence paragraph.  Why does this image suggest PDR, not NPDR? "
    )

    Exp_q4=(
        "Answer in a three-sentence paragraph. Why does this image suggest NPDR, not PDR? "
    )

    Exp_q5=(
        "Answer in a three-sentence paragraph. Describe what can you see in the 1 o’clock direction of the FAZ."
    )


    for attempt in range(max_retries):
        with Image.open(image_path) as img:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": Exp_q5},
                ],
            }]
            
            inputs = tokenizer(
                img,
                tokenizer.apply_chat_template(messages, add_generation_prompt=True),
                add_special_tokens = False,
                return_tensors="pt"
            ).to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.5,
            do_sample=True
        )

        raw_output = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()

        prediction = clean_model_output(raw_output)
        if prediction:
            return raw_output, prediction
    return raw_output, "Unknown"

def parse_ground_truth_label(csv_path):
    """Parse true label from the second row of CSV file"""
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return "Unknown"
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            data_row = next(reader)  # Read second row
            
        if len(data_row) < 2:
            print(f"Invalid CSV format, insufficient columns: {csv_path}")
            return "Unknown"
            
        true_label = data_row[1].strip().upper()
        label_map = {
            "HEALTHY": "Healthy",
            "NPDR": "NPDR",
            "PDR": "PDR"
        }
        return label_map.get(true_label, "Unknown")
    except Exception as e:
        print(f"Error parsing CSV file {csv_path}: {str(e)}")
        return "Unknown"

def calculate_metrics(stats):
    metrics = {}
    for cls in stats:
        tp = stats[cls]['TP']
        fp = stats[cls]['FP']
        fn = stats[cls]['FN']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics[cls] = {'precision': precision, 'recall': recall}
    return metrics

def print_metrics(title, metrics, class_names, correct, total):
    acc = correct / total if total > 0 else 0
    balanced_acc = sum(metrics[cls]['recall'] for cls in class_names) / len(class_names)
    
    print(f"\n{title}")
    print(f"Accuracy: {acc:.4f} | Balanced Accuracy: {balanced_acc:.4f}")
    for cls in class_names:
        prec = metrics[cls]['precision']
        rec = metrics[cls]['recall']
        print(f"{cls}: Precision={prec:.4f}, Recall={rec:.4f}")

def evaluate_model(model_path, images_dir, label_dir):
    model, tokenizer = load_local_model(model_path)
    
    # Initialize metrics
    ternary_stats = {
        'Healthy': {'TP': 0, 'FP': 0, 'FN': 0},
        'NPDR': {'TP': 0, 'FP': 0, 'FN': 0},
        'PDR': {'TP': 0, 'FP': 0, 'FN': 0},
    }
    
    binary_stats = {
        'Healthy': {'TP': 0, 'FP': 0, 'FN': 0},
        'DR': {'TP': 0, 'FP': 0, 'FN': 0},
    }

    total = correct_ternary = correct_binary = 0
    debug_info = {
        'total_csv': 0,
        'missing_images': 0,
        'invalid_labels': 0,
        'processed': 0
    }

    # Create results directory
    results_dir = os.path.join(model_path, "inference_results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(results_dir, f"results_{timestamp}.csv")
    summary_file = os.path.join(results_dir, f"summary_{timestamp}.txt")

    with open(results_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Prediction', 'Ground Truth', 'Correct'])

    # Process CSV files first
    for csv_file in os.listdir(label_dir):
        if not csv_file.lower().endswith('.csv'):
            continue
            
        debug_info['total_csv'] += 1
        csv_path = os.path.join(label_dir, csv_file)
        base_name = os.path.splitext(csv_file)[0]
        image_file = f"{base_name}.png"
        image_path = os.path.join(images_dir, image_file)
        
        # Check image existence
        if not os.path.exists(image_path):
            debug_info['missing_images'] += 1
            print(f"Missing image file: {image_file}")
            continue
            
        # Get ground truth
        gt_label = parse_ground_truth_label(csv_path)
        if gt_label == "Unknown":
            debug_info['invalid_labels'] += 1
            continue

        # Run inference
        try:
            raw_output, prediction = run_inference(model, tokenizer, image_path)
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue
        print(raw_output)

        # Create binary labels
        binary_gt = 'DR' if gt_label in ['NPDR', 'PDR'] else gt_label
        binary_pred = 'DR' if prediction in ['NPDR', 'PDR'] else prediction
        
        # Update metrics
        total += 1
        is_correct_ternary = prediction == gt_label
        is_correct_binary = binary_pred == binary_gt
        
        correct_ternary += int(is_correct_ternary)
        correct_binary += int(is_correct_binary)

        # Update ternary stats
        for cls in ternary_stats:
            if prediction == cls:
                ternary_stats[cls]['TP' if is_correct_ternary else 'FP'] += 1
            if gt_label == cls and not is_correct_ternary:
                ternary_stats[cls]['FN'] += 1

        # Update binary stats
        for cls in binary_stats:
            pred_cls = binary_pred if cls == 'DR' else cls
            gt_cls = binary_gt if cls == 'DR' else cls
            
            if pred_cls == cls:
                binary_stats[cls]['TP' if is_correct_binary else 'FP'] += 1
            if gt_cls == cls and not is_correct_binary:
                binary_stats[cls]['FN'] += 1

        # Save results
        with open(results_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([image_file, prediction, gt_label, is_correct_ternary])

        # Real-time terminal output
        print(f"\n[{total}] Processing {image_file}")
        print(f"Ternary: Pred={prediction:<6} | GT={gt_label}")
        print(f"Binary:  Pred={binary_pred:<6} | GT={binary_gt}")

        # Print interim metrics every 10 samples
        if total % 10 == 0 and total > 0:
            # Calculate metrics
            ternary_metrics = calculate_metrics(ternary_stats)
            binary_metrics = calculate_metrics(binary_stats)
            
            print("\n" + "="*40)
            print_metrics("[Ternary Classification]",
                         ternary_metrics,
                         ['Healthy', 'NPDR', 'PDR'],
                         correct_ternary,
                         total)
            
            print("\n" + "-"*40)
            print_metrics("[Binary Classification (DR vs Healthy)]",
                         binary_metrics,
                         ['Healthy', 'DR'],
                         correct_binary,
                         total)
            print("="*40 + "\n")

    # Final safety check
    if total == 0:
        error_msg = (
            "Error: No valid samples processed\n"
            "Debug Info:\n"
            f"- Total CSV files found: {debug_info['total_csv']}\n"
            f"- Missing images: {debug_info['missing_images']}\n"
            f"- Invalid labels: {debug_info['invalid_labels']}\n"
            f"- Image directory: {images_dir} ({len(os.listdir(images_dir))} files)\n"
            f"- Label directory: {label_dir} ({len(os.listdir(label_dir))} files)"
        )
        with open(summary_file, 'w') as f:
            f.write(error_msg)
        raise RuntimeError(error_msg)

    # Calculate final metrics
    ternary_metrics = calculate_metrics(ternary_stats)
    binary_metrics = calculate_metrics(binary_stats)
    
    ternary_balanced_acc = sum(ternary_metrics[cls]['recall'] for cls in ['Healthy', 'NPDR', 'PDR']) / 3
    binary_balanced_acc = sum(binary_metrics[cls]['recall'] for cls in ['Healthy', 'DR']) / 2

    # Save summary report
    with open(summary_file, 'w') as f:
        f.write(f"Evaluation Report\n{'='*40}\n")
        f.write(f"Total Processed Images: {total}\n\n")
        
        f.write("[Ternary Classification]\n")
        f.write(f"Accuracy: {correct_ternary/total:.4f}\n")
        f.write(f"Balanced Accuracy: {ternary_balanced_acc:.4f}\n")
        for cls in ['Healthy', 'NPDR', 'PDR']:
            prec = ternary_metrics[cls]['precision']
            rec = ternary_metrics[cls]['recall']
            f.write(f"{cls}:\n  Precision={prec:.4f}\n  Recall={rec:.4f}\n")
        
        f.write("\n[Binary Classification (DR vs Healthy)]\n")
        f.write(f"Accuracy: {correct_binary/total:.4f}\n")
        f.write(f"Balanced Accuracy: {binary_balanced_acc:.4f}\n")
        for cls in ['Healthy', 'DR']:
            prec = binary_metrics[cls]['precision']
            rec = binary_metrics[cls]['recall']
            f.write(f"{cls}:\n  Precision={prec:.4f}\n  Recall={rec:.4f}\n")

if __name__ == "__main__":
    evaluate_model(
        model_path="/checkpoints/llama_3.2_11b_split_2_128_stage2", #unsloth/Llama-3.2-11B-Vision-Instruct unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit
        images_dir="/data/DCP_images",
        label_dir="/data/csv_exp_samples"
    )
