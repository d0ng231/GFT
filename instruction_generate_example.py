import os
import json
import pandas as pd
from openai import OpenAI
from io import StringIO

OPENAI_API_KEY = 'XXX'
client = OpenAI(api_key=OPENAI_API_KEY)

def parse_custom_csv(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        content = f.read()

    sections = content.split("\n\n")
    data = {}

    # Parse prediction section
    pred_df = pd.read_csv(StringIO(sections[0]))
    data['prediction'] = pred_df.to_dict(orient='records')[0]

    # Parse density statistics
    density_df = pd.read_csv(StringIO(sections[1].split("Density Statistics:\n")[1]))
    data['density_stats'] = density_df.iloc[0].to_dict()

    # Parse top nodes
    nodes_df = pd.read_csv(StringIO(sections[2].split("Top Nodes:\n")[1]))
    data['top_nodes'] = nodes_df.to_dict(orient='records')

    # Parse top edges
    edges_df = pd.read_csv(StringIO(sections[3].split("Top Edges:\n")[1]))
    data['top_edges'] = edges_df.to_dict(orient='records')

    return data

def construct_promp_1(image_data, overlay_image_url, raw_image_url):
    system_content = (
        "You are a retinal specialist creating teaching cases. The task is to prediction whether an OCTA DVC image suggests healthy, non-proliferate diabetic retinopathy (NPDR), or proliferate diabetic retinopathy (PDR).  Follow these rules:\n"
        "1. Analyze BOTH raw OCTA deep capillary plexus (DVC) image and overlay image (with color-coded vessels) to triangulate findings\n"
        "2. Describe features ONLY as visible on raw DVC, but use overlay to:\n"
        "   - Confirm abnormality locations\n"
        "   - Identify subtle flow variations\n"
        "3. Translate overlay colors to OCTA features:\n"
        "   Red: High DR relevance → Look for vessel irregularities in raw\n"
        "   Blue: Normal → Verify regular capillary patterns\n"
        "   Large dark areas besides the FAZ → could be capillary dropout or non-perfusion\n"
        "4. Use 3-level localization:\n"
        "   a) Clock position from quadrant data\n"
        "   b) FAZ-relative distance\n"
        "   c) Depth-specific features (DVC layer)"
    )

    overlay_guidance = (
        "Overlay Decoding Key (For Your Reference):\n"
        "1. Color-Intensity Correlation:\n"
        "   - Red/Orange: High probability DR-related changes\n"
        "   - Blue: Normal vascular patterns\n"
        "2. Spatial Mapping Rules:\n"
        "   a) Examine high-color areas carefully\n"
        "   b) Cross-validate density stats with color intensity\n"
    )

    density_analysis = (
        "Density Guidance:\n"
        f"Quadrant Density Comparison):\n"
        f"Top-left: {image_data['density_stats']['Node Density (TL)']:.3f} | "
        f"Top-right: {image_data['density_stats']['Node Density (TR)']:.3f}\n"
        f"Bottom-left: {image_data['density_stats']['Node Density (BL)']:.3f} | "
        f"Bottom-right: {image_data['density_stats']['Node Density (BR)']:.3f}\n"
        "Low density areas may show capillary dropout - appear as dark regions with: \n"
        "- Loss of capillary continuity\n"
        "- Reduced branching complexity\n"
        "- Larger intercapillary spaces"
    )

    feature_guidance = (
        "Clinical Feature:\n"
        "1. FAZ Analysis\n"
        "   - Centration: Compare node distribution (x_percent,y_percent)\n"
        "   - Border regularity: Look for abrupt density changes and abnormal vessels\n"
        "   - Size: Relate to quadrant density averages\n\n"
        
        "2. Perfusion Patterns (Use edge densities):\n"
        "   - Look for dropout regions in low edge density areas \n"
        "   - Describe shape (wedge-shaped, patchy, confluent)\n"
        "   - Note proximity to FAZ\n\n"
        
        "3. Vascular Morphology (explanation of top nodes/edges features, be very carefully with these features in the table as they are likely NOT accurate, only mention when you can confirm that on images):\n"
        
                "Feature explanations: (note that these are here to help the teacher understand the feature, do not copy directly, and the features listed in the table are not always the correct ones, always describe what can be seen on the raw OCTA image.) \n"
        
                "volume - (Vessel segment area) The volume feature describes the volume of a vessel segment, i.e. the number of voxels that make up the vessel segment, i.e. the number of pixels assigned to the vessel segment. Large values of this feature can be indicative of vessels originating from neovascularization, which occurs in patients with proliferative diabetic retinopathy. Other reasons for large volumes may be the elimination of bifurcations due to capillary dropout, resulting in long and therefore voluminous vessel segments. This is an indicator of diabetic retinopathy. \n"
        
                "length – (Vessel segment centerline length) The length feature describes the length of the centerline of a vessel segment. Large values indicate a long centerline, and small values indicate a short centerline. A local accumulation of vessel segments with small length values that are interconnected indicate high network complexity. High network complexity is an indicator of healthiness. Conversely, a local accumulation of vessel segments with high length values may indicate capillary dropout and areas of low vessel network complexity. This is an indicator of diabetic retinopathy. \n"
        
                "distance – (Euclidean distance between vessel segment start and end point) The distance feature describes the Euclidean distance between the start and end points of a vessel segment. A local accumulation of vessel segments with small distance values that are interconnected indicate high network complexity. High network complexity is an indicator of healthiness. Conversely, a local accumulation of vessel segments with small distance values may indicate capillary dropout and areas of low vessel network complexity. This is an indicator of diabetic retinopathy. \n"
        
                "curveness - (Ratio of the length and distance feature) The curveness features describes the ratio of the length of the vessel centerline to the Euclidean distance between the start and end points of a vessel segment. Low values indicate that the vessel is not curved, which is a typical characteristic of healthiness. Loop-like structures result in very high curveness values. Loops and therefore large curveness values are associated with neovascularization, which is present in proliferative diabetic retinopathy.\n"
        
                "node1_degree - (Number of connected vessel segments at the start/end point of the vessels with the lower number of connected vessels) The node1_degree feature describes the number of connected branches at the start/end point of the vessel segment with the lower number of branches. Vessel segments can either end with a bifurcation into two or more other vessels, or they are terminal vessels. Large terminal vessels are a sign of capillary dropout. In healthy individuals, a large number of small terminal capillaries are expected. \n"
        
                "node2_degree - (Number of connected vessel segments at the start/end point of the vessels with the higher number of connected vessels) The node2_degree feature describes the number of connected branches at the start/end point of the vessel segment with the higher number of branches. Vessel segments may either end with a bifurcation into two or more other vessels, or they may be terminal vessels. If both ends of the vessel segments are terminal, this is most likely associated with under-segmentation due to abnormal changes in the vasculature (e.g. stenosis). \n"
        
                "degree - (Total number of connected vessel segments, sum of all connected vessels at the starting and end point of the vessel) The degree feature describes the number of branches that are connected to the starting and end points of the vessel segment. The degree is related to the complexity of the vascular network. High complexity is associated with a healthy vasculature, therefore areas with many vessel segments of high degree, are healthy areas. Areas with few vessel segments of low degree are areas of low complexity associated with disease. \n"
        
                "avgCrossSection - (Average thickness of a vessel segment along its centerline) The avgCrossSection feature in 2D projections is the average thickness of a vessel segment along its centerline. \n"
        
                "avgRadiusAvg - (Average of the average radius along a vessel segment centerline) The avgRadiusAvg feature in 2D projection is the average of the average radius values along all the points on the centerline. In 2D the feature corresponds to the avgCrossSection feature.\n"
        
                "avgRadiusStd - (Standard deviation of the average radius along a vessel segment centerline) The avgRadiusStd feature describes the standard deviation of the average radius along all the points on the centerline. A large standard deviation indicates irregularities in the thickness of the vessel segment. This may indicate stenosis or unnatural bulging within the vessel segment. High values are therefore indicative of abnormalities and therefore often of disease. Low values indicate a regular shape of the vessel and are therefore associated with healthiness. \n"
        
                "minRadiusAvg - (Average of the minimal radius along a vessel segment centerline) The minRadiusAvg feature describes the average of the minimal radius values along all the points in the centerline.\n"
        
                "MinRadiusStd - (Standard deviation of the minimal radius along a vessel segment centerline) The minRadiusAvg feature describes the standard deviation of the minimal radius values along all the points in the centerline. A large standard deviation indicates irregularities in the thickness of the vessel segment. This can indicate stenosis or unnatural bulges within the vessel segment. High values are therefore an indicator of abnormalities and therefore often disease. Low values indicate a regular shape of the vessel and are therefore associated with healthiness.\n"
        
                "roundnessAvg - (Average of the ratio of minimal and maximal radius for every point along the vessel segment centerline) The roundnessAvg features describes the average of the roundness (defined as the ratio of minimal and maximal radius) for every point along the centerline. High values indicate that the vessel segment has an irregular outline. Low values indicate a normal structured vessel segment.\n"
        
                "roundnessStd - (Standard deviation of the ratio of minimal and maximal radius for every point along the vessel segment centerline) The roundnessStd features describes the average of the roundness (defined as the ratio of minimal and maximal radius) for every point along the centerline.\n"
        
                "hetero_degree - (Number of neighboring intercapillary areas of a vessel segment) The hetero_degree features measures how many distinct intercapillary areas are directly neighboring a vessel segment. Terminal vessels have the lowest hetero_degree.\n"
                
    )

    user_content = [
        {
            "type": "text",
            "text": (
                f"Ground Truth condition: {image_data['prediction']['Predicted Class']} | "
                f"Probabilities of prediction - H: {image_data['prediction']['Healthy Probability']:.3f}, "
                f"NPDR: {image_data['prediction']['NPDR Probability']:.3f}, "
                f"PDR: {image_data['prediction']['PDR Probability']:.3f}\n\n"
                
                f"{density_analysis}\n\n"
                f"{feature_guidance}\n\n"
                f"{overlay_guidance}\n\n"

                # "Top Nodes (Can be inaccurate, only use for verification!!!!):\n"
                # f"1. {image_data['top_nodes']}\n"
                # 
                # "Top Edges (Can be inaccurate, only use for verification!!!!):\n"
                # f"1. {image_data['top_edges']}\n"

                 "Based on the ground truth condition given, generate 4 Q&A with these requirements:\n"
                "1. FAZ Question:\n"
                "   - Example: 'Describe the FAZ morphology and surrounding capillaries'\n"
                "   - To answer the question, check Overlay and raw DVC for vessel patterns along the central FAZ region\n"

                "2. Perfusion Question:\n" 
                "   - Example: 'Identify any areas of reduced perfusion\n"
                "   - To answer the question, check relatively low-density areas appearing as blank non-colored regions\n"

                "3. Vascular Question:\n"
                "   - Example: 'Note any abnormal vascular structures'\n"
                "   - To answer the question,look for Microaneurysms and caliber variation on DVC image\n"

                "4. Final Question (Diagnosis):\n"
                "   - Summarize and put together all observations\n"
                "Example: 'Based on thorough analysis, what is your diagnosis?'\n\n"
                
                " Requirements:\n"
                "1. For DR cases, use clock/quadrant positions or general descriptors to describe the patterns:\n"
                "2. For Healthy cases, use general descriptors:\n"
                "   - 'well-preserved vascular network'\n"
                "   - 'uniform capillary density'\n"
                "3. Use OCTA-specific terms, for example:\n"
                "   - 'flow voids' not 'dark areas'\n"
                "   - 'capillary non-perfusion' not 'dropout'\n"
                "4. Required findings localization methods (if the pattern is very obvious on the image, otherwise use general descriptors):\n"
                "   - Clock positions \n"
                "   - FAZ-relative \n"
                "   - Quadrant sector \n"
                "5. Never refer to the color of the overlay or existing number in the prompt. ONLY use them to verify your findings on the raw DVC image!!!\n"

                "Generate EXACTLY 4 Q&A pairs with STRICT format requirements:\n"
                "1. Follow this pattern without deviation:\n"
                "User: [Question 1 about FAZ]\n"
                "Assistant: [Answer 1]\n"
                "User: [Question 2 about perfusion]\n"
                "Assistant: [Answer 2]\n"
                "User: [Question 3 about vasculature]\n"
                "Assistant: [Answer 3]\n"
                "User: [Final diagnostic question]\n"
                "Assistant: [Put together all the important finding, give diagnosis and reasoning]\n\n"
                
                #  "...."

                
                "Critical Format Rules:\n"
                "- MUST contain exactly 4 User-Assistant pairs\n"
                "- NO additional text before/after the Q&A pairs\n"
                "- NEVER combine multiple questions in one User turn\n"
                "- ALWAYS maintain alternating User/Assistant sequence\n"
                
                "Keep the answer short and concise, only mention the specific area/quadrant when you very sure the feature or pattern can be observed on the DVC image. Never mention the existence of the overlay, they are for teacher model's (your) reference only.\n"
                "The following are the overlay image and raw DVC images, respectively."
            )
        },
        {
            "type": "image_url",
            "image_url": {"url": overlay_image_url, "detail": "high"}
        },
        {
            "type": "image_url", 
            "image_url": {"url": raw_image_url, "detail": "high"}
        }
    ]

    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


def generate_conversation(messages):
    response = client.chat.completions.create(
        model="o1", 
        messages=messages
    )
    return response.choices[0].message.content

def parse_conversation(conversation_text):
    # Clean and normalize the input
    cleaned_lines = []
    for line in conversation_text.split('\n'):
        line = line.strip()
        if line.startswith(('User:', 'Assistant:')):
            cleaned_lines.append(line)
        elif line and cleaned_lines:
            cleaned_lines[-1] += ' ' + line  # Merge continuation lines

    # Strict QA pair extraction
    qa_pairs = []
    current_q, current_a = None, None
    
    for line in cleaned_lines:
        if line.startswith('User:'):
            if current_q and current_a:  # Save previous pair
                qa_pairs.extend([
                    {'from': 'human', 'value': current_q},
                    {'from': 'gpt', 'value': current_a}
                ])
            current_q = line[len('User:'):].strip()
            current_a = None
        elif line.startswith('Assistant:'):
            current_a = line[len('Assistant:'):].strip()
    
    # Add the final pair
    if current_q and current_a:
        qa_pairs.extend([
            {'from': 'human', 'value': current_q},
            {'from': 'gpt', 'value': current_a}
        ])

    # Validate exactly 4 QA pairs
    if len(qa_pairs) != 8:
        return []
    
    return qa_pairs

def save_conversation(conversation_text, base_name):
    conversations = parse_conversation(conversation_text)
    if len(conversations) != 8:  # 4 pairs × 2 entries per pair
        return None
    
    return {
        'id': base_name,
        'conversations': conversations
    }


def main():
    csv_dir = ''
    # csv_dir = ''
    output_file = ''

    with open(output_file, 'w', encoding='utf-8') as f:
        for filename in os.listdir(csv_dir):
            if filename.endswith('.csv'):
                csv_path = os.path.join(csv_dir, filename)
                base_name = os.path.splitext(filename)[0]

                # Generate image URLs
                overlay_url = f""
                raw_url = f""

                # Process data
                image_data = parse_custom_csv(csv_path)
                prompt = construct_promp_1(image_data, overlay_url, raw_url)
                conversation = generate_conversation(prompt)
                data_entry = save_conversation(conversation, base_name)

                f.write(json.dumps(data_entry, ensure_ascii=False) + "\n")
                print(f"Processed {filename}")


if __name__ == "__main__":
    main()