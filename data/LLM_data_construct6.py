"""
Test script for LMaaS
"""
from openai import AzureOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

import config
from idam_token_generator import IDAMTokenGenerator


idam = IDAMTokenGenerator(
    config.IDAM_TOKEN_ENDPOINT,
    config.IDAM_APP_CLIENT_ID,
    config.IDAM_APP_CLIENT_SECRET,
    config.IDAM_LMAAS_APP_AUDIENCE
)


llm = AzureOpenAI(
        azure_endpoint = config.OPENAI_ENDPOINT,
        azure_deployment = config.OPENAI_DEPLOYMENT_MODEL,
        api_version = config.OPENAI_AZURE_API_VERSION,
        azure_ad_token = idam.get_idam_token()
    )


# load the annotations data
import json
annotation_path = "/qumulo/shared_data/aofei_summer/RegTok/data/RegAlign_data_12k.json"

with open(annotation_path, "r") as f:
    annotations = json.load(f)

image_H, image_W = 1024, 1024
def preprocess_annotations(annotation):
    processed = []
    processed_with_info = []
    for region in annotation['mask_annotations']:
        item = {
            "id": region['id'],
            "quantizer_code": region['quantizer_code']
        }
        item_with_info = {
            "id": region['id'],
            "quantizer_code": region['quantizer_code'],
            "mask_file": region.get("mask_file", ""),
            "image_id": region.get("image_id", "")
        }

        processed_bbox = [
            round(region['bbox'][1] / image_W, 3),
            round(region['bbox'][0] / image_H, 3),
            round(region['bbox'][3] / image_W, 3),
            round(region['bbox'][2] / image_H, 3)
        ]
        item['bbox'] = processed_bbox
        processed_sentences = []
        for sentence in region['sentences']:
            processed_sentences.append(sentence['raw'])
        item['sentences'] = processed_sentences
        processed.append(item)
        processed_with_info.append(item_with_info)
    return processed, processed_with_info


sampled_image_id = 0
processed_items = []
processed_items_with_info = []
for k in annotations:
    annotation = k
    processed, processed_with_info = preprocess_annotations(annotation=annotation)
    image_masks = {
        "image_id": sampled_image_id,
        "masks": processed
    }
    processed_items.append(image_masks) 
    image_masks_info = {
        "image_id": sampled_image_id,
        "image_file": annotation.get("image_file", ""),
        "num_masks": len(processed),
        "mask_code": [(item['id'], item['quantizer_code'], item['mask_file']) for item in processed_with_info]
    }

    processed_items_with_info.append(image_masks_info)
    sampled_image_id += 1

Alignment_prompt = """You are provided with region-level annotations for one or more medical images.
Each image item includes the following fields:
- "image_id": integer identifier of the image.
- "masks": a list of region objects, each containing:
  - "quantizer_code": unique identifier for this region (e.g., "M0_25").
  - "bbox": normalized bounding box coordinates [x, y, w, h] with values in [0,1].
  - "sentences": free-text region descriptions (list of strings).

Assume you can see the image implicitly, but you must rely only on the provided annotation information
(quantizer codes, bounding boxes, and sentences).

====================
Task
====================
For each input image, generate user–assistant dialogues that demonstrate how a model should reason
about the image, focusing on segmentation, detection, and localization tasks.

The output must be a single JSON list, where each element corresponds to one image:

[
  {
    "image_id": <int>,
    "dialogues": [
      {
        "User": "<user question>",
        "Assistant": "<assistant reply>",
        "mask_ids_order": [<optional list of 0-based indices>]
      },
      ...
    ]
  }
]

====================
Segmentation and Detection Rules
====================
1. When the user requests segmentation/detection/localization, the Assistant MUST:
   - Mention the corresponding quantizer codes inline, e.g., "tumor [M0_25]".
   - Include "mask_ids_order": a list of 0-based indices into the "masks" list,
     in the same order the Assistant refers to them.
2. Indices in mask_ids_order must correspond exactly to the order of "masks" in the input.

====================
Dialogue Composition Guidelines
====================
- Generate 2-4 dialogue rounds per image.
- The **first dialogue** should naturally introduce the overall task context, using a realistic and
  conversational style (not a meta or dataset tone). Examples:
  * "Let's examine this CT scan. I'll ask you to segment and detect key findings."
  * "Please review this MRI image and be ready to identify and mark important regions."
  * "We will analyze this X-ray together. I may ask for segmentation or detection of structures."
  This first round establishes the image-analysis task before the segmentation-specific requests. But notice that do not release any object information in this dialogue.
- After the introduction, you should only include those segmentation-related user intents, such as:
  * Direct segmentation/detection requests ("Please segment the tumor.")
  * Combined diagnosis + segmentation ("Identify the abnormality and segment the affected area.")
- If there are multiple masks, prioritize clinically significant regions (e.g., tumors, lesions, major organs).
  You may reference more than one mask per dialogue if appropriate.
- Only generate diagnostic questions when the provided region descriptions indicate abnormalities.
  If none are present, limit the dialogues to segmentation/detection.

====================
Behavioral Constraints
====================
- Do NOT hallucinate findings not supported by the input.
- Use only the provided sentences and bboxes for factual grounding.
- Keep responses clinically appropriate and concise.
- Never reference "annotations", "labels", or "descriptions".
  Act as if you are directly viewing the image.
  (Avoid phrases like "according to the annotations" or "as described".)
- Do not include any text outside the JSON output.
- Do not use the Unicode sequence \\u2011; use standard hyphens (-) instead (e.g., "X-ray").

====================
Example Input
====================
{
  "image_id": 0,
  "masks": [
    {"quantizer_code": "M0_16", "bbox": [0.412, 0.496, 0.389, 0.225], "sentences": ["liver"]},
    {"quantizer_code": "M0_25", "bbox": [0.555, 0.658, 0.046, 0.046], "sentences": ["tumor"]}
  ]
}

====================
Example Output
====================
[
  {
    "image_id": 0,
    "dialogues": [
      {
        "User": "Let's examine this abdomen CT scan together. I'll ask you to identify and segment key findings.",
        "Assistant": "Sure. I'm ready to analyze the scan and provide segmentation results."
      },
      {
        "User": "Please segment the liver in this image.",
        "Assistant": "The liver is segmented as [M0_16].",
        "mask_ids_order": [0]
      },
      {
        "User": "What abnormality is present in the liver? Provide the diagnosis and segment the affected area.",
        "Assistant": "Findings consistent with liver tumor (cancer) are present, segmented as [M0_25].",
        "mask_ids_order": [1]
      },
    ]
  }
]

====================
Final Reminders
====================
- Match mask_ids_order exactly to the input mask indices.
- If uncertain or inconsistent information appears, generate only segmentation or detection dialogues
  (avoid diagnosis).
- The final output must be a valid JSON list, with no extra commentary or explanatory text.
"""

if __name__ == "__main__":
    import multiprocessing

    def process_batch(args):
        batch_items, batch_items_with_info, batch_start, batch_size, Alignment_prompt = args
        import config
        from idam_token_generator import IDAMTokenGenerator
        from openai import AzureOpenAI
        from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
        import json

        idam = IDAMTokenGenerator(
            config.IDAM_TOKEN_ENDPOINT,
            config.IDAM_APP_CLIENT_ID,
            config.IDAM_APP_CLIENT_SECRET,
            config.IDAM_LMAAS_APP_AUDIENCE
        )
        llm = AzureOpenAI(
            azure_endpoint = config.OPENAI_ENDPOINT,
            azure_deployment = config.OPENAI_DEPLOYMENT_MODEL,
            api_version = config.OPENAI_AZURE_API_VERSION,
            azure_ad_token = idam.get_idam_token()
        )

        messages = [
            ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant." + Alignment_prompt + "\n\n"),
            ChatCompletionUserMessageParam(role="user", content="The input with multiple images:" + str(batch_items)),
        ]
        max_retry = 3
        num_try = 1
        while num_try <= max_retry:
            try:
                response = llm.chat.completions.create(
                    model = config.OPENAI_DEPLOYMENT_MODEL,
                    messages = messages,
                )
                llm_out = response.choices[0].message.content
                llm_out_json = json.loads(llm_out)
                # Attach extra info
                for j in range(len(batch_items)):
                    original_item = batch_items_with_info[j]
                    llm_out_json[j]['image_file'] = original_item['image_file']
                    llm_out_json[j]['mask_code'] = original_item['mask_code']
                return llm_out_json
            except Exception as e:
                # Re-init token/llm on failure
                try:
                    idam = IDAMTokenGenerator(
                        config.IDAM_TOKEN_ENDPOINT,
                        config.IDAM_APP_CLIENT_ID,
                        config.IDAM_APP_CLIENT_SECRET,
                        config.IDAM_LMAAS_APP_AUDIENCE
                    )
                    llm = AzureOpenAI(
                        azure_endpoint = config.OPENAI_ENDPOINT,
                        azure_deployment = config.OPENAI_DEPLOYMENT_MODEL,
                        api_version = config.OPENAI_AZURE_API_VERSION,
                        azure_ad_token = idam.get_idam_token()
                    )
                except Exception:
                    pass
                num_try += 1
                if num_try > max_retry:
                    print(f"Max retries exceeded for batch starting at index {batch_start}")
                    return []
        return []

    # ...existing code up to processed_items/processed_items_with_info...

    list_outputs = []
    output_json_file = "RegAlign_GPT5_mini_v12.jsonl"

    batch_size = 10
    num_workers = 20  # Adjust as needed

    # Prepare batches
    batches = []
    for i in range(950, len(processed_items), batch_size):
        batch_items = processed_items[i:i + batch_size]
        batch_items_with_info = processed_items_with_info[i:i + batch_size]
        batches.append((batch_items, batch_items_with_info, i, batch_size, Alignment_prompt))

    with multiprocessing.Pool(num_workers) as pool, open(output_json_file, "a") as ans_file:
        from tqdm import tqdm
        for llm_out_json in tqdm(pool.imap(process_batch, batches), total=len(batches)):
            if not llm_out_json:
                continue
            list_outputs.extend(llm_out_json)
            ans_file.write("\n".join([json.dumps(x) for x in llm_out_json]) + "\n")
            ans_file.flush()

    print("Finish the data construction!")