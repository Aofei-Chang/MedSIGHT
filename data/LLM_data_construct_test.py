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
annotation_path01 = "/qumulo/shared_data/aofei_summer/RegTok/data/BiomedParse_SegVQA_Diagnosis_30k.json"

annotations = []
# load jsonlines
with open(annotation_path01, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

image_H, image_W = 1024, 1024
def preprocess_annotations(annotation):
    processed = []
    processed_with_info = []
    for region in annotation['mask_annotations']:
        item = {
            "id": region['id'],
        }
        item_with_info = {
            "id": region['id'],
            "mask_file": region.get("mask_file", ""),
            "image_id": region.get("image_id", "")
        }

        processed_bbox = [
            round(region['bbox'][1] / image_W, 3),
            round(region['bbox'][0] / image_H, 3),
            round(region['bbox'][3] / image_W, 3),
            round(region['bbox'][2] / image_H, 3)
        ]
        # item['bbox'] = processed_bbox
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
    processed, processed_with_info = preprocess_annotations(annotation=annotations[k])
    annotation = annotations[k]
    image_masks = {
        "image_id": sampled_image_id,
        "masks": processed
    }
    processed_items.append(image_masks) 
    image_masks_info = {
        "image_id": sampled_image_id,
        "image_file": annotation.get("image_file", ""),
        "modality": annotation.get("modality", ""),
        "num_masks": len(processed),
        "mask_id": [(item['id'], item['mask_file']) for item in processed_with_info]
    }

    processed_items_with_info.append(image_masks_info)
    sampled_image_id += 1

Alignment_prompt = """
You are an expert radiologist and dataset curator. 
You are given region-level annotations for one or more medical images. Each image item includes:
- "image_id": an integer id,
- "masks": a list of region objects, each with:
    - "id": id of this mask
    - "sentences": free-text descriptions (list of strings).

Assume you can see the image implicitly and must use only the provided annotation information (sentences) to perform the tasks below. 

Task (produce a single JSON output per image):
- Generate evaluation data in the form of VQA about the image and segmentation based on the provided region information.
Follow the style of radiology and clinical reasoning, and make sure the generated questions are natural, factual, and unambiguous.

Output rules and format (strict):
- Always return a JSON list containing one object per input image: [{...}, {...}, ...].
- Each image object must contain:
  {
    "image_id": <the input image_id>,
    "QAs": [ <list of Q&A objects> ]
  }
- Each dialogue turn is a JSON object:
  {
    "User": "<user question>",
    "Assistant": "<assistant answer>",
    "mask_ids": [ <list of mask indices appearing in the answer> ],
    "Question_type": "open" or "close"
  }

QA generation rules:
- For each image, generate one diagnostic Q&A pair for each provided mask.
- Each QA serves with 2 evaluation targets: VQA and segmentation.
- For each QA, include a short version of answer for the convenience of evaluation VQA (short diagnosis for open-ended VQA).
- For the mask with only organ and without abnormalities, you should skip it by generating empty QAs.
- For the mask with abnormalities, you may use open-ended questions to ask the diagnosis.
- Always include the corresponding mask_id(s) in the output key "mask_ids", do not include it in the ground truth answer.

**Notice:**
- There will be images with multiple abnormality masks, in this case, please identify the central or most important one (some of them might be one object) for the diagnosis question, if you can not identify such one, just leave the QAs as empty.
- In segmentation prompts, please explicitly use words like "segment xxx" or "segmentations" to let the model know segmentation is required.

**One special case for brain tumor MRI: **
- there may be three masks including enhancing tumor, non-enhancing tumor and tumor core. In this case, your answer should be brain tumor (incuding non-enhancing and enhancing tumor), and the recorded "mask_ids" should be a (sub) list of these three mask ids(e.g., [id1, id2, id3]) as one item in the list[[id1, id2, id3]].
- Focus on whole (overall) diagnosis if there are multiple abnormality masks, for example, if there are both enhancing, non-enhancing tumors and whole tumor, you should focus on whole tumor.

Example input of one image (for reference only):
{
  "image_id": 0,
  "masks": [
    {"id": 0, "sentences": ["liver"]},
    {"id": 1, "sentences": ["tumor"]},
    {"id": 2, "sentences": ["spleen"]},
  ]
}

Example output (for one image):
[
  {
    "image_id": 0,
    "QAs": [
      {
        "User": "What abnormality is seen on the liver? Please do diagnosis and then segment it if it exists.",
        "Assistant": "There is a tumor at the center right lobe, segmented as <mask> inside the liver.",
        "mask_ids": [1],
        "short answer": "Tumor",
        "Question_type": "open"
      },
    ]
  }
]

Please do not mention words like "annotations", "annotated regions" in the generated QA.

Finally: The API will provide the "masks" list as input. Produce the JSON outputs (one object per image) strictly following the rules above. Do not include any extra text outside the JSON list in the model's final reply. """



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
                    llm_out_json[j]['mask_id'] = original_item['mask_id']
                    llm_out_json[j]['modality'] = original_item['modality']
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
    output_json_file = "BiomedParse_SegVQAD_GPT5_30k_v2.jsonl"

    batch_size = 10
    num_workers = 10  # Adjust as needed

    # Prepare batches
    batches = []
    for i in range(0, len(processed_items), batch_size):
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