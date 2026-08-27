from . import cap_generation 
import collections
import os
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from PIL import Image


def bleu(model, val_df, vocab, transform, images_dir, device, max_length=20):
    model.eval()

    # Group ground-truth references by image filename
    image_to_references = collections.defaultdict(list)
    for _, row in val_df.iterrows():
        img_id = row["image"]
        ref_tokens = vocab.tokenize(row["caption"])
        image_to_references[img_id].append(ref_tokens)

    references_list = []
    hypotheses_list = []

    unique_images = list(image_to_references.keys())

    for img_id in unique_images:
        img_path = os.path.join(images_dir, img_id)

        # Generate text tokens using the single image function logic
        caption_str = cap_generation.generate_caption_attention( model, img_path, transform, vocab, device, max_length)
        print(caption_str)      # ---------------------------------------------- Debug ---------------------------------------------- 

        predicted_tokens = caption_str.split()

        references_list.append(image_to_references[img_id])
        hypotheses_list.append(predicted_tokens)

    # Compute BLEU Scores
    smooth = SmoothingFunction().method1
    scores = {
        "BLEU-1": corpus_bleu(references_list, hypotheses_list, 
                              weights=(1.0, 0, 0, 0), smoothing_function=smooth,)* 100,

        "BLEU-2": corpus_bleu(references_list, hypotheses_list,
                              weights=(0.5, 0.5, 0, 0),)* 100,

        "BLEU-3": corpus_bleu(references_list, hypotheses_list,
                              weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth,)* 100,

        "BLEU-4": corpus_bleu(references_list, hypotheses_list,
                              weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth,)* 100,}
    return scores