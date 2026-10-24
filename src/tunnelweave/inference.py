"""Image-only automatic recognition, segmentation, and optional descriptions."""
import torch

from .predictions import assemble_predictions
from .query import prepare_fixed_queries


def forward_image(model, processor, image):
    device = next(model.parameters()).device
    inputs, query = prepare_fixed_queries(processor, image, device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model(image, inputs["input_ids"], inputs["pixel_values"],
                       inputs["image_grid_thw"], inputs.get("attention_mask"))
    return output, query


@torch.inference_mode()
def predict(model, processor, image, descriptions=False, threshold=0.5):
    was_training = model.training
    model.eval()
    try:
        output, query = forward_image(model, processor, image)
        result = assemble_predictions(output, presence_threshold=threshold, mask_threshold=threshold)
        result["query"] = query
        if descriptions:
            from .descriptions import describe_image
            result["known_descriptions"] = describe_image(model.base, processor, image,
                result["known_predictions"], mode="known", max_new_tokens=512)
            result["open_observation"] = describe_image(model.base, processor, image,
                (), mode="open_observation", max_new_tokens=512)
        return result
    finally:
        model.train(was_training)
