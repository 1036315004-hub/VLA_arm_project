import torch
from .vision_encoder import get_dinov2_vit_s_14_reg_hf

def test_encoder():
    # Load model and processor from learning module
    model, processor = get_dinov2_vit_s_14_reg_hf()

    if model and processor:
        # Example: Process a dummy image (B, C, H, W)
        dummy_image = torch.randn(1, 3, 224, 224)
        inputs = processor(images=dummy_image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            # Use last_hidden_state or pooler_output
            features = outputs.last_hidden_state
            print(f"Sucessfully extracted features with shape: {features.shape}")

if __name__ == "__main__":
    test_encoder()

