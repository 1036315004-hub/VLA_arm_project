import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import torch
from transformers import AutoImageProcessor, Dinov2Model

# Get the project root directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

def get_dinov2_vit_s_14_reg_hf(pretrained=True):
    """
    Loads DINOv2 ViT-S/14 with registers from Hugging Face Transformers.

    Weights are saved in models/pretrained/huggingface
    """
    # 1. Specify checkpoint directory
    checkpoint_dir = os.path.join(PROJECT_ROOT, "models", "pretrained", "huggingface")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 2. Model ID for ViT-S/14 with registers
    model_id = "facebook/dinov2-small-with-registers"

    print(f"[*] Loading DINOv2 {model_id} from Hugging Face...")
    print(f"[*] Cache location: {checkpoint_dir}")

    try:
        # Load model using PyTorch weights
        model = Dinov2Model.from_pretrained(
            model_id,
            cache_dir=checkpoint_dir,
            local_files_only=False,
            from_pt=True
        )

        # Manually create a processor if AutoImageProcessor fails
        from transformers import ViTImageProcessor
        processor = ViTImageProcessor(
            do_resize=True,
            size={"shortest_edge": 224},
            do_center_crop=True,
            crop_size={"height": 224, "width": 224},
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            resample=3 # BICUBIC
        )

        print(f"[+] DINOv2 {model_id} loaded successfully.")

        if not pretrained:
            model.init_weights()

        return model, processor
    except Exception as e:
        print(f"[-] Error loading model from Hugging Face: {e}")
        # Fallback to standard small if registers version naming is slightly different
        try:
            print("[*] Attempting fallback to facebook/dinov2-small...")
            model_id = "facebook/dinov2-small"
            processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=checkpoint_dir)
            model = Dinov2Model.from_pretrained(model_id, cache_dir=checkpoint_dir)
            return model, processor
        except Exception as e2:
            print(f"[-] Fallback failed: {e2}")
            return None, None

if __name__ == "__main__":
    # Test loading
    model, processor = get_dinov2_vit_s_14_reg_hf()
    if model:
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Total parameters: {num_params:.2f} M")

        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        inputs = processor(images=dummy_input, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            print(f"Output shape: {last_hidden_states.shape}")
