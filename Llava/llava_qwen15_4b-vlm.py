from transformers import AutoProcessor, AutoModelForPreTraining, LlavaForConditionalGeneration
import torch
from PIL import Image


"""
Codes from the following website:
https://huggingface.co/yuanzhoulvpi/llava_qwen15-4b-chat_openai-clip-vit-large-patch14-336-V2

Model download cmdline:
pip install -U huggingface-cli
huggingface-cli download --resume-download llava-hf/llava-1.5-7b-hf --local-dir your-path --local-dir-use-symlinks False
"""

# Specify the directory where the model will be downloaded
model_directory = "../Downloaded_Models/llava-1.5-7b-hf"
cache_directory = "../Cache/llava-1.5-7b-hf"
device = "cuda:0"
print(torch.cuda.is_available())

# Load the processor and model, specifying the cache directory
processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path=model_directory,
    cache_dir=cache_directory
)

# Used for finetune
# model = AutoModelForPreTraining.from_pretrained(
#     "yuanzhoulvpi/llava_qwen15-4b-chat_openai-clip-vit-large-patch14-336-V2",
#     cache_dir=model_directory
# )


# Used for text generation
model = LlavaForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_directory,
    cache_dir=cache_directory,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

prompt = "USER: <image>\n What's the content of the image, is there any potential hazards? ASSISTANT:"
url = "./Test_Data/Bike_Dog.jpg"
image = Image.open(fp=url)

inputs = processor(text=prompt, images = image, return_tensors="pt")

generate_ids = model.generate(**inputs, max_new_tokens=65)
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(response)