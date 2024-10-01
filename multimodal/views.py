import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, TextStreamer
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import validators
from io import BytesIO

# Load model, processor, and streamer once, to avoid reloading on each request
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)
streamer = TextStreamer(processor)

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

def _ask(question, image):
    # Prepare the input message with the image and text
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]}
    ]
    
    # Process the input with the processor
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_output = model.generate(**inputs, streamer=streamer)
    
    for token in generated_output:
        yield token

@csrf_exempt
def ask(request):
    if request.method == "POST":
        user_input = request.POST.get("user_input")
        image_input = request.POST.get("image_input")

        if validators.url(image_input):
            image = Image.open(requests.get(image_input, stream=True).raw)
        elif "image_file" in request.FILES:
            # Read and open the image file from the POST request
            image_file = request.FILES["image_file"]
            image = Image.open(BytesIO(image_file.read()))
        else:
            return StreamingHttpResponse("No valid image URL or file provided.", content_type='text/plain')

        # Stream the model's response as plain text
        response = StreamingHttpResponse(
            _ask(user_input, image),
            content_type='text/plain',
        )
        return response
    else:
        return StreamingHttpResponse("Invalid request method", content_type='text/plain')
