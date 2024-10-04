import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import validators
from io import BytesIO
from threading import Thread

# Load model, processor, and streamer once, to avoid reloading on each request
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)
streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

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
        return_tensors="pt",
    ).to(model.device)

    # Generation config
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)

    # Thread to run the model's generate function and stream results
    def generate_in_thread():
        model.generate(**generation_kwargs)

    thread = Thread(target=generate_in_thread)
    thread.start()

    # Stream the results as they are generated
    for new_text in streamer:
        yield new_text


@csrf_exempt
def ask(request):
    if request.method == "POST":
        user_input = request.POST.get("user_input")
        image_input = request.POST.get("image_input")
        print("question: ", user_input)

        if validators.url(image_input):
            image = Image.open(requests.get(image_input, stream=True).raw)
        elif "image_file" in request.FILES:
            image_file = request.FILES["image_file"]
            image = Image.open(BytesIO(image_file.read()))
        else:
            return StreamingHttpResponse("No valid image URL or file provided.", content_type='text/plain')

        # Streaming the model's response
        response = StreamingHttpResponse(
            _ask(user_input, image),
            content_type='text/plain',
        )
        return response
    else:
        return StreamingHttpResponse("Invalid request method", content_type='text/plain')
