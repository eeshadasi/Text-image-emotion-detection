from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import torch
from PIL import Image
import gradio as gr
import requests
from io import BytesIO

# Load the BLIP model for image captioning
model_name = "Salesforce/blip-image-captioning-base"

try:
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading BLIP model components: {e}")
    raise

# Load emotion detection models
emotion_models = [
    pipeline('text-classification', model='bhadresh-savani/bert-base-uncased-emotion', return_all_scores=True),
    pipeline('text-classification', model='cardiffnlp/twitter-roberta-base-emotion', return_all_scores=True),
    pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True),  # Model starting with 'J'
]

# Example weights: Adjust based on model performance
model_weights = [1.0, 0.8, 1.2]  # Give higher weight to models that perform better

def generate_caption(image):
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def blend_model_predictions(caption):
    emotion_scores = {}

    for model, weight in zip(emotion_models, model_weights):
        results = model(caption)
        for result in results:
            for emotion in result:
                label = emotion['label']
                score = emotion['score'] * weight
                if label not in emotion_scores:
                    emotion_scores[label] = 0
                emotion_scores[label] += score

    # Average the scores across models
    averaged_scores = {label: score / len(emotion_models) for label, score in emotion_scores.items()}

    # Choose the top emotion based on averaged scores
    top_emotion = max(averaged_scores.items(), key=lambda x: x[1])
    return top_emotion

def predict_step(image):
    try:
        caption = generate_caption(image)
        emotion, _ = blend_model_predictions(caption)
        return caption, emotion
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error in processing", "N/A"

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create Gradio interface with only image upload
with gr.Blocks() as demo:
    with gr.Column():
        image_input = gr.Image(type='pil', label='Upload an Image')
        label_output = gr.Text(label='Generated Caption')
        emotion_label_output = gr.Text(label='Detected Emotion')

    gr.Button("Generate").click(
        fn=predict_step,
        inputs=image_input,
        outputs=[label_output, emotion_label_output]
    )

if __name__ == '__main__':
    try:
        demo.launch()
    except Exception as e:
        print(f"Error launching Gradio interface: {e}")
