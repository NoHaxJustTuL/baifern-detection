import gradio as gr
from fastai.vision.all import *
import torchvision.transforms as T
from PIL import Image
import torch

# 1. The Labeling Function (Required for fastai to successfully unpickle the file)
def get_label(file_path):
    if 'NotBaifern' in str(file_path):
        return 'NotBaifern'
    else:
        return 'Baifern'

# 2. Load the fastai wrapper
learn = load_learner('model.pkl')
categories = learn.dls.vocab

# 3. THE BYPASS: Extract the raw PyTorch neural network
pytorch_model = learn.model
pytorch_model.eval()

# 4. Standard PyTorch Image Preprocessing (Matches fastai's default ResNet behavior)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. The Bulletproof Prediction Function
def classify_image(img):
    # Safely extract the file path regardless of what format Gradio sends
    if hasattr(img, "path"): img = img.path
    elif isinstance(img, dict) and "path" in img: img = img["path"]
    elif isinstance(img, (list, tuple)): img = img[0]
    
    # Load and convert the image to a PyTorch tensor
    pil_img = Image.open(str(img)).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0)
    
    # Predict natively with PyTorch (Bypassing fastai's pipeline)
    with torch.no_grad():
        out = pytorch_model(input_tensor)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        
    # Format the output
    confidences = {categories[i]: float(probs[i]) for i in range(len(categories))}
    pred_idx = torch.argmax(probs).item()
    
    return f"Final Answer: {categories[pred_idx]}", confidences

# 6. Define the UI
image_input = gr.Image(type="filepath") 
text_output = gr.Textbox(label="Prediction")
label_output = gr.Label(label="Confidence Levels")

intf = gr.Interface(
    fn=classify_image, 
    inputs=image_input, 
    outputs=[text_output, label_output],
    title="Baifern Detection API",
    description="Upload an image to see if it's Baifern, along with the model's confidence."
)
intf.launch()