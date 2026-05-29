import gradio as gr
from fastai.vision.all import *

# 1. Load model
learn = load_learner('model.pkl')

# Extract labels (['Baifern', 'NotBaifern'])
categories = learn.dls.vocab

# 2. Define the prediction function
def get_confidence_levels(img):
    # predict() returns the prediction, the index, and the raw probabilities.
    _, _, probs = learn.predict(img)
    
    # Create a dictionary mapping the categories to their confidence levels
    # Result looks like: {"Baifern": 0.98, "NotBaifern": 0.02}
    return dict(zip(categories, map(float, probs)))

# 3. Create the API interface
intf = gr.Interface(
    fn=get_confidence_levels, 
    inputs=gr.Image(type="pil"), # Expects an image, converts it to PIL format
    outputs=gr.JSON()            # Outputs a raw JSON dictionary
)

intf.launch()
# Triggering CI/CD pipeline# Triggering HF CLI Pipeline
 
