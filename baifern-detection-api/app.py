import gradio as gr
from fastai.vision.all import *

# 1. Load the model
learn = load_learner('model.pkl')
categories = learn.dls.vocab

# 2. Define the robust prediction function
def get_confidence_levels(img):
    # SAFETY CATCH 1: If Gradio hands us a dictionary via the JS API, extract the raw file path
    if isinstance(img, dict) and "path" in img:
        img = img["path"]
        
    # SAFETY CATCH 2: Force the input (whether it is a path, a numpy array, or a PIL image) 
    # into a native fastai PILImage object
    img_fastai = PILImage.create(img)
    
    # 3. Predict!
    pred, pred_idx, probs = learn.predict(img_fastai)
    
    # Return a clean dictionary of the confidences
    return dict(zip(categories, map(float, probs)))

# 4. Define the UI/API
# Setting type="filepath" is the most stable method for fastai compatibility
image = gr.Image(type="filepath")
output = gr.JSON()

# 5. Launch
iface = gr.Interface(fn=get_confidence_levels, inputs=image, outputs=output)
iface.launch()