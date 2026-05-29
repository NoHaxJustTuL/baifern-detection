import gradio as gr
from fastai.vision.all import *

# 1. Your custom Baifern labeling function (Required for the pickle file to load)
def get_label(file_path):
    if 'NotBaifern' in str(file_path):
        return 'NotBaifern'
    else:
        return 'Baifern'

# 2. Load your trained model
learn = load_learner('model.pkl')
categories = learn.dls.vocab

# 3. Define the prediction function
def classify_image(img):
    pred, idx, probs = learn.predict(img)
    final_answer = f"Final Answer: {str(pred)}"
    confidences = dict(zip(categories, map(float, probs)))
    return final_answer, confidences

# 4. Set up the Gradio Interface components
# This type="pil" flag fixes all the input bugs!
image_input = gr.Image(type="pil") 
text_output = gr.Textbox(label="Prediction")
label_output = gr.Label(label="Confidence Levels")

# 5. Build and launch the web app
intf = gr.Interface(
    fn=classify_image, 
    inputs=image_input, 
    outputs=[text_output, label_output],
    title="Baifern Detection API",
    description="Upload an image to see if it's Baifern, along with the model's confidence."
)
intf.launch()