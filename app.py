import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

# Load the saved model
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust to match the original model's output units
model.load_state_dict(torch.load('grass_wood_classification_model.pth'))
model.eval()

# Create a new model with the correct final layer
new_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
new_model.fc = nn.Linear(new_model.fc.in_features, 2)  # Adjust to match the desired output units

# Copy the weights and biases from the loaded model to the new model
new_model.fc.weight.data = model.fc.weight.data[0:2]  # Copy only the first 2 output units
new_model.fc.bias.data = model.fc.bias.data[0:2]

# Define the preprocessing function
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
    return input_batch

# Define the prediction function
def predict(image):
    input_batch = preprocess_image(image)
    new_model.eval()
    with torch.no_grad():
        output = new_model(input_batch)
    _, predicted_class = output.max(1)
    class_names = ['grass', 'wood']
    predicted_class_name = class_names[predicted_class.item()]
    return predicted_class_name

# Create the Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label="Upload an Image"),  # Use 'pil' to match the input type
    outputs="text",
    title="Grass or Wood Classifier Using ResNet18",
    description="Upload an image to classify it as either grass or wood."
)

# Launch the interface
demo.launch(share=True)