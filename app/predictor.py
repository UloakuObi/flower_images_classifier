import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights
from PIL import Image


# Class names in the order the model was trained on
class_names = [
    'astilbe', 'bellflower', 'black-eyed susan', 'calendula',
    'california poppy', 'carnation', 'common daisy', 'coreopsis',
    'dandelion', 'iris', 'rose', 'sunflower', 'tulip', 'water lily'
]

num_classes = len(class_names)

trained_model = None

# Load the pre-trained ResNet model
class DenseNetClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the classifier layer (can be adjusted to earlier layers if needed)
        for param in self.model.features.denseblock4.parameters():
            param.requires_grad = True

        # Replace the classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    global trained_model

    if trained_model is None:
        trained_model = DenseNetClassifier(num_classes=num_classes)
        checkpoint = torch.load("model/densenet_flower_classifier_v1.pth", map_location="cpu")
        trained_model.load_state_dict(checkpoint['model_state_dict'])  # load only the model weights
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]
