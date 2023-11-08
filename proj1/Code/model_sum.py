import argparse
import torch
import torchvision.models as models
from torchsummary import summary

def get_model(model_name):
    try:
        # Try to dynamically fetch the model from torchvision.models
        model = getattr(models, model_name)(pretrained=True)
    except AttributeError:
        raise ValueError(f"Invalid model name: {model_name}")

    return model

def main():
    parser = argparse.ArgumentParser(description='Print model summary for a specified model.')
    parser.add_argument('model_name', type=str, help='Name of the model (e.g., vgg11, resnet18, efficientnet_b0)')
    args = parser.parse_args()

    model_name = args.model_name
    model = get_model(model_name)

    print(f"--------------{model_name}--------------:")
    print(model)
    print(summary(model, (3, 224, 224)))

if __name__ == "__main__":
    main()
