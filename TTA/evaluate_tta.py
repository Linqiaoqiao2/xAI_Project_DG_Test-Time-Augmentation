import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from domainbed import datasets
from domainbed.hparams_registry import default_hparams
from domainbed.algorithms import get_algorithm_class
from domainbed.transforms.transforms import apply_tta

def evaluate(model, loader, device, tta_mode):
    model.eval()
    correct = 0
    total = 0
    to_pil = ToPILImage()

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)

            # Apply TTA
            images_tta = []
            for img in images:
                pil_img = to_pil(img.cpu())
                augmented = apply_tta(pil_img, tta_mode)
                images_tta.append(torch.stack(augmented))

            images_tta = torch.stack(images_tta)  # (B, N, C, H, W)
            B, N, C, H, W = images_tta.size()
            images_tta = images_tta.view(-1, C, H, W).to(device)

            logits = model.predict(images_tta)  # Use .predict instead of .forward()
            logits = logits.view(B, N, -1).mean(dim=1)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--test_env', type=int, required=True)
    parser.add_argument('--tta_mode', type=str, default="flip")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\U0001F4E6 Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Recreate the algorithm
    algorithm_class = get_algorithm_class("ERM")
    input_shape = checkpoint["model_input_shape"]
    num_classes = checkpoint["model_num_classes"]
    num_domains = checkpoint["model_num_domains"]
    hparams = checkpoint["model_hparams"]

    model = algorithm_class(input_shape, num_classes, num_domains, hparams)
    model.load_state_dict(checkpoint["model_dict"])
    model.to(device)

    print(f"\U0001F4C1 Loading test environment {args.test_env} from {args.dataset}")
    dataset_class = vars(datasets)[args.dataset]
    hparams["batch_size"] = 64  # Optional override
    dataset = dataset_class(args.data_dir, test_envs=[args.test_env], hparams=hparams)
    test_loader = DataLoader(dataset[0], batch_size=hparams["batch_size"], shuffle=False, num_workers=2)

    acc = evaluate(model, test_loader, device, args.tta_mode)
    print(f"\u2705 TTA accuracy (mode={args.tta_mode}): {acc:.4f}")

if __name__ == "__main__":
    main()