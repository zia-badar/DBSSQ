import torch.nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.image_net import image_net


# print(f"accuracy: {evaluate_imagenet(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).cuda())}")
def evaluate_imagenet(model: torch.nn.Module, show_tqdm=False):
    dataset = image_net('val')
    dataloader = DataLoader(dataset=dataset, batch_size=256, num_workers=20, shuffle=True)
    model.eval()

    correct = 0
    total = 0
    ce_loss = CrossEntropyLoss()
    val_loss = 0
    with torch.no_grad():
        for xs, ys in tqdm(dataloader, disable=(not show_tqdm)):
            xs = xs.cuda()
            ys = ys.cuda()
            logit = model(xs)
            val_loss += ce_loss(logit, ys.cuda()).item()
            y_pred = torch.argmax(logit, dim=1)
            correct += torch.sum(y_pred == torch.argmax(ys, dim=1)).item()
            total += len(ys)

    return val_loss, (correct / total)