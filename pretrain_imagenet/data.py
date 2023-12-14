import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader


def make_data(batch_size, n_workers):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def apply_transform_train(examples):
        examples['pixel_values'] = [transform_train(image.convert('RGB')) for image in examples['image']]
        del examples['image']
        return examples
    
    def apply_transform_eval(examples):
        examples['pixel_values'] = [transform_eval(image.convert('RGB')) for image in examples['image']]
        del examples['image']
        return examples

    def collate(examples):
        x = []
        y = []
        for example in examples:
            x.append(example['pixel_values'])
            y.append(example['label'])
        x = torch.stack(x)
        y = torch.tensor(y)
        return x, y

    dataset_train = load_dataset('imagenet-1k', split='train')
    dataset_val = load_dataset('imagenet-1k', split='validation')
    dataset_train.set_format('torch')
    dataset_val.set_format('torch')
    dataset_train.set_transform(apply_transform_train)
    dataset_val.set_transform(apply_transform_eval)
    data_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers, collate_fn=collate,
        pin_memory=True, persistent_workers=True)
    data_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=n_workers, collate_fn=collate, pin_memory=True,
        persistent_workers=True)
    return data_train, data_val