import matplotlib.pyplot as plt
from scripts.mnist_loader import CustomMNISTDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from scripts.utils import setup_all_seed, yaml_load



if __name__ == "__main__":
    # load config
    CONFIG_PATH = 'config.yaml'
    config = yaml_load(CONFIG_PATH)
    config_dataset = config['dataset']
    
    # set random seed for reproducibility
    setup_all_seed()
    print("seed : 0")

    train_dataset = CustomMNISTDataset(root=config_dataset['root_dir'],
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=config_dataset['download'], 
                                    size=config_dataset['image_size'])
    test_dataset = CustomMNISTDataset(root=config_dataset['root_dir'],
                                    train=False,
                                    transform=transforms.ToTensor(),
                                    download=config_dataset['download'], 
                                    size=config_dataset['image_size'])

    train_loader = DataLoader(train_dataset,
                            batch_size=config_dataset['batch_size'],
                            shuffle=True,
                            num_workers=config_dataset['num_workers'])
    test_loader = DataLoader(test_dataset,
                            batch_size=config_dataset['batch_size'],
                            shuffle=False,
                            num_workers=config_dataset['num_workers'])

    # show sample images with dataloader
    for i, (sample_images, sample_labels) in enumerate(train_loader):
        print(f"sample {i} images shape: {sample_images.shape}")
        print(f"sample {i} labels shape: {sample_labels.shape}")
        plt.imshow(sample_images[0].view(-1, config_dataset['image_size']), cmap='gray')
        plt.title(f"label : {sample_labels[0]}")
        plt.show()
        break

    # show sample images directly from dataset
    for i in range(10):
        image, label = train_dataset[i]
        plt.imshow(image.view(-1, config_dataset['image_size']), cmap='gray')
        plt.title(f"label : {label}")
        plt.show()
        break