import torch
import torch.backends.cudnn as cudnn
import torchvision

import argparse
import os

from model import Net

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir", default='data/Market-1501/pytorch/', type=str)
parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--gpu-id", default=0, type=int)
args = parser.parse_args()

# device
device = "cuda:{}".format(
    args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loader
root = args.data_dir
query_dir = os.path.join(root, "query")
gallery_dir = os.path.join(root, "gallery")
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
query_datasets = torchvision.datasets.ImageFolder(query_dir, transform=transform)
queryloader = torch.utils.data.DataLoader(
    query_datasets,
    batch_size=64, shuffle=False
)

gallery_datasets = torchvision.datasets.ImageFolder(gallery_dir, transform=transform)
galleryloader = torch.utils.data.DataLoader(
    gallery_datasets,
    batch_size=64, shuffle=False
)

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

query_path = query_datasets.imgs
gallery_path = gallery_datasets.imgs
query_cam, query_label = get_id(query_path)
gallery_cam, gallery_label = get_id(gallery_path)
print(gallery_cam)

# net definition
net = Net(reid=True)
assert os.path.isfile(
    "./checkpoint/ckpt-Market1501.t7"), "Error: no checkpoint file found!"
print('Loading from checkpoint/ckpt-Market1501.t7')
checkpoint = torch.load("./checkpoint/ckpt-Market1501.t7", map_location=device)
net_dict = checkpoint['net_dict']
net.load_state_dict(net_dict, strict=False)
net.eval()
net.to(device)

# compute features
query_features = torch.tensor([]).float()
query_labels = torch.tensor([]).long()
gallery_features = torch.tensor([]).float()
gallery_labels = torch.tensor([]).long()

with torch.no_grad():
    for idx, (inputs, labels) in enumerate(queryloader):
        inputs = inputs.to(device)
        features = net(inputs).cpu()
        query_features = torch.cat((query_features, features), dim=0)
        query_labels = torch.cat((query_labels, labels))

    for idx, (inputs, labels) in enumerate(galleryloader):
        inputs = inputs.to(device)
        features = net(inputs).cpu()
        gallery_features = torch.cat((gallery_features, features), dim=0)
        gallery_labels = torch.cat((gallery_labels, labels))

gallery_labels -= 2

# save features
features = {
    "qf": query_features,
    "ql": query_labels,
    "qc" : query_cam,
    "gf": gallery_features,
    "gl": gallery_labels,
    "gc": gallery_cam
}
torch.save(features, "features.pth")