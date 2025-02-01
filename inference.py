import os
from pathlib import Path
import argparse

import torch
import numpy as np
import nibabel as nib

import monai.transforms as tr
from monai.data import decollate_batch, Dataset, DataLoader
from monai.inferers import SliceInferer
from monai.networks.nets import AttentionUnet
from monai.transforms import SaveImaged, MapTransform
from tqdm import tqdm

# Constants
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_PATH = "./AttUNet.pth"
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 2
DEFAULT_IMG_SIZE = 256
DEFAULT_SUFFIX = "predicted_mask"


class SliceWiseNormalizeIntensityd(MapTransform):
    """
    Custom MONAI transform to normalize intensity slice-wise.
    """
    def __init__(self, keys, subtrahend=0.0, divisor=None, nonzero=True):
        super().__init__(keys)
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            for i in range(image.shape[-1]):
                slice_ = image[..., i]
                if self.nonzero:
                    mask = slice_ > 0
                    if np.any(mask):
                        slice_[mask] = slice_[mask] - (slice_[mask].mean() if self.subtrahend is None else self.subtrahend)
                        slice_[mask] /= (slice_[mask].std() if self.divisor is None else self.divisor)
                else:
                    slice_ -= slice_.mean() if self.subtrahend is None else self.subtrahend
                    slice_ /= slice_.std() if self.divisor is None else self.divisor
                image[..., i] = slice_
            d[key] = image
        return d


class FetalTestDataLoader:
    def __init__(self, test_data_path, img_size=DEFAULT_IMG_SIZE):
        self.test_data_path = test_data_path
        self.img_size = img_size

    def get_transforms(self):
        """Define test data transformations."""
        return [
            tr.LoadImaged(keys=["image"]),
            tr.EnsureChannelFirstd(keys=["image"]),
            tr.Spacingd(keys="image", pixdim=(1.0, 1.0, -1.0), mode="bilinear", padding_mode="zeros"),
            SliceWiseNormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=None, nonzero=True),
        ]

    def load_data(self):
        """Create and load the test dataset."""
        test_transforms_list = self.get_transforms()
        test_files = [{"image": self.test_data_path}]
        test_dataset = Dataset(data=test_files, transform=tr.Compose(test_transforms_list))
        test_dataloader = DataLoader(
            test_dataset, batch_size=DEFAULT_BATCH_SIZE, num_workers=DEFAULT_NUM_WORKERS
        )
        return test_dataloader, test_transforms_list


def load_model(model_path, device):
    """Load the pretrained model."""
    model = AttentionUnet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        kernel_size=3,
        up_kernel_size=3,
        dropout=0.15,
    )
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


def perform_inference(model, dataloader, device, test_transforms_list, output_path, suffix):
    """Run the inference and save the predictions."""
    inferer = SliceInferer(
        roi_size=(DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE),
        spatial_dim=2,
        sw_batch_size=16,
        overlap=0.50,
        progress=False
    )

    post_transforms = tr.Compose([
        tr.Invertd(
            keys="pred",
            transform=tr.Compose(test_transforms_list),
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        tr.Activationsd(keys="pred", softmax=True),
        tr.AsDiscreted(keys="pred", argmax=True),
        SaveImaged(
            keys="pred", meta_keys="pred_meta_dict", output_dir=str(Path(output_path)),
            print_log=False, separate_folder=False, output_postfix=suffix, resample=False
        ),
    ])

    with torch.no_grad():
        for test_data in tqdm(dataloader, desc="Processing"):
            test_inputs = test_data["image"].to(device)
            test_data["pred"] = inferer(test_inputs, model)
            test_data = [post_transforms(i) for i in decollate_batch(test_data)]

    print("Process completed")


def main(args):
    os.makedirs(str(Path(args.output_path).parent), exist_ok=True)

    # Load model
    model = load_model(args.saved_model_path, args.device)

    # Load data
    data_loader = FetalTestDataLoader(args.input_path)
    test_dataloader, test_transforms_list = data_loader.load_data()

    # Perform inference
    perform_inference(model, test_dataloader, args.device, test_transforms_list, args.output_path, args.suffix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, help='Device to use for computation')
    parser.add_argument('--saved_model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to the saved model')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output mask')
    parser.add_argument('--suffix', type=str, default=DEFAULT_SUFFIX, help='Suffix for output mask')

    args = parser.parse_args()
    main(args)
