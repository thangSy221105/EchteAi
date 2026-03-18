# https://cocodataset.org/#download
# Dataset 2017

import os
import logging
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torchvision.transforms.functional as TF

import EchteAI.data.dataloaders as dl
from EchteAI.models.vision.models.fasterrcnn_split import ONNXFasterRCNNWrapper, split_save_frcnn
from EchteAI.models.vision.models.fasterrcnn_utils import run_predictions_fasterrcnn, setup_fasterrcnn
from EchteAI.models.vision.models.onnx_frcnn import onnx_conv_outputs_from_batch, quantize_feature_extractor
from EchteAI.models.vision.visualization import absolute_differences, fit_and_plot_distribution, percentage_differences, visualize_cnn_outputs
import EchteAI.models.vision.models.yolo_utils as yolo
import EchteAI.models.vision.models.fasterrcnn_utils as frcnn_utils
from ultralytics import YOLO
from torchvision import models

torch.manual_seed(42)
device = "cuda"
print(f"Using device: {device}")
NUM_IMAGES = 100#00

def prepare_image(img_tensor, target_size=(224,224)):
    img_resized = frcnn_utils.resize_and_pad(img_tensor, target_size)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)
    return (img_resized - mean) / std

TARGET = 640
PAD = 10


def resize_keep_aspect(img):
    w, h = img.size
    scale = min(TARGET / w, TARGET / h)
    return img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)


def pad_bottom_right(img):
    w, h = img.size
    new_img = Image.new("RGB", (TARGET, TARGET), (0, 0, 0))
    new_img.paste(img, (0, 0))
    return new_img, TARGET - h


def process_pair(img1, img2):
    img1 = resize_keep_aspect(img1)
    img2 = resize_keep_aspect(img2)

    img1, pad1 = pad_bottom_right(img1)
    img2, pad2 = pad_bottom_right(img2)

    def remove_pad(img, pad):
        if pad == 0:
            return img
        return img.crop((0, 0, TARGET, TARGET - pad))

    img1 = remove_pad(img1, pad1)
    img2 = remove_pad(img2, pad2)

    return TF.to_tensor(img1), TF.to_tensor(img2)

def load(path):
    if not os.path.exists(path):
        return Image.new("RGB", (TARGET, TARGET))
    return Image.open(path).convert("RGB")


def build_grid_from_pairs(pairs):
    rows = []

    for t1, t2 in pairs:
        h1 = t1.shape[1]
        h2 = t2.shape[1]

        h = min(h1, h2)

        t1 = t1[:, :h, :]
        t2 = t2[:, :h, :]

        pad_col = torch.zeros(3, h, PAD)

        row = torch.cat([t1, pad_col, t2], dim=2)
        rows.append(row)

    final = []
    for i, row in enumerate(rows):
        final.append(row)

        if i != len(rows) - 1:
            pad_row = torch.zeros(3, PAD, row.shape[2])
            final.append(pad_row)

    grid = torch.cat(final, dim=1)
    return grid


def build_image(i, mode):
    base = "outputs"

    pairs_paths = [
        (
            os.path.join(base, "frcnn_int8_preds", f"batch{i}_img0.png"),
            os.path.join(base, "yolo", mode, f"img{i}.png"),
        ),
        (
            os.path.join(base, "efficientnet", mode, f"img{i}.png"),
            os.path.join(base, "densenet", mode, f"img{i}.png"),
        ),
        (
            os.path.join(base, "frcnn_fp32", mode, f"img{i}.png"),
            os.path.join(base, "frcnn_int8", mode, f"img{i}.png"),
        ),
        (
            os.path.join(base, "frcnn_diff", mode, f"img{i}.png"),
            os.path.join(base, "frcnn_diff_percent", mode, f"img{i}.png"),
        ),
        (
            os.path.join(base, "distribution_diffs", mode, f"img{i}_distribution_distribution.png"),
            os.path.join(base, "distribution_diffs", mode, f"img{i}_distribution_per_layer_stats.png"),
        ),
    ]

    tensor_pairs = []

    for p1, p2 in pairs_paths:
        img1 = load(p1)
        img2 = load(p2)

        t1, t2 = process_pair(img1, img2)
        tensor_pairs.append((t1, t2))

    return build_grid_from_pairs(tensor_pairs)


def process(mode):
    base = "outputs"

    out_dir = os.path.join(base, "combined", mode)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(3):
        imgs = []

        imgs.append(load(os.path.join(base, "frcnn_int8_preds", f"batch{i}_img0.png")))
        imgs.append(load(os.path.join(base, "yolo", mode, f"img{i}.png")))

        imgs.append(load(os.path.join(base, "efficientnet", mode, f"img{i}.png")))
        imgs.append(load(os.path.join(base, "densenet", mode, f"img{i}.png")))

        imgs.append(load(os.path.join(base, "frcnn_fp32", mode, f"img{i}.png")))
        imgs.append(load(os.path.join(base, "frcnn_int8", mode, f"img{i}.png")))

        imgs.append(load(os.path.join(base, "frcnn_diff", mode, f"img{i}.png")))
        imgs.append(load(os.path.join(base, "frcnn_diff_percent", mode, f"img{i}.png")))

        imgs.append(load(os.path.join(
            base, "distribution_diffs", mode, f"img{i}_distribution_distribution.png"
        )))
        imgs.append(load(os.path.join(
            base, "distribution_diffs", mode, f"img{i}_distribution_per_layer_stats.png"
        )))

        grid = build_grid_from_pairs(imgs)

        out_path = os.path.join(out_dir, f"img{i}.png")
        vutils.save_image(grid, out_path)

        print(f"Saved: {out_path}")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cwd = os.path.dirname(__file__)
    
    # --- Paths ---
    image_dir = os.path.join(cwd, "downloads", "val2017")
    annotation_path = os.path.join(cwd, "downloads", "annotations", "instances_val2017.json")
    model_dir = os.path.join(cwd, "outputs", "models")

    os.makedirs(model_dir, exist_ok=True)

    # --- Dataset ---
    dataset = dl.CocoDetectionDataset(
        image_dir=image_dir,
        annotation_path=annotation_path,
        transforms=T.Compose([T.ToTensor()])
    )

    total_len = len(dataset)
    calib_len = min(int(0.512 * total_len), 512)

    calib_dataset, _ = random_split(dataset, [calib_len, total_len - calib_len])

    calib_loader = DataLoader(
        calib_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda b: tuple(zip(*b))
    )

    sample_imgs = [dataset[i][0].to(device) for i in range(NUM_IMAGES)]
    from torch.utils.data import Subset
    subset_dataset = Subset(dataset, indices=list(range(NUM_IMAGES)))
    subset_dataset.class_to_idx = dataset.class_to_idx
    subset_dataset.idx_to_class = dataset.idx_to_class
    subset_loader = DataLoader(
        subset_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: tuple(zip(*b))
    )


    viz_configs = [
        ("all", {}),
        ("layer1", {"layer": 1}),
        ("first4", {"num_layers": 4})
    ]

    if False:
        # ------------------------
        # Faster R-CNN
        # ------------------------
        if True:
            logging.info("Setting up FP32 Faster R-CNN...")
            model_fp32 = setup_fasterrcnn(backbone="resnet50")
            model_fp32.to(device).eval()

            fe_onnx_path = os.path.join(model_dir, "feature_extractor.onnx")
            dh_onnx_path = os.path.join(model_dir, "detector_head.onnx")
            quantized_fe_path = os.path.join(model_dir, "feature_extractor_quant.onnx")

            if not os.path.exists(fe_onnx_path) or not os.path.exists(dh_onnx_path):

                images, _ = next(iter(calib_loader))
                calib_images = [img.to(device) for img in images]

                split_save_frcnn(
                    model_fp32,
                    calib_images,
                    device=device,
                    model_dir=model_dir
                )

            if not os.path.exists(quantized_fe_path):

                quantize_feature_extractor(
                    fe_onnx_path,
                    calib_loader,
                    model_fp32.transform,
                    quantized_fe_path,
                    num_batches=16
                )

            onnx_model_int8 = ONNXFasterRCNNWrapper(
                fe_onnx_path=quantized_fe_path,
                dh_onnx_path=dh_onnx_path,
                transform=model_fp32.transform,
                device=device
            )

            output_folder_int8 = os.path.join(cwd, "outputs", "frcnn_int8_preds")
            os.makedirs(output_folder_int8, exist_ok=True)

            run_predictions_fasterrcnn(
                model=onnx_model_int8,
                data_loader=subset_loader,
                device=device,
                dataset=subset_dataset,
                output_folder=output_folder_int8,
                evaluate=False,
                score_threshold=0.85
            )

            for name, params in viz_configs:

                fp32_dir = os.path.join(cwd, "outputs", "frcnn_fp32", name)
                int8_dir = os.path.join(cwd, "outputs", "frcnn_int8", name)
                diff_dir = os.path.join(cwd, "outputs", "frcnn_diff", name)
                diff_dir_pc = os.path.join(cwd, "outputs", "frcnn_diff_percent", name)
                diff_dist_dir = os.path.join(cwd, "outputs", "distribution_diffs", name)

                os.makedirs(fp32_dir, exist_ok=True)
                os.makedirs(int8_dir, exist_ok=True)
                os.makedirs(diff_dir, exist_ok=True)
                os.makedirs(diff_dir_pc, exist_ok=True)
                os.makedirs(diff_dist_dir, exist_ok=True)

                for i, img in enumerate(sample_imgs):

                    img_batch = img.unsqueeze(0)

                    # FP32 feature map
                    fp32_feats = onnx_conv_outputs_from_batch(
                        fe_onnx_path,
                        img_batch,
                        transform=model_fp32.transform,
                        device=device,
                        **params
                    )

                    fp32_feats.pop("logits", None)
                    fp32_feats.pop("max", None)

                    visualize_cnn_outputs(
                        fp32_feats,
                        filename=os.path.join(fp32_dir, f"img{i}")
                    )

                    # INT8 feature map
                    int8_feats = onnx_conv_outputs_from_batch(
                        quantized_fe_path,
                        img_batch,
                        transform=model_fp32.transform,
                        device=device,
                        **params
                    )

                    int8_feats.pop("logits", None)
                    int8_feats.pop("max", None)

                    visualize_cnn_outputs(
                        int8_feats,
                        filename=os.path.join(int8_dir, f"img{i}")
                    )

                    # -----------------------
                    # DIFFERENCES
                    # -----------------------
                    diffs = absolute_differences(fp32_feats, int8_feats)
                    visualize_cnn_outputs(
                        diffs,
                        filename=os.path.join(diff_dir, f"img{i}")
                    )
                    fit_and_plot_distribution(
                        outputs1=fp32_feats,
                        diffs=diffs,
                        output_folder=diff_dist_dir,
                        filename=os.path.join(diff_dist_dir, f"img{i}_distribution")
                    )
                    diffs = percentage_differences(fp32_feats, int8_feats)
                    visualize_cnn_outputs(
                        diffs,
                        filename=os.path.join(diff_dir_pc, f"img{i}")
                    )

        # ------------------------
        # YOLOv10 ONNX export
        # ------------------------
        if True:
            yolo_onnx_path = "yolov10x.onnx"

            if not os.path.exists(yolo_onnx_path):

                yolo_model = YOLO("yolov10x.pt")
                yolo_model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)

                if os.path.exists("yolov10x.onnx"):
                    os.replace("yolov10x.onnx", yolo_onnx_path)

            for name, params in viz_configs:

                yolo_dir = os.path.join(cwd, "outputs", "yolo", name)
                os.makedirs(yolo_dir, exist_ok=True)

                for i, img in enumerate(sample_imgs):

                    img_tensor = prepare_image(img, (640,640)).unsqueeze(0)

                    feats = onnx_conv_outputs_from_batch(
                        yolo_onnx_path,
                        img_tensor,
                        transform=None,
                        device=device,
                        **params,
                        pattern=r".*conv.*"
                    )

                    feats.pop("logits", None)

                    first_layer_name = list(feats.keys())[0]
                    first_layer_tensor = feats[first_layer_name]

                    print(f"First conv layer name: {first_layer_name}")
                    print(f"First conv layer shape: {first_layer_tensor.shape}")

                    visualize_cnn_outputs(
                        feats,
                        filename=os.path.join(yolo_dir, f"img{i}")
                    )

        # ------------------------
        # EfficientNet ONNX export
        # ------------------------
        if True:
            effnet_onnx_path = os.path.join(model_dir, "efficientnet_b0.onnx")

            if not os.path.exists(effnet_onnx_path):

                effnet_model = models.efficientnet_b0(
                    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
                )

                effnet_model.to(device).eval()

                dummy_input = torch.randn(1,3,224,224,device=device)

                torch.onnx.export(
                    effnet_model,
                    dummy_input,
                    effnet_onnx_path,
                    input_names=["images"],
                    output_names=["logits"],
                    opset_version=17
                )

            for name, params in viz_configs:

                eff_dir = os.path.join(cwd, "outputs", "efficientnet", name)
                os.makedirs(eff_dir, exist_ok=True)

                for i, img in enumerate(sample_imgs):

                    img_tensor = prepare_image(img, (224,224)).unsqueeze(0)

                    feats = onnx_conv_outputs_from_batch(
                        effnet_onnx_path,
                        img_tensor,
                        transform=None,
                        device=device,
                        **params
                    )

                    feats.pop("logits", None)

                    visualize_cnn_outputs(
                        feats,
                        filename=os.path.join(eff_dir, f"img{i}")
                    )

        # ------------------------
        # DenseNet ONNX export
        # ------------------------
        if True:
            densenet_onnx_path = os.path.join(model_dir, "densenet121.onnx")

            if not os.path.exists(densenet_onnx_path):

                densenet_model = models.densenet121(
                    weights=models.DenseNet121_Weights.IMAGENET1K_V1
                )

                densenet_model.to(device).eval()

                dummy_input = torch.randn(1,3,224,224,device=device)

                torch.onnx.export(
                    densenet_model,
                    dummy_input,
                    densenet_onnx_path,
                    input_names=["images"],
                    output_names=["logits"],
                    opset_version=17
                )

            for name, params in viz_configs:

                dn_dir = os.path.join(cwd, "outputs", "densenet", name)
                os.makedirs(dn_dir, exist_ok=True)

                for i, img in enumerate(sample_imgs):

                    img_tensor = prepare_image(img, (224,224)).unsqueeze(0)

                    feats = onnx_conv_outputs_from_batch(
                        densenet_onnx_path,
                        img_tensor,
                        transform=None,
                        device=device,
                        **params
                    )

                    feats.pop("logits", None)

                    visualize_cnn_outputs(
                        feats,
                        filename=os.path.join(dn_dir, f"img{i}")
                    )

    # ------------------------
    # Combined Grids
    # ------------------------
    for mode in ["all", "first4", "layer1"]:
        out_dir = os.path.join("outputs", "combined", mode)
        os.makedirs(out_dir, exist_ok=True)

        for i in range(NUM_IMAGES):
            grid = build_image(i, mode)
            out_path = os.path.join(out_dir, f"img{i}.png")
            vutils.save_image(grid, out_path)
            print("Saved:", out_path)

if __name__ == "__main__":
    main()