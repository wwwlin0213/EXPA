import math
import json
import os
import random
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw


SOURCE_ROOT = Path("/fdata/medical/Br35H-Mask-RCNN")
DEFECT_DIR = SOURCE_ROOT / "TEST"
ANNOTATION_PATH = DEFECT_DIR / "annotations_test.json"

NORMAL_DIR = Path("/fdata/medical/no")

OUTPUT_ROOT = Path("/fdata/medical/Br35H")

NORMAL_COUNT = 100
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def collect_images(folder: Path):
    if not folder.is_dir():
        raise FileNotFoundError(f"目录不存在：{folder}")

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def create_symlink(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() or target.is_symlink():
        target.unlink()

    target.symlink_to(source.resolve())


def normalize_regions(regions):
    """
    VIA 的 regions 可能是 list，也可能是 dict。
    """
    if isinstance(regions, list):
        return regions

    if isinstance(regions, dict):
        return list(regions.values())

    return []

def draw_via_shape(drawer, shape):
    """
    将 VIA 的 shape_attributes 绘制到二值掩膜。

    支持：
    - polygon
    - ellipse，包括旋转椭圆
    - circle
    """
    shape_name = str(shape.get("name", "")).lower()

    # 1. 多边形
    if shape_name == "polygon":
        points_x = shape.get("all_points_x", [])
        points_y = shape.get("all_points_y", [])

        if len(points_x) != len(points_y) or len(points_x) < 3:
            return False

        polygon = [
            (int(round(x)), int(round(y)))
            for x, y in zip(points_x, points_y)
        ]

        drawer.polygon(
            polygon,
            outline=255,
            fill=255
        )
        return True

    # 2. 圆
    if shape_name == "circle":
        required_fields = ("cx", "cy", "r")

        if not all(field in shape for field in required_fields):
            return False

        cx = float(shape["cx"])
        cy = float(shape["cy"])
        radius = float(shape["r"])

        if radius <= 0:
            return False

        drawer.ellipse(
            [
                int(round(cx - radius)),
                int(round(cy - radius)),
                int(round(cx + radius)),
                int(round(cy + radius)),
            ],
            outline=255,
            fill=255
        )
        return True

    # 3. 椭圆，包括旋转椭圆
    if shape_name == "ellipse":
        required_fields = ("cx", "cy", "rx", "ry")

        if not all(field in shape for field in required_fields):
            return False

        cx = float(shape["cx"])
        cy = float(shape["cy"])
        rx = float(shape["rx"])
        ry = float(shape["ry"])

        # VIA 中 theta 使用弧度
        theta = float(shape.get("theta", 0.0))

        if rx <= 0 or ry <= 0:
            return False

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        polygon = []

        # 用 360 个点离散旋转椭圆
        for degree in range(360):
            angle = math.radians(degree)

            local_x = rx * math.cos(angle)
            local_y = ry * math.sin(angle)

            rotated_x = (
                cx
                + local_x * cos_theta
                - local_y * sin_theta
            )

            rotated_y = (
                cy
                + local_x * sin_theta
                + local_y * cos_theta
            )

            polygon.append(
                (
                    int(round(rotated_x)),
                    int(round(rotated_y))
                )
            )

        drawer.polygon(
            polygon,
            outline=255,
            fill=255
        )
        return True

    return False

def build_defect_samples():
    with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    defect_output = OUTPUT_ROOT / "brain" / "test" / "defect"
    mask_output = OUTPUT_ROOT / "brain" / "ground_truth" / "defect"

    defect_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)

    samples = []
    missing_images = []
    empty_masks = []
    invalid_regions = []
    region_count_distribution = Counter()

    for record_key, record in annotations.items():
        filename = Path(record["filename"]).name
        source_image = DEFECT_DIR / filename

        if not source_image.is_file():
            missing_images.append(str(source_image))
            continue

        with Image.open(source_image) as image:
            width, height = image.size

        mask = Image.new("L", (width, height), 0)
        drawer = ImageDraw.Draw(mask)

        regions = normalize_regions(record.get("regions", []))
        valid_region_count = 0

        for region_index, region in enumerate(regions):
            shape = region.get("shape_attributes", {})
            shape_name = shape.get("name", "unknown")

            success = draw_via_shape(drawer, shape)

            if success:
                valid_region_count += 1
            else:
                invalid_regions.append(
                    {
                        "filename": filename,
                        "region_index": region_index,
                        "shape_name": shape_name,
                        "shape_attributes": shape,
                    }
                )

        region_count_distribution[valid_region_count] += 1

        mask_array = np.asarray(mask)

        if not np.any(mask_array > 0):
            empty_masks.append(filename)

        target_image = defect_output / filename
        target_mask = mask_output / f"{Path(filename).stem}.png"

        create_symlink(source_image, target_image)
        mask.save(target_mask)

        samples.append({
            "img_path": target_image.relative_to(
                OUTPUT_ROOT
            ).as_posix(),
            "mask_path": target_mask.relative_to(
                OUTPUT_ROOT
            ).as_posix(),
            "cls_name": "brain",
            "specie_name": "defect",
            "anomaly": 1
        })

    if missing_images:
        raise RuntimeError(
            "部分标注图像不存在：\n" +
            "\n".join(missing_images[:20])
        )

    if empty_masks:
        raise RuntimeError(
            "以下异常图像生成了空掩膜：\n" +
            "\n".join(empty_masks[:20])
        )

    if invalid_regions:
        print("\n警告：发现无法处理的区域：")
        for item in invalid_regions[:20]:
            print(item)

    print("异常图像数:", len(samples))
    print("每张图有效区域数:", region_count_distribution)

    return samples


def build_good_samples():
    normal_images = collect_images(NORMAL_DIR)

    if len(normal_images) < NORMAL_COUNT:
        raise RuntimeError(
            f"正常图像不足：需要 {NORMAL_COUNT} 张，"
            f"实际只有 {len(normal_images)} 张"
        )

    # 固定随机种子，确保每次生成的测试集一致
    rng = random.Random(RANDOM_SEED)
    selected = rng.sample(normal_images, NORMAL_COUNT)
    selected = sorted(selected, key=lambda p: p.name)

    good_output = OUTPUT_ROOT / "brain" / "test" / "good"
    good_output.mkdir(parents=True, exist_ok=True)

    samples = []

    for index, source_image in enumerate(selected):
        # 加序号，避免文件名重复或不同扩展名同名
        target_name = (
            f"normal_{index:03d}_{source_image.name}"
        )
        target_image = good_output / target_name

        create_symlink(source_image, target_image)

        samples.append({
            "img_path": target_image.relative_to(
                OUTPUT_ROOT
            ).as_posix(),
            "mask_path": "",
            "cls_name": "brain",
            "specie_name": "good",
            "anomaly": 0
        })

    selected_list_path = OUTPUT_ROOT / "selected_normal_images.txt"
    with open(selected_list_path, "w", encoding="utf-8") as f:
        for source_image in selected:
            f.write(str(source_image.resolve()) + "\n")

    print("正常图像数:", len(samples))
    print("随机种子:", RANDOM_SEED)
    print("正常图清单:", selected_list_path)

    return samples


def main():
    if not ANNOTATION_PATH.is_file():
        raise FileNotFoundError(
            f"标注文件不存在：{ANNOTATION_PATH}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    defect_samples = build_defect_samples()
    good_samples = build_good_samples()

    # 正常图放在前面，和常见 VisualAD meta.json 一致
    all_samples = good_samples + defect_samples

    meta = {
        "train": {
            "brain": []
        },
        "test": {
            "brain": all_samples
        }
    }

    meta_path = OUTPUT_ROOT / "meta.json"

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            meta,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("输出目录:", OUTPUT_ROOT)
    print("meta.json:", meta_path)
    print("正常样本:", len(good_samples))
    print("异常样本:", len(defect_samples))
    print("总样本:", len(all_samples))
    print("=" * 70)


if __name__ == "__main__":
    main()