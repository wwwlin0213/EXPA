import os
import json
import argparse

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def is_image(name):
    return name.lower().endswith(IMG_EXTS)


def find_mask_path(root, cls_name, defect_type, img_name):
    """
    Brain_AD 使用 MVTec 风格：
    brain/test/defect/xxx.png
    brain/ground_truth/defect/xxx_mask.png
    """
    stem, _ = os.path.splitext(img_name)
    gt_dir = os.path.join(root, cls_name, 'ground_truth', defect_type)

    if not os.path.isdir(gt_dir):
        return '.'

    candidate_names = []

    for ext in IMG_EXTS:
        candidate_names.append(stem + '_mask' + ext)
        candidate_names.append(stem + ext)

    for cand in candidate_names:
        abs_mask_path = os.path.join(gt_dir, cand)
        if os.path.exists(abs_mask_path):
            return os.path.relpath(abs_mask_path, root)

    # 兜底：有些数据 mask 名字可能和原图 stem 有额外后缀
    for mask_name in sorted(os.listdir(gt_dir)):
        if not is_image(mask_name):
            continue
        mask_stem, _ = os.path.splitext(mask_name)
        if stem in mask_stem:
            abs_mask_path = os.path.join(gt_dir, mask_name)
            return os.path.relpath(abs_mask_path, root)

    print(f'[WARN] mask not found for image: {cls_name}/test/{defect_type}/{img_name}')
    return '.'


def collect_train(root):
    items = []
    cls_name = 'brain'
    train_good_dir = os.path.join(root, cls_name, 'train', 'good')

    if not os.path.isdir(train_good_dir):
        print(f'[WARN] Missing train good dir: {train_good_dir}')
        return items

    for dirpath, _, filenames in os.walk(train_good_dir):
        for img_name in sorted(filenames):
            if not is_image(img_name):
                continue

            abs_img_path = os.path.join(dirpath, img_name)
            rel_img_path = os.path.relpath(abs_img_path, root)

            items.append({
                'img_path': rel_img_path,
                'mask_path': '.',
                'cls_name': cls_name,
                'specie_name': 'good',
                'anomaly': 0,
            })

    return items


def collect_test(root):
    items = []
    cls_name = 'brain'
    test_dir = os.path.join(root, cls_name, 'test')

    if not os.path.isdir(test_dir):
        print(f'[WARN] Missing test dir: {test_dir}')
        return items

    for defect_type in sorted(os.listdir(test_dir)):
        defect_dir = os.path.join(test_dir, defect_type)

        if not os.path.isdir(defect_dir):
            continue

        anomaly = 0 if defect_type == 'good' else 1

        for dirpath, _, filenames in os.walk(defect_dir):
            for img_name in sorted(filenames):
                if not is_image(img_name):
                    continue

                abs_img_path = os.path.join(dirpath, img_name)
                rel_img_path = os.path.relpath(abs_img_path, root)

                if anomaly == 0:
                    mask_path = '.'
                else:
                    mask_path = find_mask_path(root, cls_name, defect_type, img_name)

                items.append({
                    'img_path': rel_img_path,
                    'mask_path': mask_path,
                    'cls_name': cls_name,
                    'specie_name': defect_type,
                    'anomaly': anomaly,
                })

    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        type=str,
        default='/fdata/medical/Brain_AD'
    )
    args = parser.parse_args()

    root = args.root
    print(f'[INFO] Using dataset root: {root}')

    train_items = collect_train(root)
    test_items = collect_test(root)

    info = {
        'train': {'brain': train_items},
        'test': {'brain': test_items},
    }

    meta_path = os.path.join(root, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(info, f, indent=4)

    train_normal = sum(x['anomaly'] == 0 for x in train_items)
    train_anomaly = sum(x['anomaly'] == 1 for x in train_items)
    test_normal = sum(x['anomaly'] == 0 for x in test_items)
    test_anomaly = sum(x['anomaly'] == 1 for x in test_items)
    test_masked = sum(x['anomaly'] == 1 and x['mask_path'] != '.' for x in test_items)

    print(f'[OK] meta.json saved to: {meta_path}')
    print(f'train samples: {len(train_items)} | normal: {train_normal} | anomaly: {train_anomaly}')
    print(f'test samples:  {len(test_items)} | normal: {test_normal} | anomaly: {test_anomaly}')
    print(f'test anomaly with mask: {test_masked} / {test_anomaly}')

    if test_normal == 0 or test_anomaly == 0:
        print('[WARN] Test set must contain both good and defect samples for I-AUC.')
    if test_anomaly > 0 and test_masked == 0:
        print('[WARN] No anomaly mask found. Pixel-level metrics will be invalid.')


if __name__ == '__main__':
    main()