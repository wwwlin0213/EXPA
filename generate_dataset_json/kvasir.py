import os
import json
import argparse

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def is_image(name):
    return name.lower().endswith(IMG_EXTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        type=str,
        default='/fdata/medical/Kvasir'
    )
    args = parser.parse_args()

    root = args.root

    img_dir = os.path.join(root, 'images')
    mask_dir = os.path.join(root, 'masks')

    cls_name = 'kvasir'
    specie_name = 'defect'

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f'Image dir not found: {img_dir}')

    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f'Mask dir not found: {mask_dir}')

    test_items = []
    missing_masks = []

    for img_name in sorted(os.listdir(img_dir)):
        if not is_image(img_name):
            continue

        abs_img_path = os.path.join(img_dir, img_name)
        abs_mask_path = os.path.join(mask_dir, img_name)

        if not os.path.exists(abs_mask_path):
            stem = os.path.splitext(img_name)[0]
            found = None

            for ext in IMG_EXTS:
                cand = os.path.join(mask_dir, stem + ext)
                if os.path.exists(cand):
                    found = cand
                    break

            if found is None:
                missing_masks.append(img_name)
                continue

            abs_mask_path = found

        rel_img_path = os.path.relpath(abs_img_path, root)
        rel_mask_path = os.path.relpath(abs_mask_path, root)

        test_items.append({
            'img_path': rel_img_path,
            'mask_path': rel_mask_path,
            'cls_name': cls_name,
            'specie_name': specie_name,
            'anomaly': 1,
        })

    info = {
        'train': {cls_name: []},
        'test': {cls_name: test_items},
    }

    meta_path = os.path.join(root, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(info, f, indent=4)

    print(f'[OK] meta.json saved to: {meta_path}')
    print(f'test samples: {len(test_items)}')
    print(f'missing masks: {len(missing_masks)}')

    if missing_masks:
        print('[WARN] First missing masks:')
        for name in missing_masks[:10]:
            print(name)

    if len(test_items) == 0:
        print('[WARN] No valid image-mask pairs found.')


if __name__ == '__main__':
    main()