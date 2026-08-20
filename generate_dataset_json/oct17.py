import os
import json
import argparse

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
NORMAL_CLASSES = {'NORMAL'}
ANOMALY_CLASSES = {'CNV', 'DME', 'DRUSEN'}


def find_real_root(root):
    candidates = [
        root,
        os.path.join(root, 'OCT2017'),
        os.path.join(root, 'Retina_OCT2017_AD'),
    ]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, 'test')):
            return cand
    return root


def collect_phase(root, phase):
    phase_dir = os.path.join(root, phase)
    items = []

    if not os.path.isdir(phase_dir):
        return items

    for specie_name in sorted(os.listdir(phase_dir)):
        specie_dir = os.path.join(phase_dir, specie_name)
        if not os.path.isdir(specie_dir):
            continue

        upper_name = specie_name.upper()
        if upper_name in NORMAL_CLASSES:
            anomaly = 0
        elif upper_name in ANOMALY_CLASSES:
            anomaly = 1
        else:
            print(f'[WARN] Unknown OCT class skipped: {specie_name}')
            continue

        for img_name in sorted(os.listdir(specie_dir)):
            if not img_name.lower().endswith(IMG_EXTS):
                continue

            rel_img_path = os.path.relpath(
                os.path.join(specie_dir, img_name),
                root
            )

            items.append({
                'img_path': rel_img_path,
                'mask_path': '.',          # OCT2017 has no pixel mask; use dummy zero mask
                'cls_name': 'retina',
                'specie_name': specie_name,
                'anomaly': anomaly,
            })

    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        type=str,
        default='/fdata/OCT2017'
    )
    args = parser.parse_args()

    root = find_real_root(args.root)
    print(f'[INFO] Using dataset root: {root}')

    info = {
        'train': {'retina': collect_phase(root, 'train')},
        'test': {'retina': collect_phase(root, 'test')},
    }

    meta_path = os.path.join(root, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(info, f, indent=4)

    print(f'[OK] meta.json saved to: {meta_path}')
    print(f"train samples: {len(info['train']['retina'])}")
    print(f"test samples:  {len(info['test']['retina'])}")


if __name__ == '__main__':
    main()