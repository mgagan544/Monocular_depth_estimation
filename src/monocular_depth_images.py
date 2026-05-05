import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import pipeline


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def colorize_depth(depth_array: np.ndarray) -> Image.Image:
    depth_min = float(depth_array.min())
    depth_max = float(depth_array.max())
    if depth_max - depth_min < 1e-8:
        norm = np.zeros_like(depth_array, dtype=np.uint8)
    else:
        norm = ((depth_array - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    try:
        import matplotlib.cm as cm
        colored = (cm.inferno(norm / 255.0)[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(colored)
    except Exception:
        return Image.fromarray(norm)


def normalize01(arr: np.ndarray) -> np.ndarray:
    amin = float(arr.min())
    amax = float(arr.max())
    if amax - amin < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - amin) / (amax - amin)).astype(np.float32)


def simple_smooth(arr: np.ndarray, passes: int = 2) -> np.ndarray:
    out = arr.astype(np.float32).copy()
    for _ in range(passes):
        p = np.pad(out, ((1, 1), (1, 1)), mode='edge')
        out = (
            p[1:-1, 1:-1] * 4 +
            p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] +
            p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:]
        ) / 12.0
    return out


def largest_run(row: np.ndarray):
    best_len = 0
    best_start = 0
    cur_len = 0
    cur_start = 0
    for i, val in enumerate(row):
        if val:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    return best_start, best_len


def path_from_depth_anything(depth_np: np.ndarray):
    h, w = depth_np.shape
    depth01 = normalize01(depth_np)

    # Treat darker pixels as farther for guidance, as requested.
    far_map = 1.0 - depth01
    far_map = simple_smooth(far_map, passes=2)

    y0 = int(h * 0.55)
    roi = far_map[y0:, :]
    roi_h, roi_w = roi.shape

    if roi_h < 5 or roi_w < 5:
        return {
            'command': 'STOP',
            'message': 'Image too small for path guidance.',
            'roi_y0': y0,
            'best_path': None,
            'far_mask': np.zeros_like(roi, dtype=bool),
            'left_score': 0.0,
            'center_score': 0.0,
            'right_score': 0.0,
        }

    thr = float(np.quantile(roi, 0.72))
    far_mask = roi >= thr

    left_end = int(roi_w * 0.33)
    right_start = int(roi_w * 0.67)
    left = roi[:, :left_end]
    center = roi[:, left_end:right_start]
    right = roi[:, right_start:]

    left_score = float(left.mean()) if left.size else 0.0
    center_score = float(center.mean()) if center.size else 0.0
    right_score = float(right.mean()) if right.size else 0.0

    min_width = max(10, int(roi_w * 0.08))
    candidates = []
    for r in range(int(roi_h * 0.35), roi_h):
        start, length = largest_run(far_mask[r])
        if length < min_width:
            continue
        center_x = start + length / 2.0
        offset = (center_x - roi_w / 2.0) / (roi_w / 2.0)
        band_score = float(roi[r, start:start + length].mean()) if length > 0 else 0.0
        score = length * (0.6 + 0.4 * band_score)
        candidates.append({
            'row': int(r),
            'start': int(start),
            'length': int(length),
            'center': float(center_x),
            'offset': float(offset),
            'score': float(score),
            'band_score': band_score,
        })

    if not candidates:
        ordered = [('LEFT', left_score), ('CENTER', center_score), ('RIGHT', right_score)]
        best_dir, _ = max(ordered, key=lambda x: x[1])
        if best_dir == 'LEFT':
            command = 'TURN_LEFT'
            message = 'Left side looks farther away. Turn left.'
        elif best_dir == 'RIGHT':
            command = 'TURN_RIGHT'
            message = 'Right side looks farther away. Turn right.'
        else:
            command = 'MOVE_FORWARD'
            message = 'Center looks farther away. Move forward.'
        return {
            'command': command,
            'message': message,
            'roi_y0': y0,
            'best_path': None,
            'far_mask': far_mask,
            'left_score': left_score,
            'center_score': center_score,
            'right_score': right_score,
        }

    best = max(candidates, key=lambda x: x['score'])
    offset = best['offset']

    left_adv = left_score - center_score
    right_adv = right_score - center_score

    if center_score >= max(left_score, right_score):
        if left_score >= center_score * 0.92 and offset < -0.08:
            command = 'FORWARD_LEFT'
            message = 'Center is clear and left is also favorable. Move forward-left.'
        elif right_score >= center_score * 0.92 and offset > 0.08:
            command = 'FORWARD_RIGHT'
            message = 'Center is clear and right is also favorable. Move forward-right.'
        else:
            command = 'MOVE_FORWARD'
            message = 'Detected farther path is near the center. Move forward.'
    else:
        if offset < -0.18:
            command = 'TURN_LEFT'
            message = 'Detected farther path is on the left. Turn left.'
        elif offset > 0.18:
            command = 'TURN_RIGHT'
            message = 'Detected farther path is on the right. Turn right.'
        elif left_adv > 0.01:
            command = 'FORWARD_LEFT'
            message = 'Forward is possible, but left is slightly better. Move forward-left.'
        elif right_adv > 0.01:
            command = 'FORWARD_RIGHT'
            message = 'Forward is possible, but right is slightly better. Move forward-right.'
        else:
            command = 'MOVE_FORWARD'
            message = 'Detected farther path is near the center. Move forward.'

    return {
        'command': command,
        'message': message,
        'roi_y0': y0,
        'best_path': best,
        'far_mask': far_mask,
        'left_score': left_score,
        'center_score': center_score,
        'right_score': right_score,
    }


def draw_debug_overlay(image: Image.Image, analysis: dict, out_path: Path):
    img = image.copy().convert('RGB')
    draw = ImageDraw.Draw(img, 'RGBA')
    w, h = img.size
    y0 = analysis['roi_y0']
    draw.rectangle([0, y0, max(0, w - 1), max(0, h - 1)], outline=(0, 255, 255, 180), width=2)

    x1 = int(w * 0.33)
    x2 = int(w * 0.67)
    draw.line([x1, y0, x1, h - 1], fill=(255, 255, 0, 180), width=2)
    draw.line([x2, y0, x2, h - 1], fill=(255, 255, 0, 180), width=2)

    best = analysis['best_path']
    if best is not None:
        cy = min(h - 1, max(0, y0 + best['row']))
        bx0 = max(0, best['start'])
        bx1 = min(w - 1, best['start'] + best['length'] - 1)
        draw.rectangle([bx0, max(0, cy - 14), bx1, min(h - 1, cy + 14)], fill=(0, 255, 0, 90), outline=(0, 255, 0, 220), width=3)
        center = min(w - 1, max(0, best['center']))
        draw.line([center, max(0, cy - 22), center, min(h - 1, cy + 22)], fill=(0, 255, 0, 255), width=3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_summary(output_path: Path, image_name: str, analysis: dict):
    lines = [
        f'image: {image_name}',
        f'command: {analysis["command"]}',
        f'message: {analysis["message"]}',
        f'left_score: {analysis["left_score"]:.4f}',
        f'center_score: {analysis["center_score"]:.4f}',
        f'right_score: {analysis["right_score"]:.4f}',
    ]
    best = analysis['best_path']
    if best is not None:
        lines.extend([
            f'path_row: {best["row"]}',
            f'path_start: {best["start"]}',
            f'path_length: {best["length"]}',
            f'path_center: {best["center"]:.2f}',
            f'path_offset: {best["offset"]:.4f}',
            f'path_score: {best["score"]:.4f}',
            f'band_score: {best["band_score"]:.4f}',
        ])
    else:
        lines.append('best_path: none')
    output_path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Basic image processing with Depth Anything V2 to guide path.')
    parser.add_argument('input', help='Path to an input image or directory of images')
    parser.add_argument('--output', default='depth_output', help='Output directory for generated depth maps')
    parser.add_argument('--model', default='depth-anything/Depth-Anything-V2-Small-hf', help='Hugging Face model checkpoint')
    parser.add_argument('--device', default=None, help='cuda, cpu, or leave empty for auto')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = args.device
    else:
        device = 0 if torch.cuda.is_available() else 'cpu'

    print('RUNNING monocular_depth_images.py')
    print(f'Loading model: {args.model} on {device}')
    depth_estimator = pipeline('depth-estimation', model=args.model, device=device)

    if input_path.is_file() and is_image_file(input_path):
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted([p for p in input_path.iterdir() if p.is_file() and is_image_file(p)])
    else:
        raise ValueError('Input must be an image file or a directory containing images.')

    if not image_paths:
        raise ValueError('No supported image files found.')

    for image_path in image_paths:
        print(f'\nProcessing {image_path.name}')
        image = Image.open(image_path).convert('RGB')
        result = depth_estimator(image)
        predicted_depth = result['predicted_depth']

        if hasattr(predicted_depth, 'detach'):
            depth_np = predicted_depth.detach().cpu().numpy()
        else:
            depth_np = np.array(predicted_depth)

        norm = normalize01(depth_np)
        gray = (norm * 255).astype(np.uint8)

        gray_img = Image.fromarray(gray)
        color_img = colorize_depth(depth_np)
        analysis = path_from_depth_anything(depth_np)

        stem = image_path.stem
        gray_img.save(output_dir / f'{stem}_depth_gray.png')
        color_img.save(output_dir / f'{stem}_depth_color.png')
        draw_debug_overlay(image, analysis, output_dir / f'{stem}_debug_overlay.png')
        save_summary(output_dir / f'{stem}_decision.txt', image_path.name, analysis)

        print(f'LEFT SCORE      : {analysis["left_score"]:.4f}')
        print(f'CENTER SCORE    : {analysis["center_score"]:.4f}')
        print(f'RIGHT SCORE     : {analysis["right_score"]:.4f}')
        print(f'COMMAND         : {analysis["command"]}')
        print(f'ACTION          : {analysis["message"]}')
        if analysis['best_path'] is not None:
            b = analysis['best_path']
            print(f'BEST PATH       : start={b["start"]}, length={b["length"]}, offset={b["offset"]:.3f}, score={b["score"]:.3f}')
        else:
            print('BEST PATH       : none')

    print(f'\nSaved results to {output_dir.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)