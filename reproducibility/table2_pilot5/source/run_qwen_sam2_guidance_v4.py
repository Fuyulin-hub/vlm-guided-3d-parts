#!/usr/bin/env python3
"""
Qwen3-VL v4: Exhaustive part detection (anti-under-segmentation) Structured part planning + multi-point prompts + feedback loop.
Supports ablation modes: box_only, point_only, box_point, multi_point, feedback.
Designed for assemblies subset first.
"""
import argparse, csv, hashlib, json, os, re, sys, time, traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path('/data2/codefile/fuyulin/experiments/objaverse_684')

# ============================================================
# V3 Prompt: Strict JSON with part_id, visibility, multi-points, hierarchy
# ============================================================
PROMPT_V3 = '''You are an exhaustive part-segmentation analyst. Look at the foreground object in this image.

Your task is to identify EVERY visible, structurally distinct component — no part is too small.
Scan systematically: top to bottom, left to right, center to edges.
For each region you see, ask yourself: is this a distinct part? If yes, include it.

TARGET COUNT: 8-15 parts for simple objects, 15-30 parts for complex objects.
It is MUCH better to over-segment than to miss parts. When in doubt, split it.

PART CATEGORIES TO CHECK (do not skip any):
- Main structural components (e.g., seat, back, legs, base, frame, body, lid)
- Small functional parts (e.g., knobs, buttons, handles, levers, switches, wheels, casters, feet)
- Connecting elements (e.g., hinges, joints, brackets, supports, crossbars, dowels, screws)
- Decorative/detailed features (e.g., trim, molding, carvings, patterns, grooves, seams, stitching)
- Each individual repeated part (each leg, each wheel, each slat — never merge repeated parts)
- Surface sub-regions with distinct shape, geometry, or material boundaries
- Openings and cavities (e.g., seat opening, armrest gap, hollow areas)

Return ONLY a valid JSON object with this exact format:

{
  "object_id": "auto",
  "view_id": "front",
  "object_zh": "中文物体名",
  "object_en": "English object name",
  "parts": [
    {
      "part_id": "part_01",
      "label_zh": "部件中文名",
      "label_en": "English part name",
      "parent_id": null,
      "visibility": "visible",
      "bbox_xyxy_norm": [x1, y1, x2, y2],
      "positive_points_norm": [[px1, py1], [px2, py2]],
      "negative_points_norm": [[nx1, ny1]],
      "occlusion_reason": null
    }
  ]
}

STRICT RULES:
1. Coordinates are integers 0-1000 (relative to image width/height).
2. part_id: sequential "part_01", "part_02", etc. Stable across views.
3. parent_id: null for top-level parts; part_id of parent for sub-parts (e.g., armrest -> chair).
4. visibility: one of "visible" (fully), "partial_occluded" (partially blocked), "hidden" (not visible - skip).
5. bbox_xyxy_norm: TIGHT bounding box around the part (not the whole object).
6. positive_points_norm: 2 points INSIDE the part, on different sub-regions (e.g., left+right of a seat).
7. negative_points_norm: 1 point OUTSIDE the part but on an ADJACENT part or background (helps SAM2 distinguish boundaries).
8. All positive_points must be inside the part's bbox. Negative points must be outside.
9. Do NOT invent hidden parts. But DO include all visible parts — even small ones.
10. occlusion_reason: brief string if visibility != "visible", else null.
11. Output ONLY the JSON. No markdown, no explanations, no code fences.
12. Part names should be specific. Use furniture/object part terminology (e.g., "armrest" not "side piece").
13. If you can see a clear boundary between two regions, treat them as separate parts.
14. Include ALL sizes: major, minor, and detail-level parts. Do not skip small parts.
'''


# ============================================================
# Feedback Prompt: mask overlay -> Qwen correction
# ============================================================
PROMPT_FEEDBACK = '''You are a mask quality inspector. The first image is the original object.
The second image has a colored mask overlay showing the current segmentation result for a specific part.

Part name (zh): {label_zh}
Part name (en): {label_en}

Judge the mask quality and output correction hints. Return ONLY a valid JSON object:

{{
  "status": "good" | "missing_part" | "boundary_leakage" | "adhesion" | "wrong_region",
  "error_type": "brief description of error or null if good",
  "correction_box": [x1, y1, x2, y2] | null,
  "correction_positive_points": [[px, py], ...] | null,
  "correction_negative_points": [[nx, ny], ...] | null
}}

Rules:
- Coordinates are integers 0-1000 (relative to image size).
- status "good": mask is accurate and complete. All correction fields null.
- status "missing_part": part is incomplete or truncated. Add positive points in missing area.
- status "boundary_leakage": mask leaks outside the part boundary. Add negative points on leaked region.
- status "adhesion": mask sticks to an adjacent part. Add negative points on the adjacent part near the boundary.
- status "wrong_region": mask is on entirely wrong part. Provide a corrected bbox.
- Output ONLY JSON. No explanations.
'''

def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root', type=Path, default=ROOT)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--model', default='/data2/codefile/fuyulin/1trellis2/qwen3vl/Qwen3-VL-4B-Instruct')
    p.add_argument('--sam-repo', default='/data2/codefile/fuyulin/fyl/segvigen/segment-anything-2')
    p.add_argument('--sam-checkpoint', default='/data2/codefile/fuyulin/fyl/segvigen/data_toolkit/assets/full_seg_w_2d_map/sam2.1_hiera_base_plus.pt')
    p.add_argument('--sam-config', default='configs/sam2.1_hiera_b+.yaml')
    p.add_argument('--palette', default='/data2/codefile/fuyulin/1trellis2/segvigen_inputs/colors_pure.json')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--uid', action='append')
    p.add_argument('--smoke-per-group', type=int, default=0)
    p.add_argument('--max-new-tokens', type=int, default=8192)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min-mask-pixels', type=int, default=10)
    p.add_argument('--max-retries', type=int, default=2)
    p.add_argument('--group-filter', default='assemblies', help='Only process this group (default: assemblies)')
    p.add_argument('--mode', default='multi_point',
                   choices=['box_only', 'point_only', 'box_point', 'multi_point', 'feedback'],
                   help='Ablation mode: what prompts to feed SAM2')
    p.add_argument('--output-suffix', default='C_v4_exhaustive',
                   help='Output directory suffix under guidance/ and annotations/')
    return p.parse_args()

def write_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.json'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(path)

def repair_json(s):
    """Aggressive JSON repair for Qwen-VL outputs."""
    original = s
    s = re.sub(r'^```(?:json)?\s*', '', s)
    s = re.sub(r'\s*```$', '', s)
    first_brace = s.find('{')
    last_brace = s.rfind('}')
    if first_brace >= 0 and last_brace > first_brace:
        s = s[first_brace:last_brace+1]
    s = re.sub(r',\s*([\]\}])', r'\1', s)
    s = re.sub(r"'([^']+)'\s*:", r'"\1":', s)
    s = re.sub(r'\}\s*\{', '},{', s)
    return s

def parse_parts_v3(raw, w, h):
    """Parse V3 structured JSON output. Returns (data, valid_parts, invalid_parts)."""
    text = raw.strip()
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        repaired = repair_json(text)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            raise ValueError(f'JSON parse failed even after repair')
    
    if not isinstance(data.get('parts'), list):
        data['parts'] = []
    
    valid = []
    invalid = []
    
    for i, item in enumerate(data['parts']):
        try:
            if not isinstance(item, dict):
                raise ValueError('part must be object')
            
            part_id = item.get('part_id', f'part_{i+1:02d}') or f'part_{i+1:02d}'
            label_zh = item.get('label_zh', item.get('name_zh', f'部件{i+1}')) or f'部件{i+1}'
            label_en = item.get('label_en', item.get('name_en', f'part_{i+1}')) or f'part_{i+1}'
            parent_id = item.get('parent_id', None)
            visibility = item.get('visibility', 'visible')
            occlusion_reason = item.get('occlusion_reason', None)
            
            # Bbox
            bbox_val = item.get('bbox_xyxy_norm', item.get('bbox', []))
            b = np.asarray(bbox_val, dtype=float)
            if b.shape != (4,):
                raise ValueError(f'bbox must be 4 elements, got {b.shape}')
            b = np.clip(b, 0, 1000)
            if b[2] <= b[0]: b[2] = min(b[0] + 10, 1000)
            if b[3] <= b[1]: b[3] = min(b[1] + 10, 1000)
            
            # Positive points
            pos_val = item.get('positive_points_norm', item.get('positive_points', []))
            pos_points = np.asarray(pos_val, dtype=float)
            if pos_points.ndim != 2 or pos_points.shape[1] != 2:
                # Try to use single point
                if pos_points.shape == (2,):
                    pos_points = pos_points.reshape(1, 2)
                else:
                    # Fallback: center of bbox
                    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
                    pos_points = np.array([[cx, cy]])
            
            pos_points = np.clip(pos_points, 0, 1000)
            # Ensure positive points inside bbox
            for j in range(len(pos_points)):
                pos_points[j, 0] = max(b[0], min(pos_points[j, 0], b[2]))
                pos_points[j, 1] = max(b[1], min(pos_points[j, 1], b[3]))
            
            # Negative points
            neg_val = item.get('negative_points_norm', item.get('negative_points', []))
            neg_points = np.asarray(neg_val, dtype=float) if neg_val else np.zeros((0, 2))
            if neg_points.ndim == 1 and len(neg_points) == 2:
                neg_points = neg_points.reshape(1, 2)
            if neg_points.ndim != 2 or neg_points.shape[1] != 2:
                neg_points = np.zeros((0, 2))
            neg_points = np.clip(neg_points, 0, 1000)
            
            # Convert to pixel coordinates
            b_px = (b * np.array([w, h, w, h]) / 1000).tolist()
            pos_px = (pos_points * np.array([w, h]) / 1000).tolist()
            neg_px = (neg_points * np.array([w, h]) / 1000).tolist() if len(neg_points) > 0 else []
            
            # Clamp to image bounds
            pos_px = [[min(max(p[0], 0), w-1), min(max(p[1], 0), h-1)] for p in pos_px]
            neg_px = [[min(max(p[0], 0), w-1), min(max(p[1], 0), h-1)] for p in neg_px]
            
            # Skip hidden parts
            if visibility == 'hidden':
                invalid.append(dict(part_id=part_id, source_index=i,
                                   reason='hidden part skipped',
                                   label_zh=label_zh, label_en=label_en))
                continue
            
            valid.append(dict(
                part_id=part_id, source_index=i,
                label_zh=label_zh, label_en=label_en,
                parent_id=parent_id, visibility=visibility,
                occlusion_reason=occlusion_reason,
                bbox_norm=b.tolist(),
                positive_points_norm=pos_points.tolist(),
                negative_points_norm=neg_points.tolist(),
                bbox=b_px,
                positive_points=pos_px,
                negative_points=neg_px,
            ))
        except Exception as e:
            invalid.append(dict(source_index=i, reason=str(e),
                               raw=item if isinstance(item, dict) else str(item)[:200]))
    
    return data, valid, invalid

def parse_feedback(raw, w, h):
    """Parse feedback JSON output."""
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        repaired = repair_json(text)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            return {'status': 'parse_error', 'error_type': 'json parse failed',
                    'correction_box': None, 'correction_positive_points': None,
                    'correction_negative_points': None}
    
    # Convert coordinates from 0-1000 to pixels
    result = {
        'status': data.get('status', 'unknown'),
        'error_type': data.get('error_type', None),
    }
    
    cb = data.get('correction_box')
    if cb and isinstance(cb, list) and len(cb) == 4:
        cb_arr = np.asarray(cb, dtype=float)
        cb_arr = np.clip(cb_arr, 0, 1000)
        result['correction_box'] = (cb_arr * np.array([w, h, w, h]) / 1000).tolist()
    else:
        result['correction_box'] = None
    
    cpp = data.get('correction_positive_points')
    if cpp and isinstance(cpp, list) and len(cpp) > 0:
        cpp_arr = np.asarray(cpp, dtype=float)
        cpp_arr = np.clip(cpp_arr, 0, 1000)
        result['correction_positive_points'] = (cpp_arr * np.array([w, h]) / 1000).tolist()
    else:
        result['correction_positive_points'] = None
    
    cnp = data.get('correction_negative_points')
    if cnp and isinstance(cnp, list) and len(cnp) > 0:
        cnp_arr = np.asarray(cnp, dtype=float)
        cnp_arr = np.clip(cnp_arr, 0, 1000)
        result['correction_negative_points'] = (cnp_arr * np.array([w, h]) / 1000).tolist()
    else:
        result['correction_negative_points'] = None
    
    return result

def generate_qwen(model, processor, messages, max_new_tokens, do_sample, temperature, top_p, device):
    with torch.inference_mode():
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors='pt'
        ).to(device)
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
        )
    torch.cuda.synchronize()
    raw = processor.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    return raw

def find_nearby_fg(point, fg, max_radius=50):
    """Find nearest foreground pixel to the given point."""
    px, py = int(round(point[0])), int(round(point[1]))
    h, w = fg.shape
    px = min(w-1, max(0, px)); py = min(h-1, max(0, py))
    if fg[py, px]:
        return [float(px), float(py)], True
    for radius in [3, 5, 10, 20, 50]:
        y_slice = slice(max(0, py-radius), min(h, py+radius+1))
        x_slice = slice(max(0, px-radius), min(w, px+radius+1))
        roi = fg[y_slice, x_slice]
        if roi.any():
            ry, rx = np.unravel_index(np.argmax(roi), roi.shape)
            py_new, px_new = y_slice.start + ry, x_slice.start + rx
            if fg[py_new, px_new]:
                return [float(px_new), float(py_new)], True
    return [float(px), float(py)], False

def run_sam2_for_part(predictor, rgb, fg, part, mode, min_mask_pixels):
    """
    Run SAM2 for a single part based on the mode.
    Returns (mask, score, used_prompts) or None if failed.
    """
    h, w = fg.shape
    
    if mode == 'box_only':
        masks, scores, _ = predictor.predict(
            box=np.asarray(part['bbox'], dtype=np.float32),
            multimask_output=False
        )
        used = {'type': 'box_only', 'bbox': part['bbox']}
    
    elif mode == 'point_only':
        # Use first positive point
        pt = part['positive_points'][0]
        px, py = int(round(pt[0])), int(round(pt[1]))
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray([[px, py]], dtype=np.float32),
            point_labels=np.array([1]),
            multimask_output=False
        )
        used = {'type': 'point_only', 'point': [px, py]}
    
    elif mode == 'box_point':
        pt = part['positive_points'][0]
        px, py = int(round(pt[0])), int(round(pt[1]))
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray([[px, py]], dtype=np.float32),
            point_labels=np.array([1]),
            box=np.asarray(part['bbox'], dtype=np.float32),
            multimask_output=False
        )
        used = {'type': 'box_point', 'bbox': part['bbox'], 'point': [px, py]}
    
    elif mode in ('multi_point', 'feedback'):
        # Collect all positive and negative points
        all_points = []
        all_labels = []
        
        for pt in part['positive_points']:
            px, py = int(round(pt[0])), int(round(pt[1]))
            px, py = min(w-1, max(0, px)), min(h-1, max(0, py))
            all_points.append([px, py])
            all_labels.append(1)  # foreground
        
        for pt in part.get('negative_points', []):
            px, py = int(round(pt[0])), int(round(pt[1]))
            px, py = min(w-1, max(0, px)), min(h-1, max(0, py))
            all_points.append([px, py])
            all_labels.append(0)  # background
        
        if not all_points:
            return None, 0, {}
        
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray(all_points, dtype=np.float32),
            point_labels=np.array(all_labels),
            box=np.asarray(part['bbox'], dtype=np.float32),
            multimask_output=False
        )
        used = {
            'type': 'multi_point',
            'bbox': part['bbox'],
            'positive_points': [p for i, p in enumerate(all_points) if all_labels[i] == 1],
            'negative_points': [p for i, p in enumerate(all_points) if all_labels[i] == 0],
        }
    else:
        raise ValueError(f'Unknown mode: {mode}')
    
    region = np.asarray(masks[0]).astype(bool, copy=False) & fg
    if region.sum() < min_mask_pixels:
        return None, float(scores[0]), used
    
    return region, float(scores[0]), used

def create_mask_overlay(rgb, mask, color=[255, 0, 0], alpha=0.4):
    """Create a mask overlay image for feedback."""
    overlay = rgb.copy()
    mask_uint8 = mask.astype(np.uint8) * 255
    colored = np.zeros_like(rgb)
    colored[:, :] = color
    overlay = cv2.addWeighted(colored, alpha, overlay, 1 - alpha, 0)
    overlay[~mask] = rgb[~mask]
    # Add boundary
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay

def main():
    a = args()
    root = a.experiment_root
    manifest = a.manifest or root / 'manifest_frozen.csv'
    rows = list(csv.DictReader(manifest.open(encoding='utf-8', newline='')))
    
    # Filter by group
    selected = []
    counts = {}
    for r in rows:
        g = r['group']
        if a.group_filter and g != a.group_filter:
            continue
        if a.uid and r['uid'] not in a.uid:
            continue
        counts.setdefault(g, 0)
        if a.smoke_per_group and counts[g] >= a.smoke_per_group:
            continue
        counts[g] += 1
        selected.append(r)
    
    print(f'Mode: {a.mode}')
    print(f'Output suffix: {a.output_suffix}')
    print(f'Selected {len(selected)} samples from groups: {counts}')
    
    palette = np.asarray([x[:3] for x in json.loads(Path(a.palette).read_text())], dtype=np.uint8)
    if len(palette) < 2 or len(np.unique(palette[1:], axis=0)) != len(palette) - 1:
        raise ValueError('foreground palette not unique')
    if np.any(np.all(palette[1:] == 0, axis=1)):
        raise ValueError('foreground palette contains background')
    
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    
    out_suffix = a.output_suffix
    log = root / 'logs' / f'{out_suffix}_results.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    
    config = dict(vars(a))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config.update(
        prompt_v3=PROMPT_V3,
        prompt_feedback=PROMPT_FEEDBACK,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    write_json(root / 'logs' / f'{out_suffix}_config.json', config)
    
    # Load models
    print('Loading Qwen3-VL...')
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa'
    ).to(a.device).eval()
    
    print('Loading SAM2...')
    sys.path.insert(0, a.sam_repo)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(build_sam2(a.sam_config, a.sam_checkpoint, device=a.device))
    
    retry_configs = [
        (False, 1.0, 1.0),
        (True, 0.7, 0.9),
        (True, 1.0, 0.95),
    ]
    retry_configs = retry_configs[:max(1, a.max_retries)]
    
    for idx, r in enumerate(selected, 1):
        g, u = r['group'], r['uid']
        out = root / 'guidance' / out_suffix / g / (u + '.png')
        ann = root / 'annotations' / out_suffix / g / (u + '.json')
        
        if out.exists() and ann.exists():
            try:
                prev = json.loads(ann.read_text())
                if prev.get('status') in ('ok', 'partial'):
                    print(f'[{idx}/{len(selected)}] cached {g}/{u}', flush=True)
                    continue
            except:
                pass
        
        t = time.perf_counter()
        rec = dict(group=g, uid=u, status='failed', output_map=str(out),
                   invalid_parts=[], retries=0, mode=a.mode)
        
        try:
            rgba = np.array(Image.open(r['input_rgba']).convert('RGBA'))
            rgb = rgba[:, :, :3]
            fg = rgba[:, :, 3] > 10
            h, w = fg.shape
            
            # === Step 1: Qwen initial part planning ===
            raw = None
            data = None
            parts = []
            invalid = []
            
            for attempt, (do_sample, temp, top_p) in enumerate(retry_configs):
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'image': Image.fromarray(rgb)},
                    {'type': 'text', 'text': PROMPT_V3}
                ]}]
                
                tq = time.perf_counter()
                raw = generate_qwen(model, processor, messages, a.max_new_tokens,
                                   do_sample, temp, top_p, a.device)
                rec['qwen_seconds'] = rec.get('qwen_seconds', 0) + time.perf_counter() - tq
                rec['retries'] = attempt + 1
                
                rawpath = root / 'annotations' / out_suffix / g / (u + '.raw_v3.txt')
                rawpath.parent.mkdir(parents=True, exist_ok=True)
                rawpath.write_text(raw, encoding='utf-8')
                
                try:
                    data, parts, invalid = parse_parts_v3(raw, w, h)
                    if parts:
                        break
                except Exception:
                    if attempt < len(retry_configs) - 1:
                        continue
                    raise
            
            if not parts:
                # Fallback: whole foreground
                fg_rows = np.any(fg, axis=1)
                fg_cols = np.any(fg, axis=0)
                y1, y2 = np.where(fg_rows)[0][[0, -1]]
                x1, x2 = np.where(fg_cols)[0][[0, -1]]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if not fg[cy, cx]:
                    dist = cv2.distanceTransform((~fg).astype(np.uint8), cv2.DIST_L2, 5)
                    roi_dist = dist[y1:y2+1, x1:x2+1]
                    min_idx = np.unravel_index(np.argmin(roi_dist), roi_dist.shape)
                    cy, cx = y1 + min_idx[0], x1 + min_idx[1]
                
                data = {'object_zh': '物体', 'object_en': 'object', 'parts': []}
                parts = [{
                    'part_id': 'part_01',
                    'source_index': -1,
                    'label_zh': '整体',
                    'label_en': 'whole',
                    'parent_id': None,
                    'visibility': 'visible',
                    'occlusion_reason': None,
                    'bbox_norm': [x1*1000/w, y1*1000/h, x2*1000/w, y2*1000/h],
                    'positive_points_norm': [[cx*1000/w, cy*1000/h]],
                    'negative_points_norm': [],
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'positive_points': [[float(cx), float(cy)]],
                    'negative_points': [],
                }]
                invalid.append({'source_index': -1, 'reason': 'all Qwen attempts failed, used whole-object fallback'})
            
            rec.update(raw_parts=len(data.get('parts', [])),
                      valid_parts=len(parts),
                      invalid_parts=invalid,
                      object_zh=data.get('object_zh'),
                      object_en=data.get('object_en'))
            
            # === Step 2: SAM2 initial mask generation ===
            ts = time.perf_counter()
            candidates = []
            maskdir = root / 'masks' / out_suffix / g / u
            maskdir.mkdir(parents=True, exist_ok=True)
            
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                predictor.set_image(rgb)
                
                for part in parts:
                    # Ensure at least first positive point is on foreground
                    if part['positive_points']:
                        pt, found = find_nearby_fg(part['positive_points'][0], fg)
                        if found:
                            part['positive_points'][0] = pt
                        else:
                            invalid.append(dict(part_id=part.get('part_id'),
                                               reason='positive point not on foreground'))
                            continue
                    
                    region, score, used = run_sam2_for_part(
                        predictor, rgb, fg, part, a.mode, a.min_mask_pixels
                    )
                    
                    if region is None:
                        invalid.append(dict(part_id=part.get('part_id'),
                                           reason=f'SAM mask too small or failed (score={score:.3f})'))
                        continue
                    
                    part['sam_predicted_iou'] = score
                    part['sam_prompts_used'] = used
                    
                    # Save raw mask
                    Image.fromarray((region * 255).astype(np.uint8)).save(
                        maskdir / (part['part_id'] + '_initial.png'))
                    
                    candidates.append((part, region))
            
            # === Step 3: Feedback loop (if mode == 'feedback') ===
            feedback_rounds = 0
            if a.mode == 'feedback' and candidates:
                feedback_rounds = 1
                improved_candidates = []
                
                for part, initial_mask in candidates:
                    # Create mask overlay
                    overlay = create_mask_overlay(rgb, initial_mask, color=[255, 80, 80], alpha=0.35)
                    
                    # Ask Qwen to evaluate
                    fb_prompt = PROMPT_FEEDBACK.format(
                        label_zh=part['label_zh'],
                        label_en=part['label_en']
                    )
                    
                    messages = [{'role': 'user', 'content': [
                        {'type': 'image', 'image': Image.fromarray(rgb)},
                        {'type': 'image', 'image': Image.fromarray(overlay)},
                        {'type': 'text', 'text': fb_prompt}
                    ]}]
                    
                    tf = time.perf_counter()
                    fb_raw = generate_qwen(model, processor, messages, 1024,
                                          False, 1.0, 1.0, a.device)
                    rec['feedback_seconds'] = rec.get('feedback_seconds', 0) + time.perf_counter() - tf
                    
                    fb_rawpath = root / 'annotations' / out_suffix / g / (u + f'.fb_{part["part_id"]}.txt')
                    fb_rawpath.write_text(fb_raw, encoding='utf-8')
                    
                    fb_result = parse_feedback(fb_raw, w, h)
                    part['feedback'] = fb_result
                    
                    if fb_result['status'] == 'good' or fb_result['status'] == 'parse_error':
                        # Keep initial mask
                        improved_candidates.append((part, initial_mask))
                        continue
                    
                    # Build corrected prompts
                    corrected_part = dict(part)
                    
                    if fb_result['correction_box']:
                        corrected_part['bbox'] = fb_result['correction_box']
                    
                    if fb_result['correction_positive_points']:
                        corrected_part['positive_points'] = fb_result['correction_positive_points']
                    
                    if fb_result['correction_negative_points']:
                        corrected_part['negative_points'] = corrected_part.get('negative_points', []) + \
                                                            fb_result['correction_negative_points']
                    
                    # Re-run SAM2 with corrected prompts
                    corrected_mask, corrected_score, corrected_used = run_sam2_for_part(
                        predictor, rgb, fg, corrected_part, 'multi_point', a.min_mask_pixels
                    )
                    
                    if corrected_mask is not None:
                        part['sam_predicted_iou_after_feedback'] = corrected_score
                        part['sam_prompts_after_feedback'] = corrected_used
                        Image.fromarray((corrected_mask * 255).astype(np.uint8)).save(
                            maskdir / (part['part_id'] + '_after_feedback.png'))
                        improved_candidates.append((part, corrected_mask))
                    else:
                        # Keep initial
                        improved_candidates.append((part, initial_mask))
                
                candidates = improved_candidates
            
            rec['sam_seconds'] = time.perf_counter() - ts
            rec['feedback_rounds'] = feedback_rounds
            
            # === Step 4: Assemble final label map ===
            if not candidates:
                label = np.zeros((h, w), np.uint16)
                label[fg] = 1
                accepted = [{
                    'part_id': 'part_01', 'source_index': -1,
                    'label_zh': '整体', 'label_en': 'whole',
                    'label_id': 1, 'palette_index': 1,
                    'sam_predicted_iou': 0.0,
                }]
                invalid.append({'source_index': -1, 'reason': 'all parts failed, used full foreground fallback'})
            else:
                candidates.sort(key=lambda x: int(x[1].sum()), reverse=True)
                label = np.zeros((h, w), np.uint16)
                accepted = []
                
                for part, region in candidates:
                    region = region & (label == 0)
                    if region.sum() < a.min_mask_pixels:
                        invalid.append(dict(part_id=part.get('part_id'),
                                           reason='too few pixels after overlap arbitration'))
                        continue
                    label_id = len(accepted) + 1
                    label[region] = label_id
                    part.update(label_id=label_id,
                              palette_index=(label_id - 1) % (len(palette) - 1) + 1)
                    accepted.append(part)
                
                if not accepted:
                    part, region = candidates[0]
                    label_id = 1
                    label[region] = label_id
                    part.update(label_id=label_id, palette_index=1)
                    accepted = [part]
                    invalid.append({'source_index': -1, 'reason': 'overlap arbitration removed all, kept largest'})
            
            # Fill holes
            missing = fg & (label == 0)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            for _ in range(max(h, w)):
                if not missing.any():
                    break
                grown = cv2.dilate(label, kernel, iterations=1)
                fill = missing & (grown > 0)
                if not fill.any():
                    break
                label[fill] = grown[fill]
                missing = fg & (label == 0)
            
            # Save individual masks and build guidance
            for part in accepted:
                part['mask_file'] = str(maskdir / (part['part_id'] + '.png'))
                Image.fromarray(((label == part['label_id']) * 255).astype(np.uint8)).save(part['mask_file'])
                part['color'] = palette[part['palette_index']].tolist()
            
            Image.fromarray(label).save(maskdir / 'labels_uint16.png')
            color_ids = np.where(label > 0, (label.astype(np.int32) - 1) % (len(palette) - 1) + 1, 0)
            guide = palette[color_ids]
            guide[~fg] = 0
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix('.tmp.png')
            Image.fromarray(guide).save(tmp)
            tmp.replace(out)
            
            torch.cuda.synchronize()
            rec.update(
                status='partial' if invalid else 'ok',
                painted_regions=len(accepted),
                foreground_pixels=int(fg.sum()),
                foreground_black_pixels=int((fg & (label == 0)).sum()),
            )
            
            # Strip large arrays from annotation
            clean_parts = []
            for p in accepted:
                cp = {k: v for k, v in p.items() if k != 'sam_prompts_used' or True}
                clean_parts.append(cp)
            
            write_json(ann, dict(rec,
                              image_size=[w, h],
                              coordinate_system='original pixel xyxy; source normalized 0-1000',
                              parts=clean_parts))
        
        except Exception as e:
            rec.update(status='failed', error=repr(e))
            traceback.print_exc()
            write_json(ann, rec)
        
        rec['seconds'] = time.perf_counter() - t
        with log.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        
        print(f"[{idx}/{len(selected)}] {rec['status']} {g}/{u} "
              f"parts={rec.get('painted_regions', 0)} "
              f"{rec['seconds']:.1f}s (retry={rec.get('retries', 0)})",
              flush=True)
    
    print('DONE', flush=True)

if __name__ == '__main__':
    main()
