// Semantic lookup uses asset colors, never tone-mapped / lit framebuffer pixels.
const isRgb = color => (Array.isArray(color) || ArrayBuffer.isView(color)) &&
  color.length === 3 && color.every(value => Number.isFinite(value) && value >= 0 && value <= 255);

const chineseNameFields = new Set(['label_zh', 'name_zh', 'label_cn', 'name_cn',
  'labelZh', 'nameZh', 'zh_name', 'chinese_name']);

export function englishOnlyRecord(value) {
  if (Array.isArray(value)) return value.map(englishOnlyRecord);
  if (value && typeof value === 'object') {
    const exported=Object.fromEntries(Object.entries(value)
      .filter(([key]) => !chineseNameFields.has(key) && key !== 'label_en')
      .map(([key, item]) => [key, englishOnlyRecord(item)]));
    if ('label_en' in value && !('label' in exported)) exported.label=englishOnlyRecord(value.label_en);
    return exported;
  }
  return value;
}

export function parseAnnotations(doc) {
  if (!doc || !Array.isArray(doc.parts) || doc.parts.length === 0) {
    throw new Error('annotations.parts must be a nonempty array');
  }
  const parts = new Map();
  const palette = [];
  for (const part of doc.parts) {
    const labelId = part?.label_id;
    if (!Number.isSafeInteger(labelId) || labelId <= 0 || parts.has(labelId)) {
      throw new Error('Every label_id must be a unique positive integer');
    }
    if (!isRgb(part.color) || !part.color.every(Number.isInteger)) {
      throw new Error(`Invalid RGB color for label_id ${labelId}`);
    }
    parts.set(labelId, englishOnlyRecord(part));
    palette.push({labelId, color: [...part.color]});
  }
  return {parts, palette, uid: doc.uid ?? null, runId: doc.run_id ?? null};
}

export function matchRgb(rgb, palette, options = {}) {
  const maxDistance = options.maxDistance ?? 24;
  const minMargin = options.minMargin ?? 8;
  const unknown = reason => ({labelId: null, reason, distance: Infinity, margin: 0});
  if (!isRgb(rgb)) return unknown('invalid_rgb');
  if (!Array.isArray(palette) || palette.length === 0) return unknown('empty_palette');

  let best = Infinity;
  let second = Infinity;
  let labelId = null;
  for (const entry of palette) {
    const color = entry.color;
    const distance = Math.hypot(rgb[0] - color[0], rgb[1] - color[1], rgb[2] - color[2]);
    if (distance < best) {
      second = best;
      best = distance;
      labelId = entry.labelId;
    } else if (distance < second) {
      second = distance;
    }
  }
  const margin = second - best;
  if (best > maxDistance) return {labelId: null, reason: 'too_distant', distance: best, margin};
  if (margin === 0 || margin < minMargin) {
    return {labelId: null, reason: 'ambiguous_color', distance: best, margin};
  }
  return {labelId, reason: 'matched', distance: best, margin};
}

export function decideFaceLabel(samples, palette, options = {}) {
  const totalSamples = samples.length;
  let matchedSamples = 0;
  const labels = new Set();
  for (const sample of samples) {
    const match = matchRgb(sample, palette, options);
    if (match.labelId !== null) {
      matchedSamples++;
      labels.add(match.labelId);
    }
  }
  const coverage = totalSamples ? matchedSamples / totalSamples : 0;
  const stats = {coverage, matchedSamples, totalSamples};
  if (!totalSamples) return {labelId: null, reason: 'no_samples', ...stats};
  // A majority vote would erase genuine part boundaries; leave these faces unknown.
  if (labels.size > 1) return {labelId: null, reason: 'mixed_labels', ...stats};
  if (coverage < 0.8) return {labelId: null, reason: 'insufficient_coverage', ...stats};
  return {labelId: labels.values().next().value, reason: 'matched', ...stats};
}

export function sampleRGBA(image, u, v) {
  if (!image || !Number.isFinite(u) || !Number.isFinite(v)) return null;
  const {data, width, height} = image;
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0 ||
      !data || data.length < width * height * 4) return null;
  // UV has already passed through Texture.transformUv, including any flipY.
  const x = Math.max(0, Math.min(width - 1, Math.floor(u * width)));
  const y = Math.max(0, Math.min(height - 1, Math.floor(v * height)));
  const offset = (y * width + x) * 4;
  if (data[offset + 3] <= 10) return null;
  return [data[offset], data[offset + 1], data[offset + 2]];
}
