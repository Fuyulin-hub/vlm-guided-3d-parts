import assert from 'node:assert/strict';
import test from 'node:test';

// A missing implementation is an assertion failure, not an import-time crash.
const core = await import('./semantic-core.mjs').catch(error => {
  if (error.code === 'ERR_MODULE_NOT_FOUND') return {};
  throw error;
});

function api(name) {
  assert.equal(typeof core[name], 'function', `${name} must be implemented`);
  return core[name];
}

const parts = [
  {part_id: 'part_05', label_id: 4, label_zh: '前轮', label_en: 'Front wheel', color: [199, 229, 22]},
  {part_id: 'part_09', label_id: 7, label_zh: '车身', label_en: 'Body', color: [23, 90, 200]},
];
const palette = parts.map(part => ({labelId: part.label_id, color: part.color}));

test('annotations bind by label_id rather than the numeric suffix of part_id', () => {
  const parsed = api('parseAnnotations')({uid: 'firetruck', run_id: 'r04', parts});
  assert.equal(parsed.parts.get(4).label, parts[0].label_en);
  assert.equal('label_en' in parsed.parts.get(4), false);
  assert.equal('label_zh' in parsed.parts.get(4), false);
  assert.equal(parts[0].label_zh, '前轮', 'raw input remains unchanged for hash verification');
  assert.equal(parsed.parts.has(5), false);
  assert.deepEqual(parsed.palette, palette);
  assert.equal(parsed.uid, 'firetruck');
  assert.equal(parsed.runId, 'r04');
});

test('English-only records remove nested Chinese names while preserving IDs and colors', () => {
  const source={label_id:7,label_en:'Wheel',label_zh:'车轮',color:[1,2,3],
    feedback:{name_zh:'车轮',status:'good'}};
  const exported=api('englishOnlyRecord')(source);
  assert.deepEqual(exported,{label_id:7,label:'Wheel',color:[1,2,3],feedback:{status:'good'}});
  assert.equal(source.label_zh,'车轮');
});

test('canonical label takes precedence over legacy names', () => {
  const exported=api('englishOnlyRecord')({label:'Wheel',label_en:'Old wheel',label_zh:'车轮'});
  assert.deepEqual(exported,{label:'Wheel'});
  assert.deepEqual(api('englishOnlyRecord')(exported),exported);
});

test('annotations reject empty part lists', () => {
  assert.throws(() => api('parseAnnotations')({parts: []}), /parts/i);
});

test('annotations reject fractional, zero and duplicate label IDs', () => {
  const parse = api('parseAnnotations');
  for (const label_id of [1.5, 0, -1, '4']) {
    assert.throws(() => parse({parts: [{...parts[0], label_id}]}), /label_id/i);
  }
  assert.throws(() => parse({parts: [parts[0], {...parts[1], label_id: 4}]}), /label_id/i);
});

test('annotations reject non-RGB and out-of-range palette channels', () => {
  const parse = api('parseAnnotations');
  for (const color of [[1, 2], [1, 2, 3, 4], [-1, 2, 3], [1, 256, 3], [1, 2.5, 3], ['1', 2, 3]]) {
    assert.throws(() => parse({parts: [{...parts[0], color}]}), /color/i);
  }
});

test('RGB matching preserves real label ID for exact and nearby colors', () => {
  const match = api('matchRgb');
  assert.equal(match([199, 229, 22], palette).labelId, 4);
  const nearby = match([201, 228, 24], palette);
  assert.equal(nearby.labelId, 4);
  assert.equal(nearby.reason, 'matched');
  assert.equal(nearby.distance, 3);
  assert.ok(nearby.margin >= 8);
});

test('RGB matching leaves distant colors unknown instead of forcing nearest label', () => {
  const match = api('matchRgb')([255, 0, 255], palette);
  assert.equal(match.labelId, null);
  assert.equal(match.reason, 'too_distant');
});

test('duplicate palette colors cannot identify a unique semantic label', () => {
  const match = api('matchRgb')([20, 30, 40], [
    {labelId: 1, color: [20, 30, 40]},
    {labelId: 2, color: [20, 30, 40]},
  ]);
  assert.equal(match.labelId, null);
  assert.equal(match.reason, 'ambiguous_color');
  assert.equal(match.margin, 0);
});

test('near-equal palette distances remain ambiguous even inside the distance threshold', () => {
  const match = api('matchRgb')([23, 30, 40], [
    {labelId: 1, color: [20, 30, 40]},
    {labelId: 2, color: [28, 30, 40]},
  ]);
  assert.equal(match.labelId, null);
  assert.equal(match.reason, 'ambiguous_color');
});

test('empty palette and nonfinite sample colors stay unknown', () => {
  const match = api('matchRgb');
  assert.equal(match([0, 0, 0], []).reason, 'empty_palette');
  for (const rgb of [null, [NaN, 0, 0], [Infinity, 0, 0], [1, 2], [-1, 0, 0]]) {
    assert.equal(match(rgb, palette).labelId, null);
    assert.equal(match(rgb, palette).reason, 'invalid_rgb');
  }
});

test('a face accepts one consistent label with exactly 80 percent coverage', () => {
  const result = api('decideFaceLabel')([parts[0].color, parts[0].color, null, parts[0].color, parts[0].color], palette);
  assert.equal(result.labelId, 4);
  assert.equal(result.reason, 'matched');
  assert.equal(result.coverage, 0.8);
  assert.equal(result.matchedSamples, 4);
  assert.equal(result.totalSamples, 5);
});

test('a face below 80 percent coverage stays unknown', () => {
  const result = api('decideFaceLabel')([parts[0].color, parts[0].color, parts[0].color, null, null], palette);
  assert.equal(result.labelId, null);
  assert.equal(result.reason, 'insufficient_coverage');
});

test('a known cross-part face is rejected even when one label dominates', () => {
  const result = api('decideFaceLabel')([
    parts[0].color, parts[0].color, parts[0].color, parts[0].color, parts[1].color,
  ], palette);
  assert.equal(result.labelId, null);
  assert.equal(result.reason, 'mixed_labels');
});

test('a face with no samples stays unknown', () => {
  const result = api('decideFaceLabel')([], palette);
  assert.equal(result.labelId, null);
  assert.equal(result.reason, 'no_samples');
  assert.equal(result.coverage, 0);
});

const rgbaImage = {
  width: 2, height: 2,
  data: new Uint8ClampedArray([
    255, 0, 0, 255,   0, 255, 0, 255,
    0, 0, 255, 255,   255, 255, 0, 255,
  ]),
};

test('RGBA UV sampling covers all four corners without adding another vertical flip', () => {
  const sample = api('sampleRGBA');
  assert.deepEqual(sample(rgbaImage, 0, 0), [255, 0, 0]);
  assert.deepEqual(sample(rgbaImage, 1, 0), [0, 255, 0]);
  assert.deepEqual(sample(rgbaImage, 0, 1), [0, 0, 255]);
  assert.deepEqual(sample(rgbaImage, 1, 1), [255, 255, 0]);
});

test('RGBA UV sampling uses floor and clips to image bounds', () => {
  const sample = api('sampleRGBA');
  assert.deepEqual(sample(rgbaImage, 0.49, 0.49), [255, 0, 0]);
  assert.deepEqual(sample(rgbaImage, 0.5, 0.5), [255, 255, 0]);
  assert.deepEqual(sample(rgbaImage, -2, 5), [0, 0, 255]);
});

test('transparent pixels and invalid UV/image data are not assigned labels', () => {
  const sample = api('sampleRGBA');
  const data = new Uint8Array([199, 229, 22, 10]);
  assert.equal(sample({width: 1, height: 1, data}, 0, 0), null);
  data[3] = 11;
  assert.deepEqual(sample({width: 1, height: 1, data}, 0, 0), [199, 229, 22]);
  assert.equal(sample(rgbaImage, NaN, 0), null);
  assert.equal(sample({width: 2, height: 2, data: []}, 0, 0), null);
});
