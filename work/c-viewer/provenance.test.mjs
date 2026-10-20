import test from 'node:test';
import assert from 'node:assert/strict';
const api = await import('./provenance.mjs').catch(() => ({}));
const a = 'a'.repeat(64), b = 'b'.repeat(64), c = 'c'.repeat(64);
test('matching explicit output GLB and guidance hashes verify their association', () => {
  assert.equal(typeof api.verifyPair, 'function', 'pair verification is not implemented');
  assert.equal(api.verifyPair(a, {uid:'one',final_map:{sha256:b}}, {output_glb_sha256:a,guidance_sha256:b}).status, 'verified');
});
test('a wrong GLB hash fails closed', () => {
  assert.equal(typeof api.verifyPair, 'function');
  assert.throws(() => api.verifyPair(c,{final_map:{sha256:b}},{output_glb_sha256:a,guidance_sha256:b}), /GLB/);
});
test('a wrong guidance version fails closed', () => {
  assert.equal(typeof api.verifyPair, 'function');
  assert.throws(() => api.verifyPair(a,{final_map:{sha256:c}},{output_glb_sha256:a,guidance_sha256:b}), /引导/);
});
test('missing provenance never claims verification', () => {
  assert.equal(typeof api.verifyPair, 'function');
  assert.equal(api.verifyPair(a,{parts:[]},null).status, 'unverified');
});
test('different explicit UIDs fail even when hashes are supplied', () => {
  assert.equal(typeof api.verifyPair, 'function');
  assert.throws(() => api.verifyPair(a,{uid:'one',final_map:{sha256:b}},{uid:'two',output_glb_sha256:a,guidance_sha256:b}), /UID/);
});
test('legacy glb_sha256 cannot verify the output even when it matches', () => {
  assert.equal(api.verifyPair(a,{final_map:{sha256:b}},{glb_sha256:a,guidance_sha256:b}).status,'unverified');
});
test('legacy input glb_sha256 does not reject a different segmented output', () => {
  assert.equal(api.verifyPair(a,{final_map:{sha256:b}},{glb_sha256:c,guidance_sha256:b}).status,'unverified');
});
test('explicit output hash takes precedence without interpreting the legacy input field', () => {
  assert.equal(api.verifyPair(a,{final_map:{sha256:b}},{output_glb_sha256:a,glb_sha256:c,guidance_sha256:b}).status,'verified');
});
test('the annotation file hash is verified when it is declared', () => {
  assert.equal(api.verifyPair(a,{final_map:{sha256:b}},{output_glb_sha256:a,guidance_sha256:b,annotation_sha256:c},c).status,'verified');
});
test('a declared annotation hash conflict fails closed', () => {
  assert.throws(() => api.verifyPair(a,{final_map:{sha256:b}},{output_glb_sha256:a,guidance_sha256:b,annotation_sha256:c},a), /标注/);
});
test('a declared annotation hash without a measured hash remains unverified', () => {
  assert.equal(api.verifyPair(a,{final_map:{sha256:b}},{output_glb_sha256:a,guidance_sha256:b,annotation_sha256:c}).status,'unverified');
});
test('partial output or guidance records do not claim verification', () => {
  assert.equal(api.verifyPair(a,{final_map:{sha256:b}},{output_glb_sha256:a}).status,'unverified');
  assert.equal(api.verifyPair(a,{}, {output_glb_sha256:a,guidance_sha256:b}).status,'unverified');
});
test('malformed explicit hashes fail closed rather than throwing accidental type errors', () => {
  assert.throws(() => api.verifyPair(a,{final_map:{sha256:b}},{output_glb_sha256:42,guidance_sha256:b}), /GLB/);
  assert.throws(() => api.verifyPair(a,{final_map:{sha256:b}},{output_glb_sha256:a,guidance_sha256:42}), /引导/);
  assert.throws(() => api.verifyPair(a,{final_map:{sha256:b}},{output_glb_sha256:a,guidance_sha256:b,annotation_sha256:'invalid'},c), /标注/);
});
test('hash comparisons accept hexadecimal letter case', () => {
  assert.equal(api.verifyPair(a,{final_map:{sha256:b.toUpperCase()}},{output_glb_sha256:a.toUpperCase(),guidance_sha256:b,annotation_sha256:c.toUpperCase()},c).status,'verified');
});
test('GLB inspection permits embedded assets but refuses external resource fetches', () => {
  assert.equal(typeof api.inspectGLB, 'function');
  const make = doc => {const json=new TextEncoder().encode(JSON.stringify(doc));const n=Math.ceil(json.length/4)*4;const out=new Uint8Array(20+n);const d=new DataView(out.buffer);d.setUint32(0,0x46546c67,true);d.setUint32(4,2,true);d.setUint32(8,out.length,true);d.setUint32(12,n,true);d.setUint32(16,0x4e4f534a,true);out.fill(32,20);out.set(json,20);return out.buffer;};
  assert.equal(api.inspectGLB(make({asset:{version:'2.0'},buffers:[{byteLength:0}]})).asset.version,'2.0');
  assert.throws(() => api.inspectGLB(make({asset:{version:'2.0'},images:[{uri:'https://untrusted.example/a.png'}]})), /外部/);
});

const JSON_CHUNK=0x4e4f534a, BIN_CHUNK=0x004e4942;
function jsonChunk(doc) {
  const text=new TextEncoder().encode(JSON.stringify(doc));
  const data=new Uint8Array(Math.ceil(text.length/4)*4).fill(32);
  data.set(text);
  return {type:JSON_CHUNK,data};
}
function chunkedGlb(chunks, trailing=new Uint8Array()) {
  const bytes=new Uint8Array(12+chunks.reduce((n,c)=>n+8+c.data.length,0)+trailing.length);
  const view=new DataView(bytes.buffer);
  view.setUint32(0,0x46546c67,true);view.setUint32(4,2,true);view.setUint32(8,bytes.length,true);
  let offset=12;
  for(const chunk of chunks) {
    view.setUint32(offset,chunk.declaredLength??chunk.data.length,true);
    view.setUint32(offset+4,chunk.type,true);
    bytes.set(chunk.data,offset+8);offset+=8+chunk.data.length;
  }
  bytes.set(trailing,offset);
  return bytes.buffer;
}
const safeDocument={asset:{version:'2.0'},buffers:[{byteLength:4}]};
test('GLB inspection accepts a single JSON followed by a bounded aligned BIN', () => {
  const glb=chunkedGlb([jsonChunk(safeDocument),{type:BIN_CHUNK,data:new Uint8Array(4)}]);
  assert.equal(api.inspectGLB(glb).asset.version,'2.0');
});
test('duplicate JSON cannot replace the inspected document with an external-resource document', () => {
  const malicious={asset:{version:'2.0'},images:[{uri:'https://untrusted.example/replaced.png'}]};
  assert.throws(()=>api.inspectGLB(chunkedGlb([jsonChunk(safeDocument),jsonChunk(malicious)])),/JSON/);
});
test('the JSON chunk must be first even if another JSON appears later', () => {
  assert.throws(()=>api.inspectGLB(chunkedGlb([{type:BIN_CHUNK,data:new Uint8Array(4)},jsonChunk(safeDocument)])),/JSON/);
});
test('duplicate BIN chunks are rejected', () => {
  const bin={type:BIN_CHUNK,data:new Uint8Array(4)};
  assert.throws(()=>api.inspectGLB(chunkedGlb([jsonChunk(safeDocument),bin,bin])),/BIN/);
});
test('truncated chunk headers after valid JSON are rejected', () => {
  assert.throws(()=>api.inspectGLB(chunkedGlb([jsonChunk(safeDocument)],new Uint8Array(4))),/chunk|截断/i);
});
test('later chunk payload lengths cannot overflow the actual GLB buffer', () => {
  const glb=chunkedGlb([jsonChunk(safeDocument),{type:BIN_CHUNK,data:new Uint8Array(4),declaredLength:0xfffffffc}]);
  assert.throws(()=>api.inspectGLB(glb),/chunk|边界|长度/i);
});
test('declared chunk lengths must be four-byte aligned even inside an aligned file', () => {
  const glb=chunkedGlb([jsonChunk(safeDocument),{type:BIN_CHUNK,data:new Uint8Array(4),declaredLength:3}]);
  assert.throws(()=>api.inspectGLB(glb),/对齐|aligned/i);
});
test('an unaligned complete GLB length is rejected', () => {
  const glb=chunkedGlb([jsonChunk(safeDocument)],new Uint8Array(1));
  assert.throws(()=>api.inspectGLB(glb),/对齐|aligned/i);
});
test('an unknown bounded chunk cannot be interpreted as replacement JSON', () => {
  const unknown=jsonChunk({images:[{uri:'https://untrusted.example/not-a-json-chunk.png'}]});
  unknown.type=0x12345678;
  const glb=chunkedGlb([jsonChunk(safeDocument),unknown]);
  assert.deepEqual(api.inspectGLB(glb),safeDocument);
});
