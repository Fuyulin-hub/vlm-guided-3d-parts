const validHash = value => typeof value === 'string' && /^[a-f0-9]{64}$/i.test(value);
const supplied = value => value !== undefined && value !== null;
function checkHash(expected, actual, label) {
  if (!supplied(expected)) return false;
  if (!validHash(expected) || (supplied(actual) && !validHash(actual))) {
    throw new Error(`${label} SHA256 格式无效，已阻止语义绑定`);
  }
  if (!supplied(actual)) return false;
  if (expected.toLowerCase() !== actual.toLowerCase()) {
    throw new Error(`${label} SHA256 不一致，已阻止语义绑定`);
  }
  return true;
}
export function verifyPair(glbHash, annotation, meta, annotationHash) {
  if (meta?.uid && annotation?.uid && meta.uid !== annotation.uid) throw new Error('UID 不一致：模型与标注不属于同一样本');
  // Legacy glb_sha256 may describe input geometry, not the segmented output.
  // Never use that ambiguous field to accept or reject this output GLB.
  const outputChecked = checkHash(meta?.output_glb_sha256, glbHash, '输出 GLB');
  const guidanceChecked = checkHash(meta?.guidance_sha256, annotation?.final_map?.sha256, '引导图版本');
  const annotationDeclared = supplied(meta?.annotation_sha256);
  const annotationChecked = annotationDeclared ? checkHash(meta.annotation_sha256, annotationHash, '标注文件') : true;
  if (outputChecked && guidanceChecked && annotationChecked) {
    return {status:'verified', message: annotationDeclared ? '输出 GLB、引导图与标注文件 SHA256 均已核验' : '输出 GLB 与引导图 SHA256 均已核验'};
  }
  return {status:'unverified', message:'缺少完整来源记录，需要人工确认同源；这不等于哈希核验通过'};
}

export function inspectGLB(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 20) throw new Error('GLB 文件过短');
  const view = new DataView(buffer);
  if (view.getUint32(0,true)!==0x46546c67 || view.getUint32(4,true)!==2) throw new Error('仅支持 glTF 2.0 二进制 GLB');
  if (view.getUint32(8,true)!==buffer.byteLength) throw new Error('GLB 文件长度与头部不一致');
  if (buffer.byteLength%4!==0) throw new Error('GLB 文件长度必须四字节对齐');
  let offset=12, seenJSON=false, seenBIN=false, doc;
  while(offset<buffer.byteLength) {
    if(buffer.byteLength-offset<8) throw new Error('GLB chunk 头部截断');
    const length=view.getUint32(offset,true), type=view.getUint32(offset+4,true);
    if(length%4!==0) throw new Error('GLB chunk 长度必须四字节对齐');
    if(length>buffer.byteLength-offset-8) throw new Error('GLB chunk 长度超出文件边界');
    if(offset===12 && type!==0x4e4f534a) throw new Error('GLB 首个 chunk 必须是 JSON');
    if(type===0x4e4f534a) {
      // GLTFLoader walks all chunks; accepting a second JSON would let it replace
      // the document whose resource URIs we inspected before calling the loader.
      if(seenJSON) throw new Error('GLB 不允许重复 JSON chunk');
      seenJSON=true;
      doc=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,offset+8,length)).replace(/\0+$/,''));
    } else if(type===0x004e4942) {
      if(seenBIN) throw new Error('GLB 不允许重复 BIN chunk');
      seenBIN=true;
    }
    // Unknown types are ignored only after their complete bounds/alignment check.
    offset+=8+length;
  }
  if(!seenJSON) throw new Error('GLB 缺少 JSON chunk');
  for (const resource of [...(doc.buffers??[]),...(doc.images??[])]) {
    if (resource.uri && !resource.uri.startsWith('data:')) throw new Error('为避免意外外部请求，仅接受资源内嵌的 GLB');
  }
  return doc;
}
