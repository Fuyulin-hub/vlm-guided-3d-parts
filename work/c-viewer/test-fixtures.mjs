// Synthetic two-panel GLB, ONLY for software tests; never experimental evidence.
import {createHash} from 'node:crypto';
export function syntheticBundle({unknown=false}={}) {
  const pos = new Float32Array([-1,-.7,0, -.1,-.7,0, -.1,.7,0, -1,.7,0, .1,-.7,0, 1,-.7,0, 1,.7,0, .1,.7,0]);
  const rgb = new Float32Array(Array.from({length:8},(_,i)=>unknown?[0,0,1]:i<4?[1,0,0]:[0,1,0]).flat());
  const idx = new Uint16Array([0,1,2,0,2,3,4,5,6,4,6,7]);
  const bin = Buffer.concat([Buffer.from(pos.buffer),Buffer.from(rgb.buffer),Buffer.from(idx.buffer)]);
  const doc={asset:{version:'2.0',generator:'PartScope synthetic test fixture'},scene:0,scenes:[{nodes:[0]}],nodes:[{mesh:0}],meshes:[{primitives:[{attributes:{POSITION:0,COLOR_0:1},indices:2,material:0}]}],materials:[{doubleSided:true,pbrMetallicRoughness:{baseColorFactor:[1,1,1,1],metallicFactor:0,roughnessFactor:1}}],buffers:[{byteLength:bin.length}],bufferViews:[{buffer:0,byteOffset:0,byteLength:pos.byteLength},{buffer:0,byteOffset:pos.byteLength,byteLength:rgb.byteLength},{buffer:0,byteOffset:pos.byteLength+rgb.byteLength,byteLength:idx.byteLength}],accessors:[{bufferView:0,componentType:5126,count:8,type:'VEC3',min:[-1,-.7,0],max:[1,.7,0]},{bufferView:1,componentType:5126,count:8,type:'VEC3'},{bufferView:2,componentType:5123,count:12,type:'SCALAR'}]};
  const j=Buffer.from(JSON.stringify(doc));const jp=Buffer.alloc(Math.ceil(j.length/4)*4,32);j.copy(jp);
  const glb=Buffer.alloc(12+8+jp.length+8+bin.length);glb.write('glTF');glb.writeUInt32LE(2,4);glb.writeUInt32LE(glb.length,8);glb.writeUInt32LE(jp.length,12);glb.writeUInt32LE(0x4e4f534a,16);jp.copy(glb,20);const k=20+jp.length;glb.writeUInt32LE(bin.length,k);glb.writeUInt32LE(0x004e4942,k+4);bin.copy(glb,k+8);
  const annotation={uid:'synthetic-test-only',run_id:'software-test',final_map:{sha256:'a'.repeat(64)},parts:[{part_id:'part_01',label_id:7,label_zh:'左面板',label_en:'Left panel',color:[255,0,0]},{part_id:'part_02',label_id:19,label_zh:'右面板',label_en:'Right panel',color:[0,255,0]}]};
  const meta={output_glb_sha256:createHash('sha256').update(glb).digest('hex'),guidance_sha256:annotation.final_map.sha256};
  return [{name:'fixture.glb',mimeType:'model/gltf-binary',buffer:glb},{name:'annotations.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(annotation))},{name:'segmented.glb.meta.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(meta))}];
}
