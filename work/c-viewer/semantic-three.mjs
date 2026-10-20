import * as THREE from './vendor/three/build/three.module.js';
import {decideFaceLabel, sampleRGBA} from './semantic-core.mjs';

// Seven interior samples: center, near each vertex, and near each edge.
const BARYCENTRIC = [
  [1/3,1/3,1/3], [.8,.1,.1], [.1,.8,.1], [.1,.1,.8],
  [.45,.45,.1], [.45,.1,.45], [.1,.45,.45],
];
const faceCount = geometry => Math.floor((geometry?.index?.count ?? geometry?.attributes?.position?.count ?? 0)/3);
const unknown = reason => ({labelId:null,reason});
const increment = (counts,key) => { counts[key]=(counts[key]??0)+1; };
const readable = attribute => attribute && !attribute.isGLBufferAttribute && typeof attribute.getX==='function';

function materialAt(mesh,face) {
  if(!Array.isArray(mesh.material)) return mesh.material;
  const offset=face*3;
  const groups=mesh.geometry.groups.filter(group=>offset>=group.start && offset+3<=group.start+group.count);
  return groups.length===1 ? mesh.material[groups[0].materialIndex] : null;
}

function imagePixels(texture) {
  if(texture.isCompressedTexture || texture.isVideoTexture || texture.mapping!==THREE.UVMapping ||
     ![THREE.SRGBColorSpace,THREE.LinearSRGBColorSpace,THREE.NoColorSpace].includes(texture.colorSpace)) return null;
  const image=texture.image;
  if(!image) return null;
  if(texture.isDataTexture) {
    if(texture.type!==THREE.UnsignedByteType || texture.format!==THREE.RGBAFormat ||
       !(image.data instanceof Uint8Array || image.data instanceof Uint8ClampedArray) ||
       image.data.length<image.width*image.height*4) return null;
    return image;
  }
  const width=image.naturalWidth ?? image.width;
  const height=image.naturalHeight ?? image.height;
  if(!(width>0 && height>0)) return null;
  try {
    const canvas=typeof OffscreenCanvas!=='undefined' ? new OffscreenCanvas(width,height) :
      typeof document!=='undefined' ? document.createElement('canvas') : null;
    if(!canvas) return null;
    canvas.width=width;canvas.height=height;
    const context=canvas.getContext('2d',{willReadFrequently:true});
    if(!context) return null;
    context.drawImage(image,0,0,width,height);
    return context.getImageData(0,0,width,height);
  } catch { return null; } // Tainted, detached, or unsupported image: never guess.
}

function interpolate(attribute,indices,weights,component) {
  const getter=['getX','getY','getZ','getW'][component];
  return weights.reduce((sum,weight,i)=>sum+weight*attribute[getter](indices[i]),0);
}

function meshReason(mesh) {
  if(mesh.isSkinnedMesh || Object.values(mesh.geometry.morphAttributes??{}).some(values=>values.length)) return 'unsupported_deformation';
  if(mesh.isInstancedMesh) return 'unsupported_instancing';
  if(!readable(mesh.geometry.attributes.position) || (mesh.geometry.index&&!readable(mesh.geometry.index))) return 'unreadable_geometry';
  return null;
}

function faceDecision(mesh,face,palette,options,textureCache,scratch) {
  const geometry=mesh.geometry;
  const offset=face*3;
  const range=geometry.drawRange;
  if(offset<range.start || offset+3>range.start+range.count) return unknown('outside_draw_range');
  const material=materialAt(mesh,face);
  if(!material || !material.color || material.isShaderMaterial) return unknown('unsupported_material');
  if(material.alphaMap) return unknown('unsupported_alpha_map');
  if(material.alphaTest>0) return unknown('unsupported_alpha_test');
  if(material.visible===false || material.opacity<=10/255) return unknown('invisible_material');
  const indices=[0,1,2].map(i=>geometry.index ? geometry.index.getX(offset+i) : offset+i);
  if(indices.some(index=>!Number.isInteger(index)||index<0||index>=geometry.attributes.position.count)) return unknown('invalid_indices');
  const vertexColors=material.vertexColors ? geometry.attributes.color : null;
  if(material.vertexColors && (!readable(vertexColors) || vertexColors.itemSize<3 || indices.some(i=>i>=vertexColors.count))) return unknown('missing_vertex_color');
  const map=material.map;
  let uvAttribute=null;
  let pixels=null;
  if(map) {
    uvAttribute=geometry.attributes[map.channel===0?'uv':`uv${map.channel}`];
    if(!readable(uvAttribute) || uvAttribute.itemSize<2 || indices.some(i=>i>=uvAttribute.count)) return unknown('missing_uv');
    if(!textureCache.has(map)) {
      if(map.matrixAutoUpdate) map.updateMatrix();
      textureCache.set(map,imagePixels(map));
    }
    pixels=textureCache.get(map);
    if(!pixels) return unknown('unreadable_texture');
  }
  const samples=BARYCENTRIC.map(weights=>{
    const color=scratch.color.copy(material.color);
    if(map) {
      const uv=scratch.uv.set(interpolate(uvAttribute,indices,weights,0),interpolate(uvAttribute,indices,weights,1));
      map.transformUv(uv);
      const rgb=sampleRGBA(pixels,uv.x,uv.y);
      if(!rgb) return null;
      const texel=scratch.texel.setRGB(rgb[0]/255,rgb[1]/255,rgb[2]/255);
      if(map.colorSpace===THREE.SRGBColorSpace) texel.convertSRGBToLinear();
      color.multiply(texel);
    }
    if(vertexColors) {
      color.r*=interpolate(vertexColors,indices,weights,0);
      color.g*=interpolate(vertexColors,indices,weights,1);
      color.b*=interpolate(vertexColors,indices,weights,2);
      if(vertexColors.itemSize>=4 && interpolate(vertexColors,indices,weights,3)*material.opacity<=10/255) return null;
    }
    color.convertLinearToSRGB();
    return [color.r*255,color.g*255,color.b*255];
  });
  return decideFaceLabel(samples,palette,options);
}

/** Bind a conservative face-level label map without modifying the loaded model. */
export async function buildBindings(root,annotations,options={},onProgress) {
  if(!annotations?.palette?.length) throw new Error('A validated annotation palette is required');
  if(annotations.palette.some(entry=>entry.labelId>2147483647)) throw new Error('label_id exceeds the Int32 face-label range');
  const meshes=[];
  root.traverse(mesh=>{if(mesh.isMesh && !mesh.userData.semanticHighlight) meshes.push(mesh);});
  const totalFaces=meshes.reduce((sum,mesh)=>sum+faceCount(mesh.geometry),0);
  const maxFaces=options.maxFaces??2_000_000;
  if(totalFaces>maxFaces) throw new Error(`Face count ${totalFaces} exceeds limit ${maxFaces}; use a smaller model`);
  const entries=[];
  const stats={totalFaces,matchedFaces:0,unmatchedFaces:0,coverage:0,byLabel:{},byReason:{}};
  const textureCache=new Map();
  const scratch={color:new THREE.Color(),texel:new THREE.Color(),uv:new THREE.Vector2()};
  let processedFaces=0;
  const progress=()=>onProgress?.({processedFaces,totalFaces,progress:totalFaces?processedFaces/totalFaces:1});
  progress();
  for(const mesh of meshes) {
    const labels=new Int32Array(faceCount(mesh.geometry));
    const reasonCounts={};
    const unsupported=meshReason(mesh);
    for(let face=0;face<labels.length;face++) {
      if(options.signal?.aborted) throw new DOMException('Binding cancelled','AbortError');
      const decision=unsupported ? unknown(unsupported) : faceDecision(mesh,face,annotations.palette,options,textureCache,scratch);
      labels[face]=decision.labelId??0;
      if(labels[face]) {stats.matchedFaces++;increment(stats.byLabel,labels[face]);}
      else stats.unmatchedFaces++;
      increment(reasonCounts,decision.reason);increment(stats.byReason,decision.reason);
      processedFaces++;
      if(processedFaces%2000===0) {
        progress();
        await new Promise(resolve=>setTimeout(resolve,0));
      }
    }
    entries.push({mesh,labels,reasonCounts});
  }
  stats.coverage=totalFaces?stats.matchedFaces/totalFaces:0;
  progress();
  return {entries,stats};
}

/** Add the returned mesh as a child of entry.mesh; caller owns disposal. */
export function makeHighlight(entry,labelId) {
  if(!entry || !Number.isInteger(labelId) || labelId<=0) return null;
  const source=entry.mesh.geometry;
  const selected=[];
  for(let face=0;face<entry.labels.length;face++) {
    if(entry.labels[face]!==labelId) continue;
    for(let corner=0;corner<3;corner++) selected.push(source.index ? source.index.getX(face*3+corner) : face*3+corner);
  }
  if(!selected.length) return null;
  const geometry=new THREE.BufferGeometry();
  // Own the position buffer so disposing/replacing an overlay cannot evict a
  // BufferAttribute still used by the original mesh in the renderer.
  geometry.setAttribute('position',source.attributes.position.clone());
  geometry.setIndex(selected);
  const material=new THREE.MeshBasicMaterial({color:0xff9b3d,transparent:true,opacity:.68,
    side:THREE.DoubleSide,depthWrite:false,polygonOffset:true,polygonOffsetFactor:-2,
    polygonOffsetUnits:-2,toneMapped:false});
  const overlay=new THREE.Mesh(geometry,material);
  overlay.name=`semantic-highlight-${labelId}`;
  overlay.userData.semanticHighlight=true;
  overlay.renderOrder=10;
  overlay.raycast=()=>{};
  return overlay;
}
