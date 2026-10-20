import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from './vendor/three/build/three.module.js';
import {buildBindings, makeHighlight} from './semantic-three.mjs';
import {parseAnnotations} from './semantic-core.mjs';

const annotations = parseAnnotations({parts:[
  {label_id:1,color:[255,0,0],label_en:'red'},
  {label_id:2,color:[0,255,0],label_en:'green'},
  {label_id:3,color:[128,0,0],label_en:'dark red'},
]});

function triangle() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([0,0,0,1,0,0,0,1,0],3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute([0.25,0.25,0.25,0.25,0.25,0.25],2));
  return geometry;
}

test('assigns a surface label from real material color without changing geometry or material', async () => {
  const geometry=triangle();
  const material=new THREE.MeshBasicMaterial({color:0xff0000});
  const mesh=new THREE.Mesh(geometry,material);
  const result=await buildBindings(mesh,annotations);
  assert.deepEqual([...result.entries[0].labels],[1]);
  assert.equal(result.stats.coverage,1);
  assert.equal(result.stats.byLabel[1],1);
  assert.equal(mesh.geometry,geometry);
  assert.equal(mesh.material,material);
});

function texture() {
  // Top row red, bottom row green. The adapter must use Texture.transformUv.
  const map=new THREE.DataTexture(new Uint8Array([255,0,0,255,0,255,0,255]),1,2,THREE.RGBAFormat);
  map.colorSpace=THREE.SRGBColorSpace;
  return map;
}

test('reads the texture image and applies flipY rather than matching the white material factor',async()=>{
  const map=texture();
  map.flipY=true;
  const mesh=new THREE.Mesh(triangle(),new THREE.MeshBasicMaterial({map}));
  assert.deepEqual([...(await buildBindings(mesh,annotations)).entries[0].labels],[2]);
});

test('honors selected UV channel and texture offset',async()=>{
  const map=texture();
  map.channel=1;
  map.offset.y=0.5;
  const geometry=triangle();
  geometry.setAttribute('uv1',new THREE.Float32BufferAttribute([0.25,0.25,0.25,0.25,0.25,0.25],2));
  const mesh=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({map}));
  assert.deepEqual([...(await buildBindings(mesh,annotations)).entries[0].labels],[2]);
});

test('converts linear vertex COLOR_0 to sRGB and multiplies the material factor',async()=>{
  const geometry=triangle();
  const half=new THREE.Color(0x800000).r;
  geometry.setAttribute('color',new THREE.Float32BufferAttribute([half*2,0,0,half*2,0,0,half*2,0,0],3));
  const material=new THREE.MeshBasicMaterial({vertexColors:true});
  material.color.setRGB(0.5,1,1);
  const mesh=new THREE.Mesh(geometry,material);
  assert.deepEqual([...(await buildBindings(mesh,annotations)).entries[0].labels],[3]);
});

test('multiplies sRGB texture in linear space before returning to the palette color space',async()=>{
  const map=texture();
  const material=new THREE.MeshBasicMaterial({map});
  material.color.setRGB(new THREE.Color(0x800000).r,1,1);
  const mesh=new THREE.Mesh(triangle(),material);
  assert.deepEqual([...(await buildBindings(mesh,annotations)).entries[0].labels],[3]);
});

test('leaves faces crossing two palette labels unbound instead of majority-voting',async()=>{
  const geometry=triangle();
  geometry.setAttribute('uv',new THREE.Float32BufferAttribute([0.25,0.05,0.25,0.95,0.25,0.95],2));
  const mesh=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({map:texture()}));
  const result=await buildBindings(mesh,annotations);
  assert.deepEqual([...result.entries[0].labels],[0]);
  assert.equal(result.stats.byReason.mixed_labels,1);
});

test('handles indexed geometry with material groups and separate primitive meshes',async()=>{
  const geometry=triangle();
  geometry.setIndex([0,1,2,0,2,1]);
  geometry.addGroup(0,3,0);
  geometry.addGroup(3,3,1);
  const mesh=new THREE.Mesh(geometry,[new THREE.MeshBasicMaterial({color:0xff0000}),new THREE.MeshBasicMaterial({color:0x00ff00})]);
  const root=new THREE.Group();
  root.add(mesh,new THREE.Mesh(triangle(),new THREE.MeshBasicMaterial({color:0xff0000})));
  const result=await buildBindings(root,annotations);
  assert.deepEqual(result.entries.map(entry=>[...entry.labels]),[[1,2],[1]]);
  assert.equal(result.stats.totalFaces,3);
});

test('marks unsupported skinned and morphing meshes unknown',async()=>{
  const skinned=new THREE.SkinnedMesh(triangle(),new THREE.MeshBasicMaterial({color:0xff0000}));
  const morphGeometry=triangle();
  morphGeometry.morphAttributes.position=[morphGeometry.attributes.position.clone()];
  const morph=new THREE.Mesh(morphGeometry,new THREE.MeshBasicMaterial({color:0xff0000}));
  const root=new THREE.Group();root.add(skinned,morph);
  const result=await buildBindings(root,annotations);
  assert.deepEqual(result.entries.map(entry=>[...entry.labels]),[[0],[0]]);
  assert.equal(result.stats.byReason.unsupported_deformation,2);
});

test('does not invent a label for an unreadable image or missing UVs',async()=>{
  const geometry=triangle();geometry.deleteAttribute('uv');
  const root=new THREE.Group();
  root.add(new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({map:texture()})));
  root.add(new THREE.Mesh(triangle(),new THREE.MeshBasicMaterial({map:new THREE.Texture({width:1,height:1})})));
  const result=await buildBindings(root,annotations);
  assert.deepEqual(result.entries.map(entry=>[...entry.labels]),[[0],[0]]);
  assert.equal(result.stats.byReason.missing_uv,1);
  assert.equal(result.stats.byReason.unreadable_texture,1);
});

test('rejects a configured face cap before allocating face labels',async()=>{
  const mesh=new THREE.Mesh(triangle(),new THREE.MeshBasicMaterial({color:0xff0000}));
  await assert.rejects(()=>buildBindings(mesh,annotations,{maxFaces:0}),/face.*limit/i);
});

test('does not bind alpha-map-discarded surfaces whose visibility is unsupported',async()=>{
  const material=new THREE.MeshBasicMaterial({color:0xff0000,alphaMap:texture()});
  const mesh=new THREE.Mesh(triangle(),material);
  const result=await buildBindings(mesh,annotations);
  assert.deepEqual([...result.entries[0].labels],[0]);
  assert.equal(result.stats.byReason.unsupported_alpha_map,1);
});

test('leaves alpha-tested surfaces unknown instead of labeling discarded texels',async()=>{
  const map=new THREE.DataTexture(new Uint8Array([199,229,22,64]),1,1,THREE.RGBAFormat);
  map.colorSpace=THREE.SRGBColorSpace;
  const material=new THREE.MeshBasicMaterial({map,alphaTest:.5});
  const mesh=new THREE.Mesh(triangle(),material);
  const wheelAnnotations=parseAnnotations({parts:[{label_id:4,color:[199,229,22]}]});
  const result=await buildBindings(mesh,wheelAnnotations);
  assert.deepEqual([...result.entries[0].labels],[0]);
  assert.equal(result.stats.byReason.unsupported_alpha_test,1);
});

test('builds an independent highlight for only selected triangles without changing the original',async()=>{
  const geometry=triangle();geometry.setIndex([0,1,2,0,2,1]);
  geometry.addGroup(0,3,0);geometry.addGroup(3,3,1);
  const mesh=new THREE.Mesh(geometry,[new THREE.MeshBasicMaterial({color:0xff0000}),new THREE.MeshBasicMaterial({color:0x00ff00})]);
  const entry=(await buildBindings(mesh,annotations)).entries[0];
  const overlay=makeHighlight(entry,2);
  assert.ok(overlay?.isMesh);
  assert.deepEqual([...overlay.geometry.index.array],[0,2,1]);
  assert.notEqual(overlay.geometry,geometry);
  assert.notEqual(overlay.geometry.attributes.position,geometry.attributes.position);
  assert.equal(overlay.material.polygonOffset,true);
  const hits=[];overlay.raycast({},hits);assert.deepEqual(hits,[]);
  assert.equal(makeHighlight(entry,999),null);
  assert.deepEqual([...geometry.index.array],[0,1,2,0,2,1]);
});
