import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {parseAnnotations} from './semantic-core.mjs';
import {buildBindings,makeHighlight} from './semantic-three.mjs';
import {verifyPair,inspectGLB} from './provenance.mjs';

const $=id=>document.getElementById(id);
const options={maxDistance:24,minMargin:8};
// UX warning threshold only; neither a semantic-accuracy nor a research gate.
const LOW_COVERAGE_WARNING=.1;
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(38,1,.01,1000);
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setClearColor(0x111617,0);
$('viewer').appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xffffff,0x535966,2.2));
const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(3,5,4);scene.add(light);
const raycaster=new THREE.Raycaster();const pointer=new THREE.Vector2();
let root=null, binding=null, parsed=null, annotation=null, provenance=null, glbHash=null, annHash=null, selected=null, generation=0, wire=false;
let overlays=[];let meshes=[];let sourceName='';let activeLoad=null;

function status(text,error=false){$('loadStatus').textContent=text;$('loadStatus').classList.toggle('error',error);}
function clearHighlights(){for(const overlay of overlays){overlay.removeFromParent();overlay.geometry.dispose();overlay.material.dispose();}overlays=[];}
function clearSelection(message='No part selected'){
  selected=null;clearHighlights();$('partId').textContent='—';$('partEn').textContent=message;$('confidence').textContent='—';$('maskScore').textContent='—';$('faceCount').textContent='—';$('jsonRecord').textContent='';$('linked').textContent='未绑定';
  for(const element of document.querySelectorAll('.component'))element.classList.remove('active');
}
function clearBindings(){binding=null;parsed=null;clearSelection();$('componentList').replaceChildren();$('bindingSummary').textContent='等待同版本 GLB 和二维 annotations.json';$('exportBinding').disabled=true;$('confirmPair').hidden=true;$('hoverLabel').style.display='none';}
function disposeScene(model){if(!model)return;model.removeFromParent();const materials=new Set(),textures=new Set(),geometries=new Set();model.traverse(obj=>{if(!obj.isMesh)return;geometries.add(obj.geometry);for(const material of(Array.isArray(obj.material)?obj.material:[obj.material])){materials.add(material);for(const v of Object.values(material))if(v?.isTexture)textures.add(v);}});for(const g of geometries)g.dispose();for(const m of materials)m.dispose();for(const t of textures)t.dispose();}
function disposeModel(){clearHighlights();disposeScene(root);root=null;meshes=[];}
function isCurrent(request){return request?.token===generation&&!request.controller.signal.aborted;}
function assertCurrent(request){if(!isCurrent(request))throw new DOMException('旧文件加载已取消','AbortError');}
function beginLoad(){
  activeLoad?.controller.abort();
  const request={token:++generation,controller:new AbortController()};activeLoad=request;
  clearBindings();disposeModel();annotation=null;provenance=null;glbHash=null;annHash=null;sourceName='';
  $('objectId').textContent='OBJECT / 正在加载';wire=false;$('toggleWire').textContent='线框';
  return request;
}
function resize(){const r=$('viewer').getBoundingClientRect();renderer.setSize(r.width,r.height,false);camera.aspect=r.width/Math.max(r.height,1);camera.updateProjectionMatrix();}
new ResizeObserver(resize).observe($('viewer'));resize();
function fit(front=false){if(!root)return;const bounds=new THREE.Box3().setFromObject(root);const center=bounds.getCenter(new THREE.Vector3());const radius=Math.max(bounds.getSize(new THREE.Vector3()).length()/2,.001);const vFov=THREE.MathUtils.degToRad(camera.fov);const hFov=2*Math.atan(Math.tan(vFov/2)*camera.aspect);const distance=radius/Math.sin(Math.min(vFov,hFov)/2)*1.18;const direction=front?new THREE.Vector3(0,0,1):new THREE.Vector3(1,.65,1.4).normalize();camera.position.copy(center).addScaledVector(direction,distance);camera.near=Math.max(radius/1000,.0001);camera.far=distance+radius*100;camera.updateProjectionMatrix();controls.target.copy(center);controls.minDistance=radius*.1;controls.maxDistance=radius*40;controls.update();}
function record(label){const part=parsed.parts.get(label);return {...part,binding:{label_id:label,face_count:binding.stats.byLabel[label]??0,method:'palette-conservative-face-v1',provenance:provenance.status}};}
function select(label){if(!binding||!parsed.parts.has(label))return;const count=binding.stats.byLabel[label]??0;if(!count){clearSelection('该部件尚无可靠三维对应');return;}selected=label;clearHighlights();for(const entry of binding.entries){const overlay=makeHighlight(entry,label);if(overlay){entry.mesh.add(overlay);overlays.push(overlay);}}
  const p=parsed.parts.get(label);$('partId').textContent=`LABEL ${label} · ${p.part_id??'无 part_id'}`;$('partEn').textContent=p.label||`Part ${label}`;$('faceCount').textContent=count.toLocaleString();$('confidence').textContent=provenance.status==='verified'?'哈希通过':'人工确认';$('maskScore').textContent=`${(binding.stats.coverage*100).toFixed(2)}%`;$('linked').textContent='按颜色关联';$('jsonRecord').textContent=JSON.stringify(record(label),null,2);for(const element of document.querySelectorAll('.component'))element.classList.toggle('active',Number(element.dataset.labelId)===label);
}
function renderList(){const list=$('componentList');list.replaceChildren();for(const [label,p] of parsed.parts){const count=binding.stats.byLabel[label]??0;const button=document.createElement('button');button.className='component';button.dataset.labelId=String(label);button.disabled=!count;const dot=document.createElement('span');dot.style.backgroundColor=`rgb(${p.color.join(',')})`;const main=document.createElement('span');main.className='component-main';main.append(document.createTextNode(p.label||`Part ${label}`));const sub=document.createElement('span');sub.className='component-sub';sub.textContent=`label ${label}`;main.append(sub);const score=document.createElement('span');score.className='component-score';score.textContent=`${count.toLocaleString()} 面`;button.append(dot,main,score);button.addEventListener('click',()=>select(label));list.append(button);}}
async function bind(request){
  if(!isCurrent(request))return;
  const model=root,records=parsed;
  status('正在建立三维标签…');$('stageMessage').textContent='按原始纹理 / 顶点颜色匹配；边界歧义保留未匹配';
  const result=await buildBindings(model,records,{...options,signal:request.controller.signal},p=>{if(isCurrent(request))$('bindingSummary').textContent=`正在分析 ${p.processedFaces.toLocaleString()} / ${p.totalFaces.toLocaleString()} 面`;});
  if(!isCurrent(request))return;binding=result;renderList();$('exportBinding').disabled=false;
  const lowCoverage=result.stats.coverage<LOW_COVERAGE_WARNING;
  $('bindingSummary').textContent=`颜色匹配 ${(result.stats.coverage*100).toFixed(2)}% · ${result.stats.matchedFaces.toLocaleString()} / ${result.stats.totalFaces.toLocaleString()} 面 · 未匹配 ${result.stats.unmatchedFaces.toLocaleString()} 面。匹配率不是分割准确率。${lowCoverage?' 低于 10% 工程提醒阈值；请核查三维颜色是否保留二维调色板。此阈值不是准确率门槛。':''}`;
  if(lowCoverage){
    $('stageMessage').textContent='匹配不足，不能据此宣称完整语义绑定；可导出候选映射与未匹配区域进行诊断';
    status('绑定扫描完成 · 匹配不足',true);clearSelection('颜色匹配不足，请先核查数据');
  }else{
    $('stageMessage').textContent='点击有匹配标签的表面，查看部件名称';status('绑定完成');
    const first=[...parsed.parts.keys()].find(id=>result.stats.byLabel[id]>0);if(first!==undefined)select(first);else clearSelection('无匹配区域');
  }
}
async function hash(buffer){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',buffer))].map(x=>x.toString(16).padStart(2,'0')).join('');}
async function ingest(files,request=beginLoad()){
  if(!isCurrent(request))return;
  status('正在读取文件…');
  try{
    const glbs=files.filter(f=>/\.glb$/i.test(f.name));if(glbs.length!==1)throw new Error('请一次选择一个 GLB，以及配套二维 annotations.json 和可选 .meta.json');
    // Keep every identity field private until this load is ready to commit.
    let meta=null,nextAnnotation=null,nextAnnHash=null,nextParsed=null,nextProvenance=null;
    for(const file of files.filter(f=>/\.json$/i.test(f.name))){
      const bytes=await file.arrayBuffer();assertCurrent(request);
      const doc=JSON.parse(new TextDecoder().decode(bytes));
      if(Array.isArray(doc.parts)){
        if(nextAnnotation)throw new Error('请选择一份二维标注 JSON');
        nextAnnotation=doc;nextAnnHash=await hash(bytes);assertCurrent(request);
      }else if(doc.output_glb_sha256||doc.glb_sha256||doc.guidance_sha256){
        if(meta)throw new Error('请选择一份来源记录 JSON，不要同时选择旧 meta 与新 provenance');meta=doc;
      }
    }
    const bytes=await glbs[0].arrayBuffer();assertCurrent(request);inspectGLB(bytes);
    const nextGlbHash=await hash(bytes);assertCurrent(request);
    const nextSourceName=glbs[0].name;
    if(nextAnnotation){nextParsed=parseAnnotations(nextAnnotation);nextProvenance=verifyPair(nextGlbHash,nextAnnotation,meta,nextAnnHash);}
    const gltf=await new GLTFLoader().parseAsync(bytes,'');
    if(!isCurrent(request)){disposeScene(gltf.scene);return;}
    annotation=nextAnnotation;annHash=nextAnnHash;glbHash=nextGlbHash;sourceName=nextSourceName;parsed=nextParsed;provenance=nextProvenance;
    root=gltf.scene;scene.add(root);root.updateMatrixWorld(true);meshes=[];root.traverse(mesh=>{if(mesh.isMesh){mesh.userData.gltfAssociation=gltf.parser.associations.get(mesh)??null;meshes.push(mesh);}});fit();$('objectId').textContent=parsed?.uid??sourceName;$('stageMessage').textContent=`${sourceName} 已加载`;
    if(!annotation){status('模型已加载，缺少标注');$('bindingSummary').textContent='请重新同时选择此 GLB 与对应二维 annotations.json';return;}
    if(provenance.status==='unverified'){status('等待同源确认');$('confirmPair').hidden=false;$('pairConfirmed').checked=false;$('bindingSummary').textContent=provenance.message;return;}
    await bind(request);
  }catch(error){if(!isCurrent(request))return;clearBindings();status('加载 / 绑定失败',true);$('stageMessage').textContent=error.message;$('bindingSummary').textContent=error.message;}
}

function hitAt(event){if(!root)return null;const rect=renderer.domElement.getBoundingClientRect();pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);camera.updateMatrixWorld(true);raycaster.setFromCamera(pointer,camera);const hit=raycaster.intersectObjects(meshes,false)[0];if(!hit)return null;const entry=binding?.entries.find(e=>e.mesh===hit.object);return {hit,label:entry?.labels[hit.faceIndex]??0};}
let down=null,lastHover=0;
renderer.domElement.addEventListener('pointerdown',event=>{down={x:event.clientX,y:event.clientY};});
renderer.domElement.addEventListener('pointerup',event=>{if(!down||Math.hypot(event.clientX-down.x,event.clientY-down.y)>6)return;down=null;const pick=hitAt(event);if(pick?.label)select(pick.label);else if(pick)clearSelection('未匹配区域');});
renderer.domElement.addEventListener('pointermove',event=>{if(performance.now()-lastHover<80)return;lastHover=performance.now();const pick=hitAt(event),label=$('hoverLabel');if(!pick||!binding){label.style.display='none';return;}const p=parsed.parts.get(pick.label);label.textContent=p?(p.label||`Part ${pick.label}`):'未匹配：边界或颜色不确定';const r=$('viewer').getBoundingClientRect();label.style.left=Math.min(event.clientX-r.left,r.width-230)+'px';label.style.top=event.clientY-r.top+'px';label.style.display='block';});
renderer.domElement.addEventListener('pointerleave',()=>{$('hoverLabel').style.display='none';});
$('loadButton').onclick=()=>$('fileInput').click();
$('fileInput').onchange=event=>{const files=[...event.target.files];event.target.value='';if(files.length)ingest(files);};
$('sampleButton').onclick=async()=>{const request=beginLoad();try{status('读取消防车样本…');const names=['segmented.glb','annotations.json','provenance.json'];const files=await Promise.all(names.map(async name=>{const response=await fetch(`./assets/firetruck/${name}`,{signal:request.controller.signal});assertCurrent(request);if(!response.ok)throw new Error(`样本文件不可用：${name}`);const bytes=await response.arrayBuffer();assertCurrent(request);return new File([bytes],name);}));await ingest(files,request);}catch(e){if(!isCurrent(request))return;status('样本读取失败',true);$('stageMessage').textContent=e.message;}};
$('bindConfirmed').onclick=()=>{const request=activeLoad;if(!$('pairConfirmed').checked||!isCurrent(request))return;$('confirmPair').hidden=true;provenance={status:'user-confirmed',message:'用户确认同源，缺少完整哈希证据'};bind(request).catch(e=>{if(!isCurrent(request))return;clearBindings();status('绑定失败',true);$('stageMessage').textContent=e.message;});};
$('resetView').onclick=()=>fit();$('resetTop').onclick=()=>fit();$('frontView').onclick=()=>fit(true);
$('toggleWire').onclick=()=>{wire=!wire;for(const mesh of meshes)for(const m of(Array.isArray(mesh.material)?mesh.material:[mesh.material]))m.wireframe=wire;$('toggleWire').textContent=wire?'实体':'线框';};
function download(name,data){const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('copyJson').onclick=async()=>{if(selected===null)return;try{await navigator.clipboard.writeText(JSON.stringify(record(selected),null,2));$('copyJson').textContent='已复制';setTimeout(()=>$('copyJson').textContent='复制 JSON',1200);}catch{$('stageMessage').textContent='剪贴板不可用，请导出记录';}};
$('exportButton').onclick=()=>{if(selected!==null)download(`label_${selected}.json`,record(selected));};
$('exportBinding').onclick=()=>{if(!binding)return;download(`${parsed.uid??'model'}_face_labels.json`,{schema:'partscope.face-labels.v1',source:{glb_file:sourceName,glb_sha256:glbHash,annotation_sha256:annHash,uid:parsed.uid,run_id:parsed.runId,guidance_sha256:annotation.final_map?.sha256??null,verification:provenance},method:{name:'palette-conservative-face-v1',three_version:THREE.REVISION,...options,min_known_fraction:0.8,unknown_label_id:0,sampling_approximate:true,low_coverage_warning_threshold:LOW_COVERAGE_WARNING,note:'Approximate color correspondence from seven interior samples per triangle, not semantic accuracy or exact boundary recovery. Unresolved faces remain unknown. The low-coverage threshold is an engineering warning, not an accuracy gate.'},stats:binding.stats,parts:[...parsed.parts.values()],meshes:binding.entries.map((e,index)=>({mesh_index:index,mesh_name:e.mesh.name,gltf_association:e.mesh.userData.gltfAssociation,vertex_count:e.mesh.geometry.getAttribute('position')?.count??0,face_count:e.labels.length,face_label_ids:Array.from(e.labels),reason_counts:e.reasonCounts}))});};
status('等待模型和标注');$('loadButton').disabled=false;$('sampleButton').disabled=false;
function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);}animate();
