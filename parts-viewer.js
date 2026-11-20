import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export function createViewer(container, {onSelect=()=>{}, onHover=()=>{}}={}) {
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  renderer.outputColorSpace=THREE.SRGBColorSpace;
  renderer.setClearColor(0xffffff,0);
  const canvas=renderer.domElement;
  container.appendChild(canvas);
  canvas.setAttribute('role','img');
  canvas.setAttribute('aria-label','Interactive 3D parts. Click a part to inspect it. Drag to rotate, scroll or pinch to zoom. Use the part list for keyboard selection.');
  canvas.tabIndex=0;
  const scene=new THREE.Scene();
  const camera=new THREE.PerspectiveCamera(36,1,.01,100);
  const controls=new OrbitControls(camera,canvas);
  controls.enableDamping=true;controls.dampingFactor=.08;
  controls.autoRotateSpeed=1.4;controls.minDistance=.8;controls.maxDistance=8;
  controls.enablePan=false;controls.listenToKeyEvents(canvas);
  scene.add(new THREE.HemisphereLight(0xffffff,0x9aa79e,2.2));
  for(const [position,intensity] of [[[3,5,4],2.1],[[-3,1,-3],.7]]){
    const light=new THREE.DirectionalLight(0xffffff,intensity);
    light.position.fromArray(position);scene.add(light);
  }
  const loader=new GLTFLoader(),raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2();
  let current,generation=0,paused=false,parts=[],meshes=[],selected=null,isolated=false,gesture=null;

  function dispose(object){
    const geometries=new Set(),materials=new Set(),textures=new Set();
    object?.traverse(node=>{
      if(!node.isMesh)return;
      geometries.add(node.geometry);
      for(const material of Array.isArray(node.material)?node.material:[node.material]){
        materials.add(material);
        for(const value of Object.values(material))if(value?.isTexture)textures.add(value);
      }
    });
    for(const item of [...geometries,...materials,...textures])item.dispose();
  }
  function reset(){
    controls.autoRotate=false;
    camera.position.set(0,0,Math.max(3.7,3.7/camera.aspect));
    controls.target.set(0,0,0);controls.update();
  }
  function resize(){
    const width=container.clientWidth,height=container.clientHeight;
    if(!width||!height)return;
    camera.aspect=width/height;camera.updateProjectionMatrix();renderer.setSize(width,height,false);
  }
  new ResizeObserver(resize).observe(container);
  resize();reset();
  renderer.setAnimationLoop(()=>{if(!paused){controls.update();renderer.render(scene,camera);}});
  function applySelection(){
    for(const mesh of meshes){
      const active=selected===null||mesh.userData.partId===selected;
      mesh.visible=!isolated||active;
      mesh.material.opacity=active?1:.14;
      mesh.material.transparent=!active;mesh.material.depthWrite=active;
      mesh.material.needsUpdate=true;mesh.renderOrder=active?0:1;
    }
  }
  function selectPart(id){
    const part=parts.find(part=>part.id===id)??null;
    selected=part?.id??null;isolated=false;controls.autoRotate=false;
    applySelection();onSelect(part);
    container.dataset.selectedPart=part?String(part.id):'';
  }
  function pick(event){
    const rect=canvas.getBoundingClientRect();
    pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);
    raycaster.setFromCamera(pointer,camera);
    const hit=raycaster.intersectObjects(meshes.filter(mesh=>mesh.visible),false)[0];
    return hit?parts.find(part=>part.id===hit.object.userData.partId):null;
  }
  canvas.addEventListener('pointerdown',event=>{
    if(event.isPrimary&&event.button===0)gesture={id:event.pointerId,x:event.clientX,y:event.clientY,moved:false};
    else if(gesture)gesture.moved=true;
    onHover(null);
  });
  canvas.addEventListener('pointermove',event=>{
    if(gesture){if(Math.hypot(event.clientX-gesture.x,event.clientY-gesture.y)>6)gesture.moved=true;return;}
    if(event.pointerType==='touch')return;
    const part=pick(event);canvas.style.cursor=part?'pointer':'grab';
    onHover(part,{x:event.clientX,y:event.clientY});
  });
  canvas.addEventListener('pointerup',event=>{
    if(gesture?.id===event.pointerId){const clicked=!gesture.moved;gesture=null;if(clicked)selectPart(pick(event)?.id??null);}
  });
  canvas.addEventListener('pointercancel',()=>{gesture=null;onHover(null);});
  canvas.addEventListener('pointerleave',()=>onHover(null));
  canvas.addEventListener('keydown',event=>{if(event.key==='Escape')selectPart(null);});
  return {
    async load(url,rotation,details,faceLabels,progress){
      const id=++generation;
      const gltf=await loader.loadAsync(url,event=>progress(event.lengthComputable?Math.round(event.loaded/event.total*100):0));
      if(id!==generation){dispose(gltf.scene);return;}
      const object=gltf.scene,sourceMeshes=[];
      object.traverse(node=>{if(node.isMesh)sourceMeshes.push(node);});
      try{
        if(sourceMeshes.length!==1)throw new Error('Unsupported mesh layout');
        const original=sourceMeshes[0],geometry=original.geometry;
        const faceCount=(geometry.index?.count??geometry.attributes.position.count)/3;
        if(faceCount!==details.faceCount||faceLabels.length!==faceCount)throw new Error('Part binding does not match this mesh');
        const indices=new Map(details.parts.map(part=>[part.id,[]]));
        for(let face=0;face<faceCount;face++){
          const list=indices.get(faceLabels[face]);
          if(!list)throw new Error('Unknown region in part binding');
          for(let corner=0;corner<3;corner++)list.push(geometry.index?geometry.index.getX(face*3+corner):face*3+corner);
        }
        const nextMeshes=[];
        for(const part of details.parts){
          const list=indices.get(part.id);
          if(list.length!==part.faceCount*3)throw new Error('Part face count mismatch');
          const partGeometry=new THREE.BufferGeometry();
          for(const [name,attribute] of Object.entries(geometry.attributes))partGeometry.setAttribute(name,attribute);
          partGeometry.setIndex(list);partGeometry.computeBoundingSphere();
          const mesh=new THREE.Mesh(partGeometry,original.material.clone());
          mesh.position.copy(original.position);mesh.quaternion.copy(original.quaternion);mesh.scale.copy(original.scale);
          mesh.userData.partId=part.id;mesh.name=part.name;
          original.parent.add(mesh);nextMeshes.push(mesh);
        }
        original.visible=false;
        const group=new THREE.Group();group.add(object);
        if(rotation){const r=rotation;group.setRotationFromMatrix(new THREE.Matrix4().set(r[0][0],r[1][0],r[2][0],0,r[0][1],r[1][1],r[2][1],0,r[0][2],r[1][2],r[2][2],0,0,0,0,1));}
        const box=new THREE.Box3().setFromObject(group),size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3());
        const factor=2/Math.max(size.x,size.y,size.z);
        group.scale.setScalar(factor);group.position.copy(center).multiplyScalar(-factor);
        if(current){scene.remove(current);dispose(current);}
        current=group;parts=details.parts;meshes=nextMeshes;
        scene.add(current);scene.updateMatrixWorld(true);
        reset();selectPart(null);
      }catch(error){dispose(object);throw error;}
    },
    reset,selectPart,
    isolate(on){isolated=Boolean(on)&&selected!==null;applySelection();},
    setRotation(on){controls.autoRotate=on;},
    setPaused(value){paused=value;},
  };
}
