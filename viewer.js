import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export function createViewer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(0xffffff, 0);
  container.appendChild(renderer.domElement);
  renderer.domElement.setAttribute('role','img');
  renderer.domElement.setAttribute('aria-label','3D part segmentation. Drag to rotate, scroll or pinch to zoom.');
  renderer.domElement.tabIndex = 0;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 1, .01, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = .08;
  controls.autoRotateSpeed = 1.4;
  controls.minDistance = .8;
  controls.maxDistance = 8;
  controls.enablePan = false;
  controls.listenToKeyEvents(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa79e, 2.2));
  const light = new THREE.DirectionalLight(0xffffff, 2.1);
  light.position.set(3,5,4);
  scene.add(light);
  const fill = new THREE.DirectionalLight(0xffffff, .7);
  fill.position.set(-3,1,-3);
  scene.add(fill);
  const loader = new GLTFLoader();
  let current, generation=0, paused=false;

  function dispose(object) {
    object?.traverse(node => {
      if (!node.isMesh) return;
      node.geometry.dispose();
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      for (const material of materials) {
        for (const value of Object.values(material)) if (value?.isTexture) value.dispose();
        material.dispose();
      }
    });
  }
  function reset() {
    controls.autoRotate = false;
    const mobile = container.clientWidth < 550;
    camera.position.set(0,0,3.7).multiplyScalar(mobile ? 1.17 : 1);
    controls.target.set(0,0,0);
    controls.update();
  }
  function resize() {
    const width = container.clientWidth, height = container.clientHeight;
    if (!width || !height) return;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width,height,false);
  }
  new ResizeObserver(resize).observe(container);
  resize(); reset();
  renderer.setAnimationLoop(() => { if (!paused) { controls.update(); renderer.render(scene,camera); } });
  return {
    async load(url, rotation, progress) {
      const id = ++generation;
      const gltf = await loader.loadAsync(url, event => progress(event.lengthComputable ? Math.round(event.loaded/event.total*100) : 0));
      if (id !== generation) { dispose(gltf.scene); return; }
      if (current) { scene.remove(current); dispose(current); }
      const object = gltf.scene;
      const group = new THREE.Group();
      group.add(object);
      if (rotation) {
        const r=rotation;
        group.setRotationFromMatrix(new THREE.Matrix4().set(r[0][0],r[1][0],r[2][0],0,r[0][1],r[1][1],r[2][1],0,r[0][2],r[1][2],r[2][2],0,0,0,0,1));
      }
      const box = new THREE.Box3().setFromObject(group);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const factor = 2 / Math.max(size.x,size.y,size.z);
      group.scale.setScalar(factor);
      group.position.copy(center).multiplyScalar(-factor);
      current = group;
      scene.add(current);
      reset();
    },
    reset,
    setRotation(on) { controls.autoRotate = on; },
    setPaused(value) { paused = value; },
  };
}
