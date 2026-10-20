import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import {syntheticBundle} from './test-fixtures.mjs';
const require=createRequire(import.meta.url);
const {chromium}=require('C:/Users/ASUS/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

async function viewer(t,initScript) {
  const server=spawn(process.execPath,['serve.cjs'],{cwd:fileURLToPath(new URL('.',import.meta.url)),env:{...process.env,PORT:'4175'},stdio:'pipe',windowsHide:true});
  t.after(()=>server.kill());
  await new Promise((resolve,reject)=>{server.stdout.once('data',resolve);server.once('error',reject);server.once('exit',code=>reject(new Error('server exited '+code)));});
  const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',args:['--enable-webgl','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  t.after(()=>browser.close());
  const page=await browser.newPage({viewport:{width:1440,height:1100}});
  if(initScript)await page.addInitScript(initScript);
  await page.goto('http://127.0.0.1:4175/');
  await page.waitForFunction(()=>!document.querySelector('#loadButton').disabled);
  return page;
}

async function exportRecord(page) {
  const pending=page.waitForEvent('download');
  await page.locator('#exportBinding').click();
  const stream=await(await pending).createReadStream();
  let json='';for await(const chunk of stream)json+=chunk;
  return JSON.parse(json);
}

test('a slow old GLB cannot overwrite the latest model identity or exported hashes',{timeout:60000},async t=>{
  const page=await viewer(t,()=>{
    const original=File.prototype.arrayBuffer;
    File.prototype.arrayBuffer=async function(){
      const bytes=await original.call(this);
      if(this.name==='slow.glb') {
        await new Promise(resolve=>{window.releaseSlowRead=resolve;});
        window.slowReadReturned=true;
      }
      return bytes;
    };
  });
  const old=syntheticBundle({unknown:true});old[0].name='slow.glb';
  const current=syntheticBundle();current[0].name='latest.glb';
  await page.locator('#fileInput').setInputFiles(old);
  await page.waitForFunction(()=>typeof window.releaseSlowRead==='function');
  await page.locator('#fileInput').setInputFiles(current);
  await page.waitForFunction(()=>document.querySelector('#loadStatus').textContent==='绑定完成');
  await page.evaluate(()=>window.releaseSlowRead());
  await page.waitForFunction(()=>window.slowReadReturned);
  // Old parsing has no images; allow its promise chain to settle before export.
  await page.waitForTimeout(200);
  const exported=await exportRecord(page);
  assert.equal(exported.source.glb_file,'latest.glb');
  assert.equal(exported.source.glb_sha256,JSON.parse(current[2].buffer).output_glb_sha256);
  assert.deepEqual(exported.meshes[0].face_label_ids,[7,7,19,19]);
});

test('a delayed sample fetch cannot replace a newer manual upload',{timeout:60000},async t=>{
  const page=await viewer(t);
  const old=syntheticBundle({unknown:true});
  let releaseFetch;
  const gate=new Promise(resolve=>{releaseFetch=resolve;});
  let firstRequest;
  const requested=new Promise(resolve=>{firstRequest=resolve;});
  let completed=0,requestCount=0;
  await page.route('**/assets/firetruck/**',async route=>{
    if(++requestCount===3)firstRequest();await gate;
    const name=new URL(route.request().url()).pathname.split('/').pop();
    const file=old[name==='segmented.glb'?0:name==='annotations.json'?1:2];
    try{await route.fulfill({status:200,contentType:file.mimeType,body:file.buffer});}catch{}finally{completed++;}
  });
  await page.locator('#sampleButton').click();await requested;
  const current=syntheticBundle();current[0].name='manual-latest.glb';
  await page.locator('#fileInput').setInputFiles(current);
  await page.waitForFunction(()=>document.querySelector('#loadStatus').textContent==='绑定完成');
  releaseFetch();
  while(completed<3)await new Promise(resolve=>setTimeout(resolve,10));
  await page.waitForTimeout(250);
  const exported=await exportRecord(page);
  assert.equal(exported.source.glb_file,'manual-latest.glb');
  assert.equal(exported.source.glb_sha256,JSON.parse(current[2].buffer).output_glb_sha256);
  assert.deepEqual(exported.meshes[0].face_label_ids,[7,7,19,19]);
});

test('front view and surface picking in the same frame use the current camera matrix',{timeout:60000},async t=>{
  const page=await viewer(t);
  await page.locator('#fileInput').setInputFiles(syntheticBundle());
  await page.waitForFunction(()=>document.querySelector('#loadStatus').textContent==='绑定完成');
  await page.getByRole('button',{name:/Right panel/}).click();
  assert.match(await page.locator('#partId').innerText(),/LABEL 19/);
  const label=await page.evaluate(()=>{
    document.querySelector('#frontView').click();
    const canvas=document.querySelector('#viewer canvas'),rect=canvas.getBoundingClientRect();
    const event={clientX:rect.left+rect.width*.40,clientY:rect.top+rect.height*.5,bubbles:true,pointerId:1,pointerType:'mouse',button:0};
    canvas.dispatchEvent(new PointerEvent('pointerdown',{...event,buttons:1}));
    canvas.dispatchEvent(new PointerEvent('pointerup',{...event,buttons:0}));
    return document.querySelector('#partId').textContent;
  });
  assert.match(label,/LABEL 7/);
});

test('low coverage warns without claiming full semantic binding and exports sampling limitations',{timeout:60000},async t=>{
  const page=await viewer(t);
  await page.locator('#fileInput').setInputFiles(syntheticBundle({unknown:true}));
  await page.waitForFunction(()=>!document.querySelector('#exportBinding').disabled);
  assert.match(await page.locator('#loadStatus').innerText(),/匹配不足/);
  assert.match(await page.locator('#bindingSummary').innerText(),/颜色匹配 0\.00%/);
  assert.match(await page.locator('#stageMessage').innerText(),/不能.*完整语义绑定/);
  assert.equal(await page.locator('.component.active').count(),0);
  const exported=await exportRecord(page);
  assert.equal(exported.method.sampling_approximate,true);
  assert.equal(exported.method.low_coverage_warning_threshold,.1);
});
