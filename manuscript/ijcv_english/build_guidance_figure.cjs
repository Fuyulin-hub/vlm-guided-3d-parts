const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require('C:/Users/ASUS/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 try{
  const page=await browser.newPage({viewport:{width:2000,height:1580}});
  const svg=fs.readFileSync(path.join(__dirname,'fig3_guidance.svg'),'utf8');
  await page.setContent(`<html><head><style>@page{size:2000px 1580px;margin:0}html,body{margin:0;padding:0;width:2000px;height:1580px}svg{display:block}</style></head><body>${svg}</body></html>`,{waitUntil:'load'});
  await page.evaluate(()=>document.fonts.ready);
  await page.pdf({path:path.join(__dirname,'fig3_guidance.pdf'),width:'2000px',height:'1580px',printBackground:true,margin:{top:0,right:0,bottom:0,left:0},pageRanges:'1'});
  await page.screenshot({path:path.join(__dirname,'build','pptx_review','fig3_guidance_preview.png')});
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
