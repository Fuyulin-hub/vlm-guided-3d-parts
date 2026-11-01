const fs = require('node:fs');
const path = require('node:path');
const sharp = require('C:/Users/ASUS/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
const { chromium } = require('C:/Users/ASUS/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
  const browser = await chromium.launch({headless: true, executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  try {
    for (let [source, name, width, height] of [
      ['fig1_motivation.svg', 'fig1_motivation', 2572, 777],
      ['fig2_pipeline.drawio.svg', 'fig2_pipeline', 2471, 1108],
    ]) {
      let svg = fs.readFileSync(path.join(__dirname, source), 'utf8');
      if (name === 'fig2_pipeline') {
        const dims = svg.match(/viewBox="([^"]+)"/)[1].split(/\s+/).map(Number);
        width = dims[2]; height = dims[3];
      }
      svg = svg.replace(/<\?xml[^>]*>/g, '').replace(/<!DOCTYPE[\s\S]*?>/g, '');
      const page = await browser.newPage({ viewport: {width, height}, deviceScaleFactor: 1 });
      await page.setContent(`<html><head><style>@page{size:${width}px ${height}px;margin:0}html,body{margin:0;padding:0;width:${width}px;height:${height}px;background:white}svg{display:block;width:${width}px;height:${height}px}</style></head><body>${svg}</body></html>`, {waitUntil: 'load'});
      await page.evaluate(() => document.fonts.ready);
      let pageWidth = width, pageHeight = height;
      if (name === 'fig2_pipeline') {
        const preview = await page.screenshot();
        const {info} = await sharp(preview).trim({background: '#ffffff', threshold: 12}).toBuffer({resolveWithObject: true});
        const left = Math.max(0, -info.trimOffsetLeft - 8);
        const top = Math.max(0, -info.trimOffsetTop - 8);
        pageWidth = Math.min(width - left, info.width + 16);
        pageHeight = Math.min(height - top, info.height + 16);
        await page.evaluate(({left, top, pageWidth, pageHeight}) => {
          const svg = document.querySelector('svg');
          Object.assign(svg.style, {position:'absolute', left: -left + 'px', top: -top + 'px'});
          document.body.style.width = document.documentElement.style.width = pageWidth + 'px';
          document.body.style.height = document.documentElement.style.height = pageHeight + 'px';
          document.head.querySelector('style').textContent += `@page{size:${pageWidth}px ${pageHeight}px;margin:0}`;
        }, {left, top, pageWidth, pageHeight});
        await page.setViewportSize({width: pageWidth, height: pageHeight});
        console.log('fig2: trimmed exterior canvas margins only', {left, top, pageWidth, pageHeight});
      }
      await page.pdf({path: path.join(__dirname, name + '.pdf'), width: pageWidth + 'px', height: pageHeight + 'px', printBackground: true, margin: {top: 0, right: 0, bottom: 0, left: 0}, pageRanges: '1'});
      await page.screenshot({path: path.join(__dirname, 'build', name + '.png')});
      console.log(name + ': PDF and preview rendered');
      await page.close();
    }
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
