// Read-only browser checks of the live dashboard and existing video library.
const {chromium} = require('/home/lsx/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({headless:true,args:['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1080},deviceScaleFactor:1});
    const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto('http://127.0.0.1:8765/',{waitUntil:'networkidle'});
    await page.waitForFunction(()=>document.querySelector('#connection').textContent.includes('已连接'));
    await page.waitForFunction(()=>document.querySelectorAll('.metric-chart svg').length===6,{},{timeout:180000});
    const runName=await page.locator('#run-select').inputValue();
    const balanceFirst=runName.startsWith('push_balance_first18')||runName.startsWith('push_sustained_balance18')||runName.startsWith('push_rated_balance18');
    assert.equal(await page.locator('#arm-legend button').count(),balanceFirst ? 1:2);
    if (balanceFirst) {
      assert.ok((await page.locator('#learning-note').textContent()).includes('当前课程标准'));
      assert.ok((await page.locator('#eval-body').textContent()).includes('原严格标准'));
      assert.ok((await page.locator('#experiment-description').textContent()).includes('4 个腿电机'));
    }
    await page.locator('#learning').scrollIntoViewIfNeeded();
    const chart=await page.locator('#chart-reward').boundingBox();
    await page.mouse.move(chart.x+chart.width*.65,chart.y+80);
    await page.waitForFunction(()=>!document.querySelector('#chart-reward .chart-tooltip').hidden);
    const tip=await page.locator('#chart-reward .chart-tooltip').textContent();
    assert.ok(tip.includes('步'));
    await page.locator('#arm-legend button').first().click();
    assert.equal(await page.locator('#arm-legend button').first().getAttribute('aria-pressed'),'false');
    await page.locator('#arm-legend button').first().click();
    await page.locator('#chart-group').selectOption('standing');
    assert.equal(await page.locator('#chart-group').inputValue(),'standing');
    await page.locator('#chart-group').selectOption('exact18');
    await page.screenshot({path:'/home/lsx/wheel_leg/training_console/学习曲线.png',fullPage:true});
    const live=await page.evaluate(()=>({run:document.querySelector('#run-select').value,
      phase:document.querySelector('#phase').textContent,progress:document.querySelector('#progress-caption').textContent,
      charts:document.querySelectorAll('.metric-chart svg').length}));
    await page.setViewportSize({width:390,height:844});
    await page.waitForTimeout(300);
    const sizes=await page.evaluate(()=>({viewport:innerWidth,scroll:document.documentElement.scrollWidth}));
    assert.ok(sizes.scroll<=sizes.viewport,JSON.stringify(sizes));
    await page.screenshot({path:'/home/lsx/wheel_leg/training_console/上位机窄屏.png',fullPage:true});
    // Verify historical records keep distinct branch curves.
    await page.setViewportSize({width:1440,height:1080});
    await page.locator('#run-select').selectOption('push_anchor_ab18_20261111_1815');
    await page.waitForFunction(()=>document.querySelector('#phase').textContent==='本轮已结束');
    await page.waitForFunction(()=>document.querySelectorAll('#chart polyline').length===2);
    // Simulate a disconnected read in this test tab only; never change training.
    await page.route('**/api/status*',async route=>{
      const response=await route.fetch();const data=await response.json();
      await route.fulfill({json:{...data,connected:false,error:'UI test offline'}});
    });
    await page.locator('#refresh').click();
    await page.waitForFunction(()=>document.querySelector('#connection').textContent.includes('未连接'));
    assert.equal(await page.locator('#remaining-time').textContent(),'连接恢复后重新估算');
    await page.unroute('**/api/status*');
    await page.goto('http://127.0.0.1:8765/',{waitUntil:'networkidle'});
    await page.waitForFunction(()=>document.querySelector('#connection').textContent.includes('已连接'));
    await page.screenshot({path:'/home/lsx/wheel_leg/training_console/上位机界面.png',fullPage:true});
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({passed:true,live,tooltip:tip,mobile:sizes,errors},null,2));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
