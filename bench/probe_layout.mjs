// POSITION, not DOM order. My 12304 verification read the body TEXT ORDER and concluded the bar had
// moved to the bottom. Text order is DOM order; a flex layout can put a DOM-last element anywhere,
// and 形's 12340 says that is exactly what happened — render order changed, position did not.
// This measures where things actually ARE.
import { chromium } from 'playwright';
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 390, height: 844 } });
await p.goto('http://localhost:5312/', { waitUntil: 'networkidle' });
await p.waitForTimeout(500);
const box = async (sel) => { try { return await sel.boundingBox(); } catch { return null; } };
const ask = await box(p.getByText('Ask', { exact: true }).first());
const offer = await box(p.getByText('Haircut', { exact: true }).first());
const vp = p.viewportSize();
console.log(`viewport ${vp.width}x${vp.height}`);
console.log(`  offer chip  y=${offer && Math.round(offer.y)}  (top of the head)`);
console.log(`  Ask button  y=${ask && Math.round(ask.y)}  bottom edge=${ask && Math.round(ask.y + ask.height)}`);
if (ask && offer) {
  const barBelow = ask.y > offer.y;
  const nearFloor = (vp.height - (ask.y + ask.height)) < 120;
  console.log(`  bar BELOW the head: ${barBelow}`);
  console.log(`  bar within 120px of the viewport floor: ${nearFloor}  (gap ${Math.round(vp.height - (ask.y + ask.height))}px)`);
  console.log(`  VERDICT: ${barBelow && nearFloor ? 'pinned to the bottom' : 'NOT pinned — it sits where content ends'}`);
}
await b.close();
