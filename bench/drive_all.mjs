// SELECTOR CONTRACT (理 12247 adds a frame to every STATIC screen: venue name, step title,
// a 'so far' summary like 'Haircut · Marcus · Tue 10:00', a back control and the AskBar).
// EVERY click below is exact-match on the chip's own text node, so the summary line cannot
// be hit by accident: 'Haircut' substring-matches the summary, 'Haircut' exact does not.
// If a future frame renders a chip's label as its OWN node too, this breaks loudly (strict
// mode violation) rather than silently clicking the wrong thing — which is the behaviour I want.
// WO-312 — the SANITY GATE, not the comparison. Prereg §4: "any arm's completion < 100% on the
// scripted user -> the world or the engine is broken, not the interface... the measurement is void
// until it does". §2b registered that the scripted run on v0 is a floor; this runs it.
// Every task is driven on its own page load so each gets its own logger session.
import { chromium } from 'playwright';
import fs from 'node:fs';

const tasksFile = process.argv[2];
const outFile = process.argv[3];
const limit = Number(process.argv[4] || 0);
const all = JSON.parse(fs.readFileSync(tasksFile, 'utf8'));
const tasks = limit ? all.tasks.slice(0, limit) : all.tasks;

const b = await chromium.launch();
const ctx = await b.newContext({ viewport: { width: 480, height: 900 } });
const results = [];
let failures = 0;

for (const task of tasks) {
  const p = await ctx.newPage();
  const warns = [];
  p.on('console', m => { if (m.type() === 'error') warns.push(m.text()); });
  p.on('pageerror', e => warns.push('PAGEERROR ' + String(e)));
  const t0 = Date.now();
  let taps = 0, error = null, finalText = null;
  try {
    await p.goto('http://localhost:5312/', { waitUntil: 'networkidle' });
    await p.waitForTimeout(150);
    const click = async (sel) => { await sel(); taps += 1; await p.waitForTimeout(120); };
    await click(() => p.getByText('Haircut', { exact: true }).first().click());
    await click(() => p.getByText(task.target.staff, { exact: true }).first().click());
    await click(() => p.getByText(task.target.date, { exact: true }).first().click());
    await click(() => p.getByText(task.target.slot, { exact: true }).first().click());
    // BY PLACEHOLDER, NOT BY POSITION. 理 12247's frame put the AskBar ABOVE the form, so
    // inputs.nth(0) silently became the chat bar: a run typed the NAME into ask.draft and the PHONE
    // into form.book.name, left the phone empty — and still scored 5 taps, hit@N 1.0 and a reached
    // submit screen. None of the five measurements can see a wrongly-filled form, so a positional
    // selector here is a hole the measurement itself cannot detect.
    await p.getByPlaceholder('Your name').fill('Test User');
    await p.getByPlaceholder('Phone number').fill('5551234567');
    await p.waitForTimeout(120);
    await click(() => p.getByText('Request this booking', { exact: true }).first().click());
    await p.waitForTimeout(250);
    finalText = (await p.locator('body').innerText()).includes('Request sent');
  } catch (e) {
    error = String(e).split('\n')[0].slice(0, 160);
    failures += 1;
  }
  results.push({
    task_id: task.task_id, goal: task.goal, taps, optimal_taps: task.optimal_taps,
    seconds: (Date.now() - t0) / 1000, submit_screen_reached: finalText, error,
    console_errors: warns.slice(0, 3),
  });
  await p.close();
  if (results.length % 20 === 0) console.log(`  ${results.length}/${tasks.length} driven, failures ${failures}`);
}
await b.close();

const ok = results.filter(r => !r.error && r.submit_screen_reached);
const secs = results.map(r => r.seconds).sort((a, z) => a - z);
fs.writeFileSync(outFile, JSON.stringify({
  document: 'WO-312 — scripted sweep on STATIC, the sanity gate (prereg §4 / §2b)',
  arm: 'static', tasks_file: tasksFile, n: results.length,
  reached_submit_screen: ok.length,
  completion_rate_SCREEN_ONLY: +(ok.length / results.length).toFixed(4),
  completion_note: 'SCREEN, not engine. §2 measurement 3 is the ENGINE\'s answer and the bench makes no engine call; this is the sanity gate only.',
  taps_values: [...new Set(results.map(r => r.taps))].sort(),
  optimal_taps_values: [...new Set(results.map(r => r.optimal_taps))].sort(),
  seconds_median: secs[Math.floor(secs.length / 2)], seconds_min: secs[0], seconds_max: secs[secs.length - 1],
  failures: results.filter(r => r.error).map(r => ({ task_id: r.task_id, error: r.error })),
  results,
}, null, 1));
console.log(`SWEEP: ${results.length} tasks · reached submit screen ${ok.length} · failures ${failures}`);
console.log(`taps values ${[...new Set(results.map(r => r.taps))].sort()} · seconds median ${secs[Math.floor(secs.length/2)].toFixed(2)}`);
