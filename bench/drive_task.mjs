// SELECTOR CONTRACT (理 12247 adds a frame to every STATIC screen: venue name, step title,
// a 'so far' summary like 'Haircut · Marcus · Tue 10:00', a back control and the AskBar).
// EVERY click below is exact-match on the chip's own text node, so the summary line cannot
// be hit by accident: 'Haircut' substring-matches the summary, 'Haircut' exact does not.
// If a future frame renders a chip's label as its OWN node too, this breaks loudly (strict
// mode violation) rather than silently clicking the wrong thing — which is the behaviour I want.
// WO-312 — drive ONE task from the scripted list all the way to submit, through the real STATIC
// head, with 鉋's logger running. This is the first end-to-end check that the three pieces meet:
// my task list's arg shapes, 形's reducer, and 鉋's replay log. 形 (12117) and 鉋 (12093) both
// asked to see a REAL tap rather than a synthetic one.
import { chromium } from 'playwright';
import fs from 'node:fs';

const tasks = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const task = tasks.tasks.find(t => t.task_id === (process.argv[3] || 'qc-001'));
if (!task) throw new Error('no such task');
console.log('TASK', task.task_id, '::', task.goal);

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 480, height: 900 } });
const warns = [];
p.on('console', m => { if (m.type() === 'warning' || m.type() === 'error') warns.push(m.text()); });
p.on('pageerror', e => warns.push('PAGEERROR ' + String(e)));

const t0 = Date.now();
await p.goto('http://localhost:5312/', { waitUntil: 'networkidle' });
await p.waitForTimeout(600);

let taps = 0;
const tap = async (label, how) => {
  await how();
  taps += 1;
  await p.waitForTimeout(250);
  console.log(`  tap ${taps}: ${label}`);
};

await tap(`offer ${task.optimal_path[0].args.offer_id}`, () => p.getByText('Haircut', { exact: true }).first().click());
await tap(`staff ${task.target.staff}`, () => p.getByText(task.target.staff, { exact: true }).first().click());
await tap(`day ${task.target.date}`, () => p.getByText(task.target.date, { exact: true }).first().click());
await tap(`slot ${task.target.slot}`, () => p.getByText(task.target.slot, { exact: true }).first().click());

// form-fill is typing, NOT taps — the prereg says one submission is one tap and typing is not taps
// BY PLACEHOLDER, NOT BY POSITION. 理 12247's frame put the AskBar ABOVE the form, so
// inputs.nth(0) silently became the chat bar: a run typed the NAME into ask.draft and the PHONE
// into form.book.name, left the phone empty — and still scored 5 taps, hit@N 1.0 and a reached
// submit screen. None of the five measurements can see a wrongly-filled form, so a positional
// selector here is a hole the measurement itself cannot detect.
await p.getByPlaceholder('Your name').fill('Test User');
await p.getByPlaceholder('Phone number').fill('5551234567');
await p.waitForTimeout(250);

await tap('submit', () => p.getByText('Request this booking', { exact: true }).first().click());
await p.waitForTimeout(600);
const seconds = (Date.now() - t0) / 1000;
const finalText = (await p.locator('body').innerText()).split('\n').map(s => s.trim()).filter(Boolean);
console.log('FINAL SCREEN:', JSON.stringify(finalText));
console.log(`TAPS ${taps} (optimal ${task.optimal_taps})   SECONDS ${seconds.toFixed(1)}`);
console.log('CONSOLE WARN/ERR:', warns.length ? warns.slice(0, 6) : 'none');
await b.close();
