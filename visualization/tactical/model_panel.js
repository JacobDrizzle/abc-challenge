// model_panel.js: deck 2's live stage, the model (ticket 06b; SPEC A6, C4 to C5, G5, H3 to H6; CONTRACTS sections 3, 4, 6).
// Loaded by the deck script (story.js) as <script type="module" src="model_panel.js"> once the browser is idle and the reader has
// left the first step. It adds no 3D of its own: it starts the viewer (viewer.mjs) embedded on the deck's open stage and adapts the
// viewer's small API (window.K8Viewer) to the live-stage contract, so the story never reaches into the viewer's internals.
//
//   mount(el)             el = #modelStage. A full-size child element is made inside it (the "host": the viewer draws into that, so
//                         the host can grow past the stage on the last step without restyling #modelStage itself). Then the start
//                         globals are set (K8_NOSTART = null, K8_EMBED = { el: host, user: 'test50', view, clear: true }) and viewer.mjs
//                         is inserted as a new module script. When K8Viewer.ready resolves (the first trace is on screen),
//                         data-live="on" goes on #modelStage and the deck's CSS fades the fallback cards out.
//   setState(s)           'current': drawing and the trace playing (unless paused); 'near': drawing, not playing; 'far': asleep, the
//                         viewer's loop is blocked so nothing at all is drawn.
//   showStep(view)        debounced: only a step the reader rests on for 150 ms is sent to the viewer (fast scrolling asks for one).
//   setPos(s, key, extra) the arrival (extra.lead 1 → 0: the input strip comes down into place) and the last step's growth to the window.
//   release()             asleep, the inset back home, data-live removed (the viewer keeps its WebGL context: see release()); from
//                         then on the deck's calls are ignored. Also done by itself when the viewer's WebGL context is lost (the
//                         browser can drop a context, e.g. after a GPU reset): the cards come back instead of an empty stage.
// Also: #modelPause toggles playback (aria-pressed), #modelSpeed states the speed the viewer really plays at, and on the head and traj
// steps the step's formula lines (expert.js, through K8Viewer.math) and the viewer's inset graph are shown in the text column.

const VIEW_DELAY = 150;            // ms without a step change before the viewer is asked for that step
const INSET_VIEWS = ['head', 'traj'];   // the steps whose text column shows the formulas and the viewer's inset graph (SPEC H5)
const REAL_RATE = 100;             // samples per second of the recording: "real speed"

const clamp = v => v < 0 ? 0 : v > 1 ? 1 : v;
const ease = t => t < .5 ? 2 * t * t : 1 - (2 - 2 * t) * (1 - t);   // the deck's own ease (story.js), so growth moves like the cards did

let el = null, host = null, V = null, ready = false, starting = false, released = false;
let state = 'far', paused = false;
let want = null, timer = 0;        // the view the deck asked for last, and the pending debounce
let pos = { s: 0, key: null, lead: 0, tail: 0 }, grownTo = -1;
let insetHome = null, insetStyle = '', insetAt = null;
let speedTick = 0;
let entered = false;               // ticket 07: true once the reader has entered the viewer (enterViewer)
let geo = null, nSteps = 9;        // review Q6: the stage's box on the page and the window width, cached (see measureStage); the deck's step count
let full = false;                  // review Q7: true while the host has its full-window size (the deck's last step)

// ---------------------------------------------------------------- start
function mount(target) {
  if (el || released) return;
  if (!(window.K8_BOOT && window.K8_BOOT.live3d)) return;   // no real 3D: the cards stay (CONTRACTS 3)
  el = target;
  if (getComputedStyle(el).position === 'static') el.style.position = 'relative';
  host = document.createElement('div');
  host.setAttribute('aria-hidden', 'true');
  Object.assign(host.style, { position: 'absolute', left: '0', top: '0', width: '100%', height: '100%', pointerEvents: 'none', overflow: 'hidden' });
  el.appendChild(host);
  const track = el.closest('.dk-track'); if (track) nSteps = track.querySelectorAll('.dk-step').length;
  // A lost WebGL context (review Q2): webglcontextlost is fired at the viewer's canvas, which sits inside the host while embedded. A
  // listener in the capture phase on the host hears it whatever the event's bubbling, and once the viewer has been entered (its canvas
  // moved out of the host) it no longer hears anything, so it is embedded-only by construction.
  host.addEventListener('webglcontextlost', () => { if (!entered) release(); }, true);
  // Review Q6: the stage's box is read when its size changes (a ResizeObserver runs after layout, so reading there forces nothing) and on
  // a window resize, never in the per-frame growth below; a change re-applies the growth with the new numbers.
  new ResizeObserver(restage).observe(el);
  addEventListener('resize', restage, { passive: true });
  // The deck calls showStep right after mount (same task), so the viewer is started a moment later and can begin on that step.
  queueMicrotask(start);
}

function start() {
  if (starting) return; starting = true;
  window.K8_NOSTART = null;
  window.K8_EMBED = { el: host, user: 'test50', view: want || 'input', clear: true };
  // ?embed=f2 is a version tag (see MODULES in story.js). It must stay different from index.html's own viewer.mjs?r=... address:
  // a browser runs a module once per address, and this second address is what makes the viewer run again, embedded.
  const sc = document.createElement('script'); sc.type = 'module'; sc.src = 'viewer.mjs?embed=f2';
  // A module's load event fires after it has run, so K8Viewer exists by then (or the viewer could not start: the cards stay).
  sc.onload = () => { if (window.K8Viewer && window.K8Viewer.ready) window.K8Viewer.ready.then(onReady, () => {}); };
  document.head.appendChild(sc);
}

function onReady() {
  if (released) return;
  V = window.K8Viewer; ready = true;
  // the formula lines, as text, into the head and traj steps' maths blocks (existing content of expert.js only)
  for (const v of INSET_VIEWS) { const m = stepOf(v, '.dk-math'); if (m) m.textContent = V.math(v).join('\n'); }
  applyState();
  if (want && V.step() !== want) V.showStep(want);
  placeInset(want);
  setPos(pos.s, pos.key, { lead: pos.lead, tail: pos.tail });
  el.setAttribute('data-live', 'on');                  // the first frame is on screen (ready resolves after it is drawn)
  speedText();
}

// ---------------------------------------------------------------- the contract
function setState(s) {
  if (entered || released) return;                      // the viewer is the page now, or the stage was released: the deck no longer drives it
  state = s === 'current' || s === 'near' ? s : 'far';
  applyState();
}
function applyState() {
  if (!V) return;
  if (state === 'far') { V.setPlaying(false); V.setActive(false); }
  else { V.setActive(true); V.setPlaying(state === 'current' && !paused); }
  // while current the speed tag is refreshed once a second: the viewer may drop to its fastest setting if frames cannot keep up
  clearInterval(speedTick); speedTick = state === 'current' ? setInterval(speedText, 1000) : 0;
  speedText();
}

// Fast scrolling: every request restarts a 150 ms timer, so only the step the reader stops on reaches the viewer (whose flight and
// animations would otherwise restart for every step scrolled past).
function showStep(view) {
  if (entered || released) return;
  want = view;
  clearTimeout(timer);
  timer = setTimeout(() => { timer = 0; if (!ready) return; if (V.step() !== want) V.showStep(want); placeInset(want); V.arrive(pos.lead); }, VIEW_DELAY);
}

function setPos(s, key, extra) {
  if (entered || released) return;
  pos = { s: +s || 0, key, lead: extra ? +extra.lead || 0 : 0, tail: extra ? +extra.tail || 0 : 0 };
  if (!ready) return;
  V.arrive(pos.lead);                                    // the arrival: the viewer moves its own input strip (input step only)
  grow(pos.s);
}

// Growth on the last step (SPEC A6): over the scroll from the second-last step to the last, the scene grows from the stage's box to the
// whole window width (the stage already runs from under the top bar to the window's bottom).
// Review Q7: resizing the viewer's canvas reallocates its drawing buffer (the window's size times min(device pixel ratio, 2)), so it is
// done once, not every frame: as the last step begins the host takes its final size (the whole window) and as it ends, going up, it
// takes the stage's size again; the viewer's resize observer then resizes the buffer once each time. In between only cheap things follow
// the scroll: the CSS mask shows the part of the full-size canvas that the growing scene covers at this point (with the soft edge
// towards the text), and K8_EMBED.lens puts the picture's centre in the middle of that part, so the model moves exactly as it did when
// the canvas itself grew. K8_EMBED.free (in the host's pixels) keeps the model framed in the stage's own box while the text column is
// still there.
function measureStage() {
  const r = el.getBoundingClientRect();
  geo = { left: r.left, width: r.width, height: r.height, W: document.documentElement.clientWidth };
}
function restage() {
  if (!el || released || entered) return;
  measureStage();
  if (!ready) return;
  full = false; grownTo = -1; grow(pos.s);                // the new box: the host's size and the mask again
}
function grow(s) {
  const g = ease(clamp(s - (nSteps - 2)));
  if (Math.abs(g - grownTo) < 1e-4) return;
  grownTo = g;
  if (!geo) measureStage();                              // only until the first ResizeObserver call has filled the cache
  const E = window.K8_EMBED, { left, width, height, W } = geo;
  if (g === 0) {
    if (full || host.style.width !== '100%') { Object.assign(host.style, { left: '0', width: '100%' }); }
    full = false;
    if (E) { E.free = null; E.lens = null; }
    edge(0, 0, width);
    return;
  }
  const wasFull = full;
  if (!full) {                                           // once, as the last step begins: the host is the whole window
    full = true;
    host.style.left = (-left).toFixed(1) + 'px';
    host.style.width = W.toFixed(1) + 'px';
    if (E) E.free = { left, top: 0, width, height };
  }
  // the part of the window the scene covers at g, in the host's pixels (the host starts at the window's left edge)
  const L = left * (1 - g), R = L + width + (W - width) * g;
  edge(g, L, R);
  if (E) E.lens = g >= 1 ? null : { cx: (L + R) / 2, w: W, h: height };
  // The frame the host was resized in, the viewer's resize observer re-aims (with the lens just set); after that, a re-aim is asked for.
  if (wasFull && V.reframe) V.reframe();
}
// The scene has no frame (SPEC G1), but a canvas still ends somewhere: on the side of the text column its last EDGE_FADE of width fades
// to transparent with a CSS mask, so the model dissolves towards the text instead of being cut by a straight line. The mask goes as
// the scene grows to the whole window. (The deck's text is on the right in this deck; the side is read from the track anyway.)
// Not grown (g = 0) the host is the stage's box and the fade is a share of it; growing, L and R are the edges of the part the scene covers,
// in the host's pixels, and outside them the mask is transparent.
const EDGE_FADE = 12;              // % of the scene's width
function edge(g, L, R) {
  const track = el.closest('.dk-track'), toRight = !track || track.dataset.side !== 'left';
  const f = EDGE_FADE * (1 - g) / 100 * (R - L), px = v => v.toFixed(1) + 'px';
  const m = g === 0 ? `linear-gradient(to ${toRight ? 'left' : 'right'}, transparent 0, #000 ${EDGE_FADE.toFixed(2)}%)` : g >= 1 ? 'none'
    : toRight ? `linear-gradient(to right, transparent ${px(L)}, #000 ${px(L)}, #000 ${px(R - f)}, transparent ${px(R)})`
    : `linear-gradient(to right, transparent ${px(L)}, #000 ${px(L + f)}, #000 ${px(R)}, transparent ${px(R)})`;
  host.style.webkitMaskImage = m; host.style.maskImage = m;
}

// The viewer has no way to give its WebGL context back and start again (its module runs once), so release() puts it to sleep, takes
// the story's borrowed pieces back and shows the cards again; the context itself is freed when the page goes.
function release() {
  released = true; clearTimeout(timer); clearInterval(speedTick);
  if (V) { V.setPlaying(false); V.setActive(false); }
  placeInset(null);
  for (const v of INSET_VIEWS) { const m = stepOf(v, '.dk-math'); if (m) m.textContent = ''; }
  if (el) el.removeAttribute('data-live');
}

// ---------------------------------------------------------------- the maths and the inset (SPEC H5)
function stepOf(view, sel) {
  const track = el && el.closest('.dk-track'), st = track && track.querySelector(`.dk-step[data-view="${view}"]`);
  return st ? st.querySelector(sel) : null;
}
// The viewer's inset (its live curve graph, redrawn as the trace plays) is moved into the current step's div.dk-inset on head and traj,
// and put back exactly where it was otherwise. While borrowed it is inert (no focus, no pointer, not read twice: the formulas beside it
// are the text version) and drawn flat, without its floating-panel look.
const BORROWED = { position: 'static', top: 'auto', right: 'auto', width: 'auto', maxHeight: 'none', overflow: 'visible', padding: '0',
  background: 'none', border: '0', borderRadius: '0', backdropFilter: 'none', webkitBackdropFilter: 'none', boxShadow: 'none' };
// The text column is narrow and must fit the window, so the borrowed inset shows its graphs only: its tab row, notes and number tables
// (which the viewer's side panel has room for) are hidden by this one rule, made here because only this module borrows the inset.
const COMPACT = '.dk-inset > #inset .tabs, .dk-inset > #inset .m, .dk-inset > #inset table { display:none !important; }'
  + ' .dk-inset > #inset h3 { font-size:12px; margin:0 0 6px; } .dk-inset > #inset svg[role="img"] { max-height:16vh; width:auto !important; max-width:100%; }'
  + ' @media (max-height: 820px) { .dk-inset > #inset h3 { display:none; } .dk-inset > #inset svg[role="img"] { max-height:8.5vh; } }';   // a short window: graphs only, smaller
function compactStyle() {
  if (document.getElementById('k8ModelInsetCss')) return;
  const st = document.createElement('style'); st.id = 'k8ModelInsetCss'; st.textContent = COMPACT; document.head.appendChild(st);
}
function placeInset(view) {
  const ins = V && V.inset ? V.inset() : null; if (!ins) return;
  compactStyle();
  const slot = INSET_VIEWS.includes(view) ? stepOf(view, '.dk-inset') : null;
  if (slot) {
    if (!insetHome) { insetHome = [ins.parentNode, ins.nextSibling]; insetStyle = ins.getAttribute('style') || ''; }
    if (ins.parentNode !== slot) slot.appendChild(ins);
    Object.assign(ins.style, BORROWED); ins.inert = true; insetAt = view;
  } else if (insetHome) {
    insetHome[0].insertBefore(ins, insetHome[1]);
    if (insetStyle) ins.setAttribute('style', insetStyle); else ins.removeAttribute('style');
    ins.inert = false; insetHome = null; insetAt = null;
  }
}

// ---------------------------------------------------------------- pause and the speed tag
function speedText() {
  const t = document.getElementById('modelSpeed'); if (!t) return;
  if (!V) { t.textContent = ''; return; }
  const r = V.speed();
  t.textContent = paused ? 'paused' : r === REAL_RATE ? 'playing at real speed' : `playing at ${+(r / REAL_RATE).toFixed(2)}× speed`;
}
function setPaused(p) {
  paused = !!p;
  const b = document.getElementById('modelPause'); if (b) b.setAttribute('aria-pressed', paused ? 'true' : 'false');
  applyState();
}
// #modelPause is a real button in the text column (CONTRACTS 4). Delegated, so it works whatever the markup's timing.
document.addEventListener('click', e => { const b = e.target && e.target.closest && e.target.closest('#modelPause'); if (b) setPaused(!paused); });
{ const b = document.getElementById('modelPause'); if (b) b.setAttribute('aria-pressed', 'false'); }

// ---------------------------------------------------------------- ticket 07: entering the viewer (SPEC C6, C7, B7, G8)
// a#exploreBtn is an ordinary link to ?explore=1 (a page load into the viewer). Once the embed is ready (K8Viewer.ready resolved and
// this stage live) a plain click is taken over and the page becomes the viewer with no load: the scene the reader is looking at is
// the viewer. A click with a modifier key (new tab or window) and any click before ready stay the link.
const canEnter = () => ready && !released && !entered && !!V && typeof V.enter === 'function' && !!el && el.getAttribute('data-live') === 'on';
document.addEventListener('click', e => {
  const a = e.target && e.target.closest && e.target.closest('#exploreBtn');
  if (!a || e.defaultPrevented || e.button || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || !canEnter()) return;
  e.preventDefault(); enterViewer();
});
function enterViewer() {
  const wasPlaying = V.playing();                        // the story pauses as its stages go far; the viewer goes on as the reader left it
  clearTimeout(timer); timer = 0;
  // 1. every live stage goes far (draws nothing), 2. deck 1's stage frees its WebGL context (at most two contexts, SPEC B7 and G8),
  // 3. the deck script stops (no more measuring or telling stages anything), 4. the borrowed inset goes back to the viewer's side panel.
  setState('far');
  const task = window.K8StageTask;
  if (task) { try { task.setState('far'); if (task.release) task.release(); } catch (err) { setTimeout(() => { throw err; }); } }
  if (window.K8Story && window.K8Story.suspend) window.K8Story.suspend();
  placeInset(null);
  entered = true; clearInterval(speedTick); speedTick = 0;
  // 5. the hidden story is `inert`: no focus, no clicks, nothing read out, so Tab cannot reach anything in it from the viewer.
  const story = document.getElementById('story'); if (story) story.inert = true;
  // side.js (the Data | Model tabs) makes everything but its own parts inert while the Data side is open, and un-inerts all of
  // <body>'s children when it closes, the story too. It announces each switch with k8:side after doing so, so the story is made
  // inert again right after it.
  addEventListener('k8:side', () => { if (story) story.inert = true; });
  // 6. the viewer takes the page (viewer.mjs enter(): canvas and labels home, embed and story mode off, controls, address, Back button)
  V.enter();
  V.setPlaying(wasPlaying);
}

// Back from the viewer is a page load of the story at #s-explore (viewer.mjs toStory, which leaves a marker in sessionStorage). The deck
// script restores that step on load and clears the marker (K8Story.returned); this puts the keyboard focus back on the button the reader left by.
// (This module loads a little after the page, once the browser is idle, so the focus comes back then; it is not taken from anything
// the reader has focused meanwhile. preventScroll: focusing must not move the page off the restored step.)
(function returnFocus() {
  // story.js reads and clears the sessionStorage marker at load (it loads in every case; this module does not), and passes it on here.
  const back = !!(window.K8Story && window.K8Story.returned);
  if (!back || location.hash !== '#s-explore') return;
  let tries = 0;
  const go = () => {
    const b = document.getElementById('exploreBtn'), a = document.activeElement;
    if (b && (!a || a === document.body)) b.focus({ preventScroll: true });
    if (document.activeElement !== b && ++tries < 120 && (!a || a === document.body)) requestAnimationFrame(go);   // the step may not be shown yet
  };
  go();
})();

// homeInset(): puts the borrowed inset back in the viewer's side panel; ticket 07 calls it before K8Viewer.enter().
export const stage = { mount, setState, setPos, showStep, release, homeInset: () => placeInset(null) };
window.K8StageModel = stage;
// test hooks for the checks (read only): which view was last sent, whether the viewer is ready, where the inset is, the paused flag
window.__model = { ready: () => ready, want: () => want, pending: () => !!timer, insetAt: () => insetAt, paused: () => paused };
if (window.K8Story && typeof window.K8Story.addLive === 'function') window.K8Story.addLive('model', stage);
