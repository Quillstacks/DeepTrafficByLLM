/*
 * Node reference harness for MIT DeepTraffic v2.0.
 *
 * Loads the ORIGINAL gameopt.js (the full simulation engine) headless, exactly
 * as the official eval_webworker.js does: it stubs the browser globals
 * (`self`, `window`, `document`), loads the ConvNetJS files into a SHARED global
 * scope (the way `importScripts` works in a real WebWorker), defines a brain /
 * policy, then calls `doEvalRun`.
 *
 * It produces the REFERENCE scores (median + mean) for:
 *   (a) the network_basic.js brain               --policy=brain
 *   (b) fixed policy "always accelerate" (act 1)  --policy=accel
 *   (c) fixed policy "do nothing"      (act 0)    --policy=nop
 *
 * It can also DUMP per-frame state of all 20 cars (b, a, c, y) under an
 * identical scripted action sequence + identical deterministic seeds, which is
 * the substrate for the trajectory fidelity test against the Python port.
 *
 * Usage:
 *   node run_ref.js eval  --policy=accel  --runs=500 --frames=2000 [--det=true]
 *   node run_ref.js dump  --policy=accel  --frames=120 --seed-offset=1 > traj.json
 *   node run_ref.js dump  --actions=1,1,2,0,4,3,... --frames=300 --seed-offset=5
 *
 * All simulation parameters (lanesSide, patchesAhead, ...) default to the
 * network_basic config and can be overridden via flags.
 */

'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

// ----------------------------------------------------------------------------
// CLI parsing
// ----------------------------------------------------------------------------
const argv = process.argv.slice(2);
const mode = argv[0] || 'eval';
const opts = {};
for (const a of argv.slice(1)) {
  const m = a.match(/^--([^=]+)=(.*)$/);
  if (m) opts[m[1]] = m[2];
  else if (a.startsWith('--')) opts[a.slice(2)] = true;
}
function numOpt(k, d) { return opts[k] !== undefined ? Number(opts[k]) : d; }
function boolOpt(k, d) {
  if (opts[k] === undefined) return d;
  return opts[k] === true || opts[k] === 'true' || opts[k] === '1';
}

const LANES_SIDE     = numOpt('lanes-side', 1);
const PATCHES_AHEAD  = numOpt('patches-ahead', 10);
const PATCHES_BEHIND = numOpt('patches-behind', 0);
const OTHER_AGENTS   = numOpt('other-agents', 0);
const TEMPORAL       = numOpt('temporal-window', 0);
const POLICY         = opts['policy'] || 'nop';

// ----------------------------------------------------------------------------
// Build a sandbox that emulates the WebWorker browser-ish global environment.
// gameopt.js + convnetjs all run in ONE shared global scope (like importScripts),
// so we use a single vm context and runInContext each file into it.
// ----------------------------------------------------------------------------
function noop() {}
function makeFakeElement() {
  return {
    innerText: '', innerHTML: '', value: '',
    getContext: function () {
      return {
        scale: noop, translate: noop, clearRect: noop, fillRect: noop,
        drawImage: noop, beginPath: noop, arc: noop, fill: noop,
        save: noop, restore: noop, globalCompositeOperation: '',
        globalAlpha: 1, fillStyle: '',
      };
    },
    addEventListener: noop, appendChild: noop,
    style: {}, width: 0, height: 0,
  };
}

function buildSandbox() {
  const sandbox = {};
  sandbox.self = sandbox;
  sandbox.window = sandbox;
  sandbox.global = sandbox;
  // Route ALL engine console output to STDERR so stdout carries ONLY our JSON
  // result. gameopt.js logs "cloning" / periodic G to console.log; keeping those
  // off stdout makes the result trivially parseable by the Python tests.
  sandbox.console = {
    log: function () { console.error.apply(console, arguments); },
    error: function () { console.error.apply(console, arguments); },
    warn: function () { console.error.apply(console, arguments); },
  };
  sandbox.Math = Math;
  sandbox.Array = Array;
  sandbox.Date = Date;
  sandbox.JSON = JSON;
  sandbox.Object = Object;
  sandbox.parseInt = parseInt;
  sandbox.parseFloat = parseFloat;
  sandbox.isNaN = isNaN;
  sandbox.setTimeout = noop;          // never schedule the W() loop
  sandbox.requestAnimationFrame = noop;
  sandbox.document = {
    getElementById: function () { return makeFakeElement(); },
    addEventListener: noop,
    createElement: function () { return makeFakeElement(); },
  };
  sandbox.headless = true;

  // Simulation parameters (gameopt.js reads these as globals).
  sandbox.lanesSide = LANES_SIDE;
  sandbox.patchesAhead = PATCHES_AHEAD;
  sandbox.patchesBehind = PATCHES_BEHIND;
  sandbox.otherAgents = OTHER_AGENTS;
  sandbox.temporal_window = TEMPORAL;

  // module shim so convnetjs files take the nodejs export branch cleanly,
  // but we ALSO want them attached to the shared global (window===self===sandbox)
  // the way importScripts does. The convnetjs files declare `var convnetjs=...`
  // at top scope and, in node, do `module.exports = lib`. Because each file runs
  // in this same context with `var`, the top-level `var convnetjs` becomes a
  // context global -> shared across files. So we just need `module` to exist
  // (defined as undefined-exports) so they ALSO attach to window. We force the
  // browser branch by leaving module undefined.
  // (Do NOT define `module`: then `typeof module === 'undefined'` is true and
  //  they attach lib to window.* which is our sandbox.)

  return vm.createContext(sandbox);
}

function loadFile(ctx, file) {
  const code = fs.readFileSync(file, 'utf8');
  vm.runInContext(code, ctx, { filename: file });
}

const JS_DIR = path.resolve(__dirname, '..', 'original_js');
const CONV_DIR = path.join(JS_DIR, 'convnetjs');
const SRC_DIR = '/tmp/deeptraffic_src';

function setupEngine() {
  const ctx = buildSandbox();

  // Load ConvNetJS (order matters: convnet -> util -> deepqlearn).
  loadFile(ctx, path.join(CONV_DIR, 'convnet.js'));
  loadFile(ctx, path.join(CONV_DIR, 'util.js'));
  loadFile(ctx, path.join(CONV_DIR, 'deepqlearn.js'));

  // Define brain / learn() depending on the requested policy.
  if (POLICY === 'brain' || opts['code']) {
    // Load a brain definition (defines `brain` + `learn` in global scope and may
    // restore trained weights via brain.value_net.fromJSON). draw_net/draw_stats
    // are stubbed. --code=<path> loads an arbitrary submission; default is the
    // official network_basic.js.
    vm.runInContext('var draw_net=function(){};var draw_stats=function(){};', ctx);
    const brainFile = opts['code']
      ? path.resolve(opts['code'])
      : path.join(SRC_DIR, 'network_basic.js');
    loadFile(ctx, brainFile);
  } else {
    // Fixed-policy: define a `learn(state, lastReward)` returning a constant
    // action. gameopt.js calls learn(K.s(), reward) for the ego every 30 frames.
    // There is no `brain` object -> the engine's `typeof brain` guards skip it.
    let act;
    if (POLICY === 'accel') act = 1;
    else if (POLICY === 'nop') act = 0;
    else if (POLICY === 'decel') act = 2;
    else if (/^\d+$/.test(POLICY)) act = parseInt(POLICY, 10);
    else throw new Error('unknown policy ' + POLICY);
    // gameopt.js's V() unconditionally reads `brain.forward_passes` on its first
    // line. With no brain that throws a ReferenceError. We therefore install a
    // minimal STUB brain whose presence is inert: forward_passes=0 and
    // temporal_window=0 make the brain-cloning guard `forward_passes > tw`
    // (0 > 0) false forever, and reset_seed/learning are no-ops. This stub does
    // NOT participate in action selection -- the fixed `learn()` does.
    vm.runInContext(
      'var draw_net=function(){};var draw_stats=function(){};' +
        'brain={forward_passes:0,temporal_window:0,learning:true,' +
        'reset_seed:function(){}};' +
        'learn=function(state,lastReward){return ' + act + ';};',
      ctx
    );
  }

  // Force the simulation mode (otherAgents) AFTER the brain loads, since a
  // submission's code may hard-set `otherAgents` itself. This lets us evaluate
  // the SAME trained net across 1 / 5 / 11 controlled-car modes.
  if (opts['other-agents'] !== undefined) {
    vm.runInContext('otherAgents = ' + OTHER_AGENTS + ';', ctx);
  }

  // Finally load the engine itself.
  loadFile(ctx, path.join(JS_DIR, 'gameopt.js'));
  return ctx;
}

// ----------------------------------------------------------------------------
// EVAL mode: call the engine's own doEvalRun for an apples-to-apples reference.
// ----------------------------------------------------------------------------
function runEval() {
  const ctx = setupEngine();
  const runs = numOpt('runs', 500);
  const frames = numOpt('frames', 2000);
  const det = boolOpt('det', true);

  // We use runEvalDistribution() — a verbatim copy of doEvalRun's scoring loop —
  // so we can report the FULL distribution (median, mean, min, max). The median
  // returned here is identical to what the engine's own doEvalRun returns
  // (f[Math.floor(runs/2)] after sorting). This is verified by --check-engine.
  vm.runInContext('if(typeof brain!="undefined"){brain.learning=false;}', ctx);

  const t0 = Date.now();
  const dist = runEvalDistribution(ctx, runs, frames, det);
  dist.sort((a, b) => a - b);
  const mean = dist.reduce((s, x) => s + x, 0) / dist.length;
  const median = dist[Math.floor(dist.length / 2)]; // matches doEvalRun's f[a/2]
  const elapsed = (Date.now() - t0) / 1000;

  let engineMedian = null;
  if (boolOpt('check-engine', false)) {
    // Cross-check against the engine's own doEvalRun on a FRESH context (so the
    // v stream is at fresh-load state, matching runEvalDistribution's run 1).
    // doEvalRun's deterministic branch only sets t=r=0 and does not restore v,
    // so reusing the consumed ctx here would diverge on run-1 non-controlled
    // gas; a fresh context avoids that. (doubles runtime)
    const ctx2 = setupEngine();
    vm.runInContext('if(typeof brain!="undefined"){brain.learning=false;}', ctx2);
    ctx2.__cb = function () {};
    engineMedian = vm.runInContext(
      'doEvalRun(' + runs + ',' + frames + ',' + det + ',__cb,' + (runs * frames + 1) + ')',
      ctx2
    );
  }

  console.log(JSON.stringify({
    mode: 'eval', policy: POLICY,
    config: { lanesSide: LANES_SIDE, patchesAhead: PATCHES_AHEAD,
              patchesBehind: PATCHES_BEHIND, otherAgents: OTHER_AGENTS,
              temporalWindow: TEMPORAL },
    runs, frames, deterministic: det,
    median, mean,
    min: dist[0], max: dist[dist.length - 1],
    engine_median: engineMedian,
    elapsed_sec: elapsed,
  }));
}

// Re-implement doEvalRun's scoring loop so we can report the full distribution
// (median + mean). This is a faithful copy of the loop body in gameopt.js so the
// per-run scores are identical to the engine's own doEvalRun. The ENTIRE loop is
// executed inside the vm context in one call (native JS speed); calling V() once
// per frame via separate vm.runInContext calls would be ~1e6 boundary crossings
// and is far too slow.
function runEvalDistribution(ctx, runs, frames, det) {
  ctx.__RUNS = runs;
  ctx.__FRAMES = frames;
  ctx.__DET = det;
  const code = `
    (function(){
      // doEvalRun's seed setup (deterministic vs random).
      if (__DET) {
        t = r = 0;
        if (typeof brain != 'undefined') { brain.reset_seed(0); }
        for (var __d = 0; __d < brains.length; __d++) { brains[__d].reset_seed(0); }
      } else {
        if (typeof brain != 'undefined') { brain.reset_seed(Math.floor(1e7*Math.random())); }
        for (var __d = 0; __d < brains.length; __d++) { brains[__d].reset_seed(Math.floor(1e7*Math.random())); }
        r = Math.floor(1e7*Math.random());
        t = Math.floor(1e7*Math.random());
      }
      headless = true;
      var f = [];
      for (var g = 0; g < __RUNS; g++) {
        reset();
        var O = 0;
        for (var P = 0; P < __FRAMES; P++) {
          V();
          nOtherAgents = Math.floor(nOtherAgents);
          for (var B = 0; B < nOtherAgents + 1; B++) {
            O += Math.max(0, z[B].c * z[B].a) / (nOtherAgents + 1);
          }
        }
        f.push(Math.floor(O / __FRAMES * 2e3) / 100);
      }
      reset();
      return f;
    })()
  `;
  return vm.runInContext(code, ctx);
}

// ----------------------------------------------------------------------------
// DUMP mode: drive the engine with a SCRIPTED ego action sequence (bypassing the
// neural net) and emit per-frame (b, a, c, y) for all 20 cars as JSON. This is
// the ground truth for the trajectory fidelity test.
//
// We override `learn` to pop the next scripted action; everything else (move,
// safety u(), stamp l(), accel, lane-change, RNG) runs exactly as in the engine.
// ----------------------------------------------------------------------------
function runDump() {
  const ctx = setupEngine();
  const frames = numOpt('frames', 200);
  const seedOffset = numOpt('seed-offset', 1); // how many reset()s past r=t=0
  const det = true;

  // Build the scripted ego action list.
  let actions;
  if (opts['actions']) {
    actions = String(opts['actions']).split(',').map((x) => parseInt(x, 10));
  } else {
    // default: repeat a single action based on policy
    let act;
    if (POLICY === 'accel') act = 1;
    else if (POLICY === 'decel') act = 2;
    else act = 0;
    actions = [act];
  }

  // Install a scripted learn() that ignores the state and returns the next action
  // (one per ego decision, i.e. every 30 frames). Cycles if it runs out.
  ctx.__actions = actions;
  ctx.__decIdx = 0;
  vm.runInContext(
    'learn=function(state,lastReward){' +
    '  var a=__actions[__decIdx % __actions.length];' +
    '  __decIdx++;' +
    '  return a;' +
    '};',
    ctx
  );

  // Set deterministic seeds exactly like the eval path: r=t=0, then reset()
  // advances r,t by +1 each call (initializeMap uses u=new p(r), v=new p(r)).
  // We do seedOffset reset()s so different "seeds" produce different layouts,
  // mirroring how doEvalRun iterates runs.
  vm.runInContext('t=r=0;headless=true;', ctx);
  for (let i = 0; i < seedOffset; i++) vm.runInContext('reset();', ctx);

  const out = { mode: 'dump', frames, seedOffset, actions, policy: POLICY,
                config: { lanesSide: LANES_SIDE, patchesAhead: PATCHES_AHEAD,
                          patchesBehind: PATCHES_BEHIND, otherAgents: OTHER_AGENTS },
                states: [] };

  // Capture the INITIAL state (post-reset, pre-frame) too.
  out.initial = captureCars(ctx);

  for (let P = 0; P < frames; P++) {
    vm.runInContext('V();', ctx);
    out.states.push(captureCars(ctx));
  }
  process.stdout.write(JSON.stringify(out));
}

function captureCars(ctx) {
  // Return [b, a, c, y, x] for all 20 cars + globals G,E,N.
  return vm.runInContext(
    '(function(){var r=[];for(var i=0;i<z.length;i++){r.push([z[i].b,z[i].a,z[i].c,z[i].y,z[i].x]);}' +
    'return {cars:r,G:G,E:E,N:N};})()',
    ctx
  );
}

// ----------------------------------------------------------------------------
// OBS mode: dump the .s() observation vector for the ego at frame 0 (for the
// num_inputs / observation fidelity check).
// ----------------------------------------------------------------------------
function runObs() {
  const ctx = setupEngine();
  const seedOffset = numOpt('seed-offset', 1);
  vm.runInContext('t=r=0;headless=true;', ctx);
  for (let i = 0; i < seedOffset; i++) vm.runInContext('reset();', ctx);
  const obs = vm.runInContext('H.o(0,K);K.s();', ctx);
  console.log(JSON.stringify({ mode: 'obs', num_inputs: obs.length, obs }));
}

// ----------------------------------------------------------------------------
switch (mode) {
  case 'eval': runEval(); break;
  case 'dump': runDump(); break;
  case 'obs': runObs(); break;
  default:
    console.error('unknown mode: ' + mode);
    process.exit(1);
}
