/***********************
 * InterDMS Position Identity (fixed)
 * - 500 ms image, 2 s response window
 * - Validates image paths to avoid preload crashes
 ***********************/

import {core, data, util, visual} from './lib/psychojs-2025.1.1.js';

const {PsychoJS} = core;
const {TrialHandler, MultiStairHandler} = data;
const {Scheduler} = util;

// ---- Experiment metadata ----
let expName = 'interdms_position_identity_trials';
let expInfo = {
    workerId: '',           // <— NEW: user will type Worker ID here
    mturkLink: ''            // <— NEW: user will paste MTurk worker page URL here
};
let PILOTING = util.getUrlParameters().has('__pilotToken');


function generateSurveyCode(workerId, assignmentId) {
    // short, deterministic-ish code that’s easy to read/paste
    const base = (workerId || 'W').slice(-4).toUpperCase() + (assignmentId || 'A').slice(-4).toUpperCase();
    const t = Date.now().toString(36).slice(-5).toUpperCase();
    return `${base}-${t}`;
}


// Escape values for CSV building when fallback is used
function _csvEscape(v) {
    if (v === null || v === undefined) return '';
    const s = String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

// Fallback builder if psychoJS.experiment.toCsv() isn't available
function buildCsvFallback() {
    // PsychoJS stores trial rows here in 2024+/2025+; adjust if yours differs
    const rows = (psychoJS?.experiment?._trialsData) || [];
    const keys = new Set();
    rows.forEach(r => Object.keys(r).forEach(k => keys.add(k)));
    const header = Array.from(keys);

    const lines = [];
    lines.push(header.map(_csvEscape).join(','));
    for (const r of rows) {
        lines.push(header.map(k => _csvEscape(r[k])).join(','));
    }
    return lines.join('\n');
}

// Helper to normalize action tokens into null or a lowercase key
function normalizeAction(a) {
    if (a === undefined || a === null) return null;
    if (typeof a !== 'string') return a;
    const s = a.trim().toLowerCase();
    if (s === '' || s === 'none' || s === 'null' || s === 'nan') return null;
    return s;
}

// ===== Cloud upload config =====
const WEB_APP_URL = 'https://script.google.com/macros/s/AKfycbx43V8Aha-JTWTKj51PHQo5SkQztRsV0EYfyAsULh2-NQeFcC1Y8k6wYyhO0_5b_p2amg/exec';

async function uploadCsvToSheets(csv, meta) {
    const res = await fetch(WEB_APP_URL, {
        method: 'POST',
        headers: {'Content-Type': 'text/plain;charset=utf-8'}, // no preflight
        body: JSON.stringify({csv, meta}),
    });
    const json = await res.json().catch(() => ({ok: false, error: 'Bad JSON response'}));
    if (!json.ok) throw new Error(json.error || 'Sheets upload failed');
    return json;
}

function buildCsvFromExperiment() {
    const rows = psychoJS?.experiment?._trialsData || [];
    const keys = new Set();
    rows.forEach(r => Object.keys(r).forEach(k => keys.add(k)));
    const header = Array.from(keys);
    const lines = [header.map(_csvEscape).join(',')];
    for (const r of rows) lines.push(header.map(k => _csvEscape(r[k])).join(','));
    return lines.join('\n');
}

async function finalizeAndSave(psychoJS, expInfo) {
    const csv = buildCsvFromExperiment();
    const meta = {
        workerId: expInfo?.workerId || '',
        assignmentId: expInfo?.assignmentId || '',
        hitId: expInfo?.hitId || '',
        participant: expInfo?.participant || '',
        timestamp: new Date().toISOString(),
    };
    await uploadCsvToSheets(csv, meta);
    console.log('Sheets upload OK');
    console.log('[Upload meta]', {
        workerId: expInfo?.workerId,
        assignmentId: expInfo?.assignmentId,
        hitId: expInfo?.hitId
    });

}

const ASSETS_DIR = 'resources';
const IMG_DIR = `${ASSETS_DIR}/images`;
const FALLBACK = `${IMG_DIR}/157_Chairs.png`;

// --- Resource preload (collect from CSV + add fallback) ---
const TRIALS_CSV = 'resources/interdms_position_identity_trials.csv';
const ALWAYS_RESOURCES = ['resources/images/157_Chairs.png']; // fallback

async function collectImagePathsFromCSV(csvPath) {
    const res = await fetch(csvPath, {cache: 'no-store'});
    if (!res.ok) throw new Error(`Failed to load ${csvPath}: ${res.status}`);
    // Handle BOM + normalize line endings
    let text = await res.text();
    if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);

    const lines = text
        .split(/\r?\n/)
        .map(l => l.trim())
        .filter(l => l.length);

    const header = lines[0]
        .split(',')
        .map(s => s.trim().toLowerCase().replace(/^"|"$/g, '')); // strip quotes

    const stimCols = header
        .map((h, i) => ({h, i}))
        .filter(({h}) => /^stim[1-6]$/.test(h))
        .map(({i}) => i);

    const paths = new Set(ALWAYS_RESOURCES);
    for (let r = 1; r < lines.length; r++) {
        const cols = lines[r].split(',').map(s => s.trim().replace(/^"|"$/g, ''));
        for (const ci of stimCols) {
            const raw = cols[ci];
            if (raw && raw.toLowerCase() !== 'default.png') paths.add(raw);
        }
    }
    return Array.from(paths);
}

function normPath(p) {
    if (p == null) return '';
    p = String(p).trim();
    if (!p) return '';
    if (/^https?:\/\//i.test(p)) return p;
    if (p.startsWith('resources/')) return p;
    if (p.startsWith('images/')) return `resources/${p}`;
    if (!p.includes('/')) return `resources/images/${p}`;
    return `resources/${p}`;
}

// ---- init psychoJS ----
const psychoJS = new PsychoJS({debug: true});

// ---- open window ----
psychoJS.openWindow({
    fullscr: false,
    color: new util.Color([0, 0, 0]),
    units: 'height',
    waitBlanking: true,
    backgroundImage: '',
    backgroundFit: 'none',
});

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);

(async function bootstrap() {
    const csvPathsRaw = await collectImagePathsFromCSV(TRIALS_CSV);

    const allPaths = Array.from(new Set([
        'resources/images/157_Chairs.png',
        ...csvPathsRaw.map(normPath)
    ])).filter(p => typeof p === 'string' && p.length > 0);

    const resources = [
        {name: TRIALS_CSV, path: TRIALS_CSV},
        ...allPaths.map(p => ({name: p, path: p}))
    ].filter(r => typeof r.name === 'string' && r.name && typeof r.path === 'string' && r.path);

    // Optional: sanity check in the console
    console.log('[Preload] resources count:', resources.length);
    const bad = resources.filter(r => !r.name || !r.path);
    if (bad.length) {
        console.error('[Preload] Found invalid resources:', bad);
        throw new Error('Invalid resource entries detected.');
    }

    psychoJS.schedule(psychoJS.gui.DlgFromDict({dictionary: expInfo, title: expName}));
    psychoJS.scheduleCondition(
        () => (psychoJS.gui.dialogComponent.button === 'OK'),
        flowScheduler,
        dialogCancelScheduler
    );

    flowScheduler.add(updateInfo);
    flowScheduler.add(experimentInit);

    // Welcome/instructions
    flowScheduler.add(WelcomeRoutineBegin());
    flowScheduler.add(WelcomeRoutineEachFrame());
    flowScheduler.add(WelcomeRoutineEnd());

    flowScheduler.add(GlobalsRoutineBegin());
    flowScheduler.add(GlobalsRoutineEachFrame());
    flowScheduler.add(GlobalsRoutineEnd());
    const trialsLoopScheduler = new Scheduler(psychoJS);

    flowScheduler.add(trialsLoopBegin(trialsLoopScheduler));
    flowScheduler.add(trialsLoopScheduler);
    flowScheduler.add(trialsLoopEnd);
    flowScheduler.add(ITIRoutineBegin());
    flowScheduler.add(ITIRoutineEachFrame());
    flowScheduler.add(ITIRoutineEnd());
    flowScheduler.add(ThanksRoutineBegin());
    flowScheduler.add(ThanksRoutineEachFrame());
    flowScheduler.add(ThanksRoutineEnd());
    flowScheduler.add(quitPsychoJS, '', true);

    dialogCancelScheduler.add(quitPsychoJS, '', false);

    // Start WITH resources so they’re preloaded
    psychoJS.start({expName, expInfo, resources});
    psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.EXP);
})();


// =========================
// Globals & helpers
// =========================
var currentLoop;
var frameDur;
// --- End/Thanks routine ---
var ThanksClock;
var thanksText;
var thanksKey;
var ThanksMaxDurationReached;
var _thanksKey_allKeys;
var ThanksMaxDuration;
var ThanksComponents;

function ThanksRoutineBegin(snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        t = 0;
        frameN = -1;
        continueRoutine = true;
        routineForceEnded = false;

        ThanksClock.reset();
        routineTimer.reset();
        ThanksMaxDurationReached = false;

        thanksKey.keys = undefined;
        thanksKey.rt = undefined;
        _thanksKey_allKeys = [];

        psychoJS.experiment.addData('Thanks.started', globalClock.getTime());
        ThanksMaxDuration = null;

        ThanksComponents = [thanksText, thanksKey];
        ThanksComponents.forEach(c => {
            if ('status' in c) c.status = PsychoJS.Status.NOT_STARTED;
        });

        return Scheduler.Event.NEXT;
    };
}

function ThanksRoutineEachFrame() {
    return async function () {
        t = ThanksClock.getTime();
        frameN += 1;

        if (t >= 0.0 && thanksText.status === PsychoJS.Status.NOT_STARTED) {
            thanksText.tStart = t;
            thanksText.frameNStart = frameN;
            thanksText.setAutoDraw(true);
        }

        if (t >= 0.0 && thanksKey.status === PsychoJS.Status.NOT_STARTED) {
            thanksKey.tStart = t;
            thanksKey.frameNStart = frameN;
            psychoJS.window.callOnFlip(() => {
                thanksKey.clock.reset();
            });
            psychoJS.window.callOnFlip(() => {
                thanksKey.start();
            });
            psychoJS.window.callOnFlip(() => {
                thanksKey.clearEvents();
            });
        }

        if (thanksKey.status === PsychoJS.Status.STARTED) {
            // accept Enter/Return (and Space as a friendly fallback)
            let theseKeys = thanksKey.getKeys({keyList: ['return', 'enter', 'space'], waitRelease: false});
            _thanksKey_allKeys = _thanksKey_allKeys.concat(theseKeys);
            if (_thanksKey_allKeys.length > 0) {
                const last = _thanksKey_allKeys[_thanksKey_allKeys.length - 1];
                thanksKey.keys = last.name;
                thanksKey.rt = last.rt;
                thanksKey.duration = last.duration;
                // Immediately stop listening to prevent duplicate captures
                thanksKey.stop();
                thanksKey.status = PsychoJS.Status.FINISHED;
                continueRoutine = false;
            }
        }

        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList: ['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }

        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        continueRoutine = false;
        ThanksComponents.forEach(c => {
            if ('status' in c && c.status !== PsychoJS.Status.FINISHED) continueRoutine = true;
        });

        if (continueRoutine) return Scheduler.Event.FLIP_REPEAT;
        return Scheduler.Event.NEXT;
    };
}

function ThanksRoutineEnd(snapshot) {
    return async function () {
        ThanksComponents.forEach(c => {
            if (typeof c.setAutoDraw === 'function') c.setAutoDraw(false);
        });
        psychoJS.experiment.addData('Thanks.stopped', globalClock.getTime());

        psychoJS.experiment.addData('thanksKey.keys', thanksKey.keys);
        if (typeof thanksKey.keys !== 'undefined') {
            psychoJS.experiment.addData('thanksKey.rt', thanksKey.rt);
            psychoJS.experiment.addData('thanksKey.duration', thanksKey.duration);
            routineTimer.reset();
        }

        thanksKey.stop();
        routineTimer.reset();

        if (currentLoop === psychoJS.experiment) psychoJS.experiment.nextEntry(snapshot);
        return Scheduler.Event.NEXT;
    };
}


async function updateInfo() {
    currentLoop = psychoJS.experiment;
    expInfo['date'] = util.MonotonicClock.getDateStr();
    expInfo['expName'] = expName;
    expInfo['psychopyVersion'] = '2025.1.1';
    expInfo['OS'] = window.navigator.platform;

    expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
    frameDur = (typeof expInfo['frameRate'] !== 'undefined')
        ? 1.0 / Math.round(expInfo['frameRate'])
        : 1.0 / 60.0;

    // This may import URL params into expInfo
    util.addInfoFromUrl(expInfo);


    return Scheduler.Event.NEXT;
}


// =========================
var WelcomeClock;
var welcomeText;
var welcomeKey;
var WelcomeMaxDurationReached;
var _welcomeKey_allKeys;
var WelcomeMaxDuration;
var WelcomeComponents;

var GlobalsClock;
var prevSession;
var frameIdx;
var stimPaths;
var actKeys;

var SessionIntroClock;
var sessText;
var sessKey;

var TrialIntroClock;
var trialText;
var trialKey;

var FrameClock;
var img;
var resp;

var ITIClock;
var itiText;
var itiKey;

// local ITI helper clock (no globals() magic)
var itiClock;

var globalClock;
var routineTimer;

async function experimentInit() {
    // ---- Globals Routine ----
    GlobalsClock = new util.Clock();
    prevSession = null;
    frameIdx = -1;
    stimPaths = [];
    actKeys = [];

    // ---- SessionIntro ----
    SessionIntroClock = new util.Clock();
    sessText = new visual.TextStim({
        win: psychoJS.window,
        name: 'sessText',
        text: 'Press Enter to start',
        font: 'Open Sans',
        pos: [0, 0], draggable: false, height: 0.05,
        languageStyle: 'LTR',
        color: new util.Color('white'),
        depth: 0.0
    });
    sessKey = new core.Keyboard({psychoJS, clock: new util.Clock(), waitForStart: true});

    // ---- TrialIntro ----
    TrialIntroClock = new util.Clock();
    trialText = new visual.TextStim({
        win: psychoJS.window,
        name: 'trialText',
        text: 'Press Enter to start the trial',
        font: 'Open Sans',
        pos: [0, 0], draggable: false, height: 0.05,
        languageStyle: 'LTR',
        color: new util.Color('white'),
        depth: 0.0
    });
    trialKey = new core.Keyboard({psychoJS, clock: new util.Clock(), waitForStart: true});

    // ---- Frame ----
    FrameClock = new util.Clock();
    img = new visual.ImageStim({
        win: psychoJS.window,
        name: 'img',
        image: 'resources/images/157_Chairs.png',
        anchor: 'center',
        ori: 0.0,
        pos: [0, 0],
        draggable: false,
        size: [0.5, 0.5],
        color: new util.Color([1, 1, 1]),
        interpolate: true,
        depth: 0.0
    });

    resp = new core.Keyboard({psychoJS, clock: new util.Clock(), waitForStart: true});

    // ---- ITI ----
    ITIClock = new util.Clock();
    itiText = new visual.TextStim({
        win: psychoJS.window,
        name: 'itiText',
        text: 'Next trial in 5s - press Enter to start now',
        font: 'Open Sans',
        pos: [0, 0], draggable: false, height: 0.05,
        languageStyle: 'LTR',
        color: new util.Color('white'),
        depth: 0.0
    });
    itiKey = new core.Keyboard({psychoJS, clock: new util.Clock(), waitForStart: true});

    // Local ITI timer (we manage it ourselves)
    itiClock = new util.Clock();

    // ---- Welcome (instructions) ----
    WelcomeClock = new util.Clock();
    welcomeText = new visual.TextStim({
        win: psychoJS.window,
        name: 'welcomeText',
        text:
            `
    Task instructions:

    • You will perform an Inter(-leaved)DMS ABAB Position–Identity task.

    • Each trial consists of a sequence of  4 images. You will make two comparisons:

      1) Location match (3rd vs 1st frame):
         – Compare the 3rd frame with the 1st frame.
         – If the stimulus appears in the same LOCATION, press 'X' as soon as you see the 3rd frame.
         – Otherwise, press 'B'.

      2) Identity match (4th vs 2nd frame):
         – Compare the 4th frame with the 2nd frame.
         – If the object is the same (for example, the same plane), press 'X' as soon as you see the 4th frame.
         – Otherwise, press 'B'.

    • Each image is shown for 500 ms, followed by a 2-second response window in which you can press 'X' or 'B'.

    • There are 5 sessions of 20 trials each. You can take short breaks between sessions.

    Press Enter to start.`,
        font: 'Open Sans',

        // ↓↓↓ layout bits ↓↓↓
        pos: [0, 0],
        anchor: 'center',
        alignHoriz: 'center',
        alignVert: 'center',
        height: 0.028,
        wrapWidth: 1.3,
        // ↑↑↑ layout bits ↑↑↑

        draggable: false,
        color: new util.Color('white'),
        depth: 0.0
    });
    welcomeKey = new core.Keyboard({psychoJS, clock: new util.Clock(), waitForStart: true});

    // ---- timers ----
    globalClock = new util.Clock();
    routineTimer = new util.CountdownTimer();

    // ---- Thanks (end screen) ----
    ThanksClock = new util.Clock();
    thanksText = new visual.TextStim({
        win: psychoJS.window,
        name: 'thanksText',
        text:
            `Thank you for participating!

Please submit the following survey code on the MTurk page before submit: AWDR

Press Enter to finish.

Important: after you press Enter, do not close this page until the data file has
finished uploading (this may take a few seconds / you might have to press Enter multiple times). Please wait until you see next page`,
        font: 'Open Sans',
        pos: [0, 0], draggable: false, height: 0.05,
        color: new util.Color('white'),
        wrapWidth: 1.2,
        depth: 0.0
    });
    let FINAL_SURVEY_CODE = '';

    thanksKey = new core.Keyboard({psychoJS, clock: new util.Clock(), waitForStart: true});

    return Scheduler.Event.NEXT;
}

// =========================
// Routines
// =========================
var t;
var frameN;
var continueRoutine;
var routineForceEnded;

var GlobalsMaxDurationReached;
var GlobalsMaxDuration;
var GlobalsComponents;

function GlobalsRoutineBegin(snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        t = 0;
        frameN = -1;
        continueRoutine = true;
        routineForceEnded = false;

        GlobalsClock.reset();
        routineTimer.reset();
        GlobalsMaxDurationReached = false;

        psychoJS.experiment.addData('Globals.started', globalClock.getTime());
        GlobalsMaxDuration = null;

        GlobalsComponents = [];
        GlobalsComponents.forEach(c => {
            if ('status' in c) c.status = PsychoJS.Status.NOT_STARTED;
        });

        return Scheduler.Event.NEXT;
    };
}

function GlobalsRoutineEachFrame() {
    return async function () {
        t = GlobalsClock.getTime();
        frameN += 1;

        // quit?
        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList: ['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }

        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        continueRoutine = false;
        GlobalsComponents.forEach(c => {
            if ('status' in c && c.status !== PsychoJS.Status.FINISHED)
                continueRoutine = true;
        });

        if (continueRoutine) return Scheduler.Event.FLIP_REPEAT;
        return Scheduler.Event.NEXT;
    };
}

function GlobalsRoutineEnd(snapshot) {
    return async function () {
        GlobalsComponents.forEach(c => {
            if (typeof c.setAutoDraw === 'function') c.setAutoDraw(false);
        });
        psychoJS.experiment.addData('Globals.stopped', globalClock.getTime());
        routineTimer.reset();

        if (currentLoop === psychoJS.experiment) psychoJS.experiment.nextEntry(snapshot);
        return Scheduler.Event.NEXT;
    };
}

// ------------- Trials (outer loop) -------------
var trials;

function trialsLoopBegin(trialsLoopScheduler, snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        trials = new TrialHandler({
            psychoJS,
            nReps: 1, method: TrialHandler.Method.SEQUENTIAL,
            extraInfo: expInfo, originPath: undefined,
            trialList: 'resources/interdms_position_identity_trials.csv',
            seed: undefined, name: 'trials'
        });
        psychoJS.experiment.addLoop(trials);
        currentLoop = trials;

        trials.forEach(function () {
            snapshot = trials.getSnapshot();

            trialsLoopScheduler.add(importConditions(snapshot));
            trialsLoopScheduler.add(SessionIntroRoutineBegin(snapshot));
            trialsLoopScheduler.add(SessionIntroRoutineEachFrame());
            trialsLoopScheduler.add(SessionIntroRoutineEnd(snapshot));

            trialsLoopScheduler.add(TrialIntroRoutineBegin(snapshot));
            trialsLoopScheduler.add(TrialIntroRoutineEachFrame());
            trialsLoopScheduler.add(TrialIntroRoutineEnd(snapshot));

            const framesLoopScheduler = new Scheduler(psychoJS);
            trialsLoopScheduler.add(framesLoopBegin(framesLoopScheduler, snapshot));
            trialsLoopScheduler.add(framesLoopScheduler);
            trialsLoopScheduler.add(framesLoopEnd);

            trialsLoopScheduler.add(trialsLoopEndIteration(trialsLoopScheduler, snapshot));
        });

        return Scheduler.Event.NEXT;
    };
}

async function trialsLoopEnd() {
    psychoJS.experiment.removeLoop(trials);
    if (psychoJS.experiment._unfinishedLoops.length > 0)
        currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
    else
        currentLoop = psychoJS.experiment;
    return Scheduler.Event.NEXT;
}

function trialsLoopEndIteration(scheduler, snapshot) {
    return async function () {
        if (typeof snapshot !== 'undefined') {
            if (snapshot.finished) {
                if (psychoJS.experiment.isEntryEmpty()) {
                    psychoJS.experiment.nextEntry(snapshot);
                }
                scheduler.stop();
            } else {
                psychoJS.experiment.nextEntry(snapshot);
            }
            return Scheduler.Event.NEXT;
        }
    };
}

// ------------- Frames (inner loop) -------------
var frames;

function framesLoopBegin(framesLoopScheduler, snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        frames = new TrialHandler({
            psychoJS,
            nReps: 4, method: TrialHandler.Method.SEQUENTIAL,
            extraInfo: expInfo, originPath: undefined,
            trialList: undefined,
            seed: undefined, name: 'frames'
        });
        psychoJS.experiment.addLoop(frames);
        currentLoop = frames;

        frames.forEach(function () {
            snapshot = frames.getSnapshot();

            framesLoopScheduler.add(importConditions(snapshot));
            framesLoopScheduler.add(FrameRoutineBegin(snapshot));
            framesLoopScheduler.add(FrameRoutineEachFrame());
            framesLoopScheduler.add(FrameRoutineEnd(snapshot));
            framesLoopScheduler.add(framesLoopEndIteration(framesLoopScheduler, snapshot));
        });

        return Scheduler.Event.NEXT;
    };
}

function WelcomeRoutineBegin(snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        t = 0;
        frameN = -1;
        continueRoutine = true;
        routineForceEnded = false;
        WelcomeClock.reset();
        routineTimer.reset();
        WelcomeMaxDurationReached = false;

        welcomeKey.keys = undefined;
        welcomeKey.rt = undefined;
        _welcomeKey_allKeys = [];

        psychoJS.experiment.addData('Welcome.started', globalClock.getTime());
        WelcomeMaxDuration = null;

        WelcomeComponents = [welcomeText, welcomeKey];
        WelcomeComponents.forEach(c => {
            if ('status' in c) c.status = PsychoJS.Status.NOT_STARTED;
        });

        return Scheduler.Event.NEXT;
    };
}

function WelcomeRoutineEachFrame() {
    return async function () {
        t = WelcomeClock.getTime();
        frameN += 1;

        if (t >= 0.0 && welcomeText.status === PsychoJS.Status.NOT_STARTED) {
            welcomeText.tStart = t;
            welcomeText.frameNStart = frameN;
            welcomeText.setAutoDraw(true);
        }

        if (t >= 0.0 && welcomeKey.status === PsychoJS.Status.NOT_STARTED) {
            welcomeKey.tStart = t;
            welcomeKey.frameNStart = frameN;
            psychoJS.window.callOnFlip(() => {
                welcomeKey.clock.reset();
            });
            psychoJS.window.callOnFlip(() => {
                welcomeKey.start();
            });
            psychoJS.window.callOnFlip(() => {
                welcomeKey.clearEvents();
            });
        }

        if (welcomeKey.status === PsychoJS.Status.STARTED) {
            let theseKeys = welcomeKey.getKeys({keyList: ['return', 'enter'], waitRelease: false});
            _welcomeKey_allKeys = _welcomeKey_allKeys.concat(theseKeys);
            if (_welcomeKey_allKeys.length > 0) {
                const last = _welcomeKey_allKeys[_welcomeKey_allKeys.length - 1];
                welcomeKey.keys = last.name;
                welcomeKey.rt = last.rt;
                welcomeKey.duration = last.duration;
                continueRoutine = false;
            }
        }

        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList: ['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }

        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        continueRoutine = false;
        WelcomeComponents.forEach(c => {
            if ('status' in c && c.status !== PsychoJS.Status.FINISHED) continueRoutine = true;
        });

        if (continueRoutine) return Scheduler.Event.FLIP_REPEAT;
        return Scheduler.Event.NEXT;
    };
}

function WelcomeRoutineEnd(snapshot) {
    return async function () {
        WelcomeComponents.forEach(c => {
            if (typeof c.setAutoDraw === 'function') c.setAutoDraw(false);
        });
        psychoJS.experiment.addData('Welcome.stopped', globalClock.getTime());

        if (currentLoop instanceof MultiStairHandler) currentLoop.addResponse(welcomeKey.corr, level);
        psychoJS.experiment.addData('welcomeKey.keys', welcomeKey.keys);
        if (typeof welcomeKey.keys !== 'undefined') {
            psychoJS.experiment.addData('welcomeKey.rt', welcomeKey.rt);
            psychoJS.experiment.addData('welcomeKey.duration', welcomeKey.duration);
            routineTimer.reset();
        }
        welcomeKey.stop();
        routineTimer.reset();

        if (currentLoop === psychoJS.experiment) psychoJS.experiment.nextEntry(snapshot);
        return Scheduler.Event.NEXT;
    };
}

async function framesLoopEnd() {
    psychoJS.experiment.removeLoop(frames);
    if (psychoJS.experiment._unfinishedLoops.length > 0)
        currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
    else
        currentLoop = psychoJS.experiment;
    return Scheduler.Event.NEXT;
}

function framesLoopEndIteration(scheduler, snapshot) {
    return async function () {
        if (typeof snapshot !== 'undefined') {
            if (snapshot.finished) {
                if (psychoJS.experiment.isEntryEmpty()) {
                    psychoJS.experiment.nextEntry(snapshot);
                }
                scheduler.stop();
            } else {
                psychoJS.experiment.nextEntry(snapshot);
            }
            return Scheduler.Event.NEXT;
        }
    };
}

// =========================
// SessionIntro
// =========================
var SessionIntroMaxDurationReached;
var _sessKey_allKeys;
var SessionIntroMaxDuration;
var SessionIntroComponents;

function SessionIntroRoutineBegin(snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        t = 0;
        frameN = -1;
        continueRoutine = true;
        routineForceEnded = false;

        SessionIntroClock.reset();
        routineTimer.reset();
        SessionIntroMaxDurationReached = false;

        sessKey.keys = undefined;
        sessKey.rt = undefined;
        _sessKey_allKeys = [];

        // show only when session changes or at the very start
        if ((typeof prevSession === 'undefined') || (prevSession === null) || (session !== prevSession)) {
            continueRoutine = true;
            sessText.setText(`Session: ${session}\nPress Enter to start`);
        } else {
            continueRoutine = false;
        }

        psychoJS.experiment.addData('SessionIntro.started', globalClock.getTime());
        SessionIntroMaxDuration = null;

        SessionIntroComponents = [sessText, sessKey];
        SessionIntroComponents.forEach(c => {
            if ('status' in c) c.status = PsychoJS.Status.NOT_STARTED;
        });

        return Scheduler.Event.NEXT;
    };
}

function SessionIntroRoutineEachFrame() {
    return async function () {
        t = SessionIntroClock.getTime();
        frameN += 1;

        if (t >= 0.0 && sessText.status === PsychoJS.Status.NOT_STARTED) {
            sessText.tStart = t;
            sessText.frameNStart = frameN;
            sessText.setAutoDraw(true);
        }

        if (t >= 0.0 && sessKey.status === PsychoJS.Status.NOT_STARTED) {
            sessKey.tStart = t;
            sessKey.frameNStart = frameN;
            psychoJS.window.callOnFlip(() => {
                sessKey.clock.reset();
            });
            psychoJS.window.callOnFlip(() => {
                sessKey.start();
            });
            psychoJS.window.callOnFlip(() => {
                sessKey.clearEvents();
            });
        }

        if (sessKey.status === PsychoJS.Status.STARTED) {
            let theseKeys = sessKey.getKeys({keyList: ['return', 'enter', 'space'], waitRelease: false});
            _sessKey_allKeys = _sessKey_allKeys.concat(theseKeys);
            if (_sessKey_allKeys.length > 0) {
                const last = _sessKey_allKeys[_sessKey_allKeys.length - 1];
                sessKey.keys = last.name;
                sessKey.rt = last.rt;
                sessKey.duration = last.duration;
                continueRoutine = false;
            }
        }

        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList: ['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }

        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        continueRoutine = false;
        SessionIntroComponents.forEach(c => {
            if ('status' in c && c.status !== PsychoJS.Status.FINISHED) continueRoutine = true;
        });

        if (continueRoutine) return Scheduler.Event.FLIP_REPEAT;
        return Scheduler.Event.NEXT;
    };
}

function SessionIntroRoutineEnd(snapshot) {
    return async function () {
        SessionIntroComponents.forEach(c => {
            if (typeof c.setAutoDraw === 'function') c.setAutoDraw(false);
        });
        psychoJS.experiment.addData('SessionIntro.stopped', globalClock.getTime());

        if (currentLoop instanceof MultiStairHandler) currentLoop.addResponse(sessKey.corr, level);
        psychoJS.experiment.addData('sessKey.keys', sessKey.keys);
        if (typeof sessKey.keys !== 'undefined') {
            psychoJS.experiment.addData('sessKey.rt', sessKey.rt);
            psychoJS.experiment.addData('sessKey.duration', sessKey.duration);
            routineTimer.reset();
        }

        sessKey.stop();
        prevSession = session;

        routineTimer.reset();
        if (currentLoop === psychoJS.experiment) psychoJS.experiment.nextEntry(snapshot);
        return Scheduler.Event.NEXT;
    };
}

// =========================
// TrialIntro
// =========================
var TrialIntroMaxDurationReached;
var _trialKey_allKeys;
var TrialIntroMaxDuration;
var TrialIntroComponents;

function TrialIntroRoutineBegin(snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        t = 0;
        frameN = -1;
        continueRoutine = true;
        routineForceEnded = false;

        TrialIntroClock.reset();
        routineTimer.reset();
        TrialIntroMaxDurationReached = false;

        trialKey.keys = undefined;
        trialKey.rt = undefined;
        _trialKey_allKeys = [];

        // --- Prepare stim & expected keys for the 6 frames ---
        frameIdx = -1;

        stimPaths = [stim1, stim2, stim3, stim4];
        // act1 unused; allow nulls in the rest
        actKeys = [
            null, // align indices with frameIdx if you start counting at 1 later
            null,
            normalizeAction(act3),
            normalizeAction(act4),
        ];

        psychoJS.experiment.addData('TrialIntro.started', globalClock.getTime());
        TrialIntroMaxDuration = null;

        TrialIntroComponents = [trialText, trialKey];
        TrialIntroComponents.forEach(c => {
            if ('status' in c) c.status = PsychoJS.Status.NOT_STARTED;
        });

        return Scheduler.Event.NEXT;
    };
}

function TrialIntroRoutineEachFrame() {
    return async function () {
        t = TrialIntroClock.getTime();
        frameN += 1;

        if (t >= 0.0 && trialText.status === PsychoJS.Status.NOT_STARTED) {
            trialText.tStart = t;
            trialText.frameNStart = frameN;
            trialText.setAutoDraw(true);
        }

        if (t >= 0.0 && trialKey.status === PsychoJS.Status.NOT_STARTED) {
            trialKey.tStart = t;
            trialKey.frameNStart = frameN;
            psychoJS.window.callOnFlip(() => {
                trialKey.clock.reset();
            });
            psychoJS.window.callOnFlip(() => {
                trialKey.start();
            });
            psychoJS.window.callOnFlip(() => {
                trialKey.clearEvents();
            });
        }

        if (trialKey.status === PsychoJS.Status.STARTED) {
            let theseKeys = trialKey.getKeys({keyList: ['return', 'enter', 'space'], waitRelease: false});
            _trialKey_allKeys = _trialKey_allKeys.concat(theseKeys);
            if (_trialKey_allKeys.length > 0) {
                const last = _trialKey_allKeys[_trialKey_allKeys.length - 1];
                trialKey.keys = last.name;
                trialKey.rt = last.rt;
                trialKey.duration = last.duration;
                continueRoutine = false;
            }
        }

        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList: ['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }

        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        continueRoutine = false;
        TrialIntroComponents.forEach(c => {
            if ('status' in c && c.status !== PsychoJS.Status.FINISHED) continueRoutine = true;
        });

        if (continueRoutine) return Scheduler.Event.FLIP_REPEAT;
        return Scheduler.Event.NEXT;
    };
}

function TrialIntroRoutineEnd(snapshot) {
    return async function () {
        TrialIntroComponents.forEach(c => {
            if (typeof c.setAutoDraw === 'function') c.setAutoDraw(false);
        });
        psychoJS.experiment.addData('TrialIntro.stopped', globalClock.getTime());

        if (currentLoop instanceof MultiStairHandler) currentLoop.addResponse(trialKey.corr, level);
        psychoJS.experiment.addData('trialKey.keys', trialKey.keys);
        if (typeof trialKey.keys !== 'undefined') {
            psychoJS.experiment.addData('trialKey.rt', trialKey.rt);
            psychoJS.experiment.addData('trialKey.duration', trialKey.duration);
            routineTimer.reset();
        }

        trialKey.stop();
        routineTimer.reset();

        if (currentLoop === psychoJS.experiment) psychoJS.experiment.nextEntry(snapshot);
        return Scheduler.Event.NEXT;
    };
}

// =========================
// Frame (core 500ms image + 2s response)
// =========================
var FrameMaxDurationReached;
var _resp_allKeys;
var FrameMaxDuration;
var FrameComponents;
var currStim;
var expected;

function FrameRoutineBegin(snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        t = 0;
        frameN = -1;
        continueRoutine = true;
        routineForceEnded = false;

        FrameClock.reset(routineTimer.getTime());
        routineTimer.add(2.500000); // 0.5s image + 2.0s response = 2.5s total
        FrameMaxDurationReached = false;

        resp.keys = undefined;
        resp.rt = undefined;
        _resp_allKeys = [];

        // ---- Begin Routine: safe prep of current frame ----
        frameIdx += 1;

        // guard against out-of-range
        if (frameIdx >= stimPaths.length) {
            continueRoutine = false;
        } else {
            const rawStim = stimPaths[frameIdx];
            const fallbackStim = 'resources/images/157_Chairs.png';
            currStim = normPath(rawStim) || fallbackStim;
            img.setImage(currStim);

            psychoJS.eventManager.clearEvents();
        }
        // -----------------------------------------------

        psychoJS.experiment.addData('Frame.started', globalClock.getTime());
        FrameMaxDuration = null;

        FrameComponents = [img, resp];
        FrameComponents.forEach(c => {
            if ('status' in c) c.status = PsychoJS.Status.NOT_STARTED;
        });

        return Scheduler.Event.NEXT;
    };
}

var frameRemains;

function FrameRoutineEachFrame() {
    return async function () {
        t = FrameClock.getTime();
        frameN += 1;

        // image on for 0.5s
        if (t >= 0.0 && img.status === PsychoJS.Status.NOT_STARTED) {
            img.tStart = t;
            img.frameNStart = frameN;
            img.setAutoDraw(true);
        }
        frameRemains = 0.0 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;
        if (img.status === PsychoJS.Status.STARTED && t >= frameRemains) {
            img.tStop = t;
            img.frameNStop = frameN;
            img.status = PsychoJS.Status.FINISHED;
            img.setAutoDraw(false);
        }

        // response active during the 2.0s delay ONLY on frames 3 and 4
        // frameIdx: 0 → 1st frame, 1 → 2nd, 2 → 3rd, 3 → 4th
        if (frameIdx === 2 || frameIdx === 3) {

            if (t >= 0.0 && resp.status === PsychoJS.Status.NOT_STARTED) {
                resp.tStart = t;
                resp.frameNStart = frameN;
                psychoJS.window.callOnFlip(() => {
                    resp.clock.reset();
                });
                psychoJS.window.callOnFlip(() => {
                    resp.start();
                });
                psychoJS.window.callOnFlip(() => {
                    resp.clearEvents();
                });
            }

            frameRemains = 0.0 + 2.5 - psychoJS.window.monitorFramePeriod * 0.75;
            if (resp.status === PsychoJS.Status.STARTED && t >= frameRemains) {
                resp.tStop = t;
                resp.frameNStop = frameN;
                resp.status = PsychoJS.Status.FINISHED;
            }

            if (resp.status === PsychoJS.Status.STARTED) {
                let theseKeys = resp.getKeys({keyList: ['x', 'b'], waitRelease: false});
                _resp_allKeys = _resp_allKeys.concat(theseKeys);
                if (_resp_allKeys.length > 0) {
                    const last = _resp_allKeys[_resp_allKeys.length - 1];
                    resp.keys = last.name;
                    resp.rt = last.rt;
                    resp.duration = last.duration;
                }
            }
        }


        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList: ['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }

        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        continueRoutine = false;
        if (routineTimer.getTime() > 0) {
            FrameComponents.forEach(c => {
                if ('status' in c && c.status !== PsychoJS.Status.FINISHED) continueRoutine = true;
            });
        }

        if (continueRoutine && routineTimer.getTime() > 0) return Scheduler.Event.FLIP_REPEAT;
        return Scheduler.Event.NEXT;
    };
}

function FrameRoutineEnd(snapshot) {
    return async function () {
        FrameComponents.forEach(c => {
            if (typeof c.setAutoDraw === 'function') c.setAutoDraw(false);
        });
        psychoJS.experiment.addData('Frame.stopped', globalClock.getTime());

        // Only log responses for frame 3 and 4
        if (frameIdx === 2 || frameIdx === 3) {
            if (currentLoop instanceof MultiStairHandler) currentLoop.addResponse(resp.corr, level);

            psychoJS.experiment.addData('resp.keys', resp.keys);
            if (typeof resp.keys !== 'undefined') {
                psychoJS.experiment.addData('resp.rt', resp.rt);
                psychoJS.experiment.addData('resp.duration', resp.duration);
            }
        }

        resp.stop();


        if (routineForceEnded) {
            routineTimer.reset();
        } else if (FrameMaxDurationReached) {
            FrameClock.add(FrameMaxDuration);
        } else {
            FrameClock.add(2.500000);
        }

        if (currentLoop === psychoJS.experiment) psychoJS.experiment.nextEntry(snapshot);
        return Scheduler.Event.NEXT;
    };
}

// =========================
/** ITI: 5s or press Enter/Space/Return to skip **/
// =========================
var ITIMaxDurationReached;
var _itiKey_allKeys;
var ITIMaxDuration;
var ITIComponents;

function ITIRoutineBegin(snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        t = 0;
        frameN = -1;
        continueRoutine = true;
        routineForceEnded = false;

        ITIClock.reset();
        routineTimer.reset();
        ITIMaxDurationReached = false;

        itiKey.keys = undefined;
        itiKey.rt = undefined;
        _itiKey_allKeys = [];

        // local clock for 5s countdown
        itiClock.reset();

        psychoJS.experiment.addData('ITI.started', globalClock.getTime());
        ITIMaxDuration = null;

        ITIComponents = [itiText, itiKey];
        ITIComponents.forEach(c => {
            if ('status' in c) c.status = PsychoJS.Status.NOT_STARTED;
        });

        return Scheduler.Event.NEXT;
    };
}

function ITIRoutineEachFrame() {
    return async function () {
        t = ITIClock.getTime();
        frameN += 1;

        if (t >= 0.0 && itiText.status === PsychoJS.Status.NOT_STARTED) {
            itiText.tStart = t;
            itiText.frameNStart = frameN;
            itiText.setAutoDraw(true);
        }

        if (t >= 0.0 && itiKey.status === PsychoJS.Status.NOT_STARTED) {
            itiKey.tStart = t;
            itiKey.frameNStart = frameN;
            psychoJS.window.callOnFlip(() => {
                itiKey.clock.reset();
            });
            psychoJS.window.callOnFlip(() => {
                itiKey.start();
            });
            psychoJS.window.callOnFlip(() => {
                itiKey.clearEvents();
            });
        }

        if (itiKey.status === PsychoJS.Status.STARTED) {
            let theseKeys = itiKey.getKeys({keyList: ['return', 'enter', 'space'], waitRelease: false});
            _itiKey_allKeys = _itiKey_allKeys.concat(theseKeys);
            if (_itiKey_allKeys.length > 0) {
                const last = _itiKey_allKeys[_itiKey_allKeys.length - 1];
                itiKey.keys = last.name;
                itiKey.rt = last.rt;
                itiKey.duration = last.duration;
                continueRoutine = false;
            }
        }

        // auto-advance after 5s
        if (itiClock.getTime() >= 5.0) {
            continueRoutine = false;
        }

        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList: ['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }

        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        continueRoutine = false;
        ITIComponents.forEach(c => {
            if ('status' in c && c.status !== PsychoJS.Status.FINISHED) continueRoutine = true;
        });

        if (continueRoutine) return Scheduler.Event.FLIP_REPEAT;
        return Scheduler.Event.NEXT;
    };
}

function ITIRoutineEnd(snapshot) {
    return async function () {
        ITIComponents.forEach(c => {
            if (typeof c.setAutoDraw === 'function') c.setAutoDraw(false);
        });
        psychoJS.experiment.addData('ITI.stopped', globalClock.getTime());

        if (currentLoop instanceof MultiStairHandler) currentLoop.addResponse(itiKey.corr, level);
        psychoJS.experiment.addData('itiKey.keys', itiKey.keys);
        if (typeof itiKey.keys !== 'undefined') {
            psychoJS.experiment.addData('itiKey.rt', itiKey.rt);
            psychoJS.experiment.addData('itiKey.duration', itiKey.duration);
            routineTimer.reset();
        }

        itiKey.stop();
        routineTimer.reset();

        if (currentLoop === psychoJS.experiment) psychoJS.experiment.nextEntry(snapshot);
        return Scheduler.Event.NEXT;
    };
}

// =========================
// Utilities
// =========================
function importConditions(currentLoop) {
    return async function () {
        psychoJS.importAttributes(currentLoop.getCurrentTrial());
        return Scheduler.Event.NEXT;
    };
}

async function quitPsychoJS(message, isCompleted) {
    if (psychoJS.experiment.isEntryEmpty()) {
        psychoJS.experiment.nextEntry();
    }

    // Try uploading results first
    try {
        const csv = buildCsvFromExperiment();
        const meta = {
            workerId: expInfo?.workerId || '',
            assignmentId: expInfo?.assignmentId || '',
            hitId: expInfo?.hitId || '',
            participant: expInfo?.participant || '',
            timestamp: new Date().toISOString(),
        };
        await uploadCsvToSheets(csv, meta);
    } catch (e) {
        console.error('Sheets upload error:', e);
        // continue anyway; we still want to submit/return to MTurk
    }


    psychoJS.window.close();
    psychoJS.quit({message, isCompleted});
    return Scheduler.Event.QUIT;
}
