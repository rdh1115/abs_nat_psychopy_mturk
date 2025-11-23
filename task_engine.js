/***********************
 * Generic PsychoJS task engine
 * Uses taskConfig passed into runTask(taskConfig)
 ***********************/

import {core, data, util, visual} from './lib/psychojs-2025.1.1.js';

const {PsychoJS} = core;
const {TrialHandler, MultiStairHandler} = data;
const {Scheduler} = util;
const Status = PsychoJS.Status;

// Keep a reference to the current task config (set in runTask)
let currentTaskConfig = null;

// ========== Cloud upload config ==========
// const WEB_APP_URL = 'https://script.google.com/macros/s/AKfycbx43V8Aha-JTWTKj51PHQo5SkQztRsV0EYfyAsULh2-NQeFcC1Y8k6wYyhO0_5b_p2amg/exec';
const WEB_APP_URL_PART1 = 'https://script.google.com/macros/s/AKfycbxWmyZZ5duXZCoFYcwdJdueZuoSMft1OAkx4Hru4-';
const WEB_APP_URL_PART2 = '9ChvAQXoVI5zsxgfixJMWcGWkmpA/exec';
const SECURE_TOKEN = 'fdajknarofr'

// ========== Helpers: CSV & upload ==========
function _csvEscape(v) {
    if (v === null || v === undefined) return '';
    const s = String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

function buildCsvFromExperiment(psychoJS) {
    const rows = psychoJS?.experiment?._trialsData || [];
    const keys = new Set();
    rows.forEach(r => Object.keys(r).forEach(k => keys.add(k)));
    const header = Array.from(keys);
    const lines = [header.map(_csvEscape).join(',')];
    for (const r of rows) lines.push(header.map(k => _csvEscape(r[k])).join(','));
    return lines.join('\n');
}

async function uploadCsvToSheets(csv, meta) {
    const fullWebAppUrl = WEB_APP_URL_PART1 + WEB_APP_URL_PART2;
    // const fullWebAppUrl = 'http://localhost:3000/upload'

    const payload = JSON.stringify({
        csv: csv,
        meta: meta,
        token: SECURE_TOKEN // <-- ADDED SECURITY TOKEN HERE
    });

    const res = await fetch(fullWebAppUrl, {
        method: 'POST',
        headers: {'Content-Type': 'text/plain;charset=utf-8'},
        body: payload,
    });

    const json = await res.json().catch(() => ({ok: false, error: 'Bad JSON response'}));

    if (!json.ok) {
        // This will catch both network errors and the 'Unauthorized token.' error from your Apps Script.
        throw new Error(json.error || 'Sheets upload failed');
    }

    return json;
}

// ========== Resource helpers ==========
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

async function collectImagePathsFromCSV(csvPath, taskConfig) {
    const res = await fetch(csvPath, {cache: 'no-store'});
    if (!res.ok) throw new Error(`Failed to load ${csvPath}: ${res.status}`);

    let text = await res.text();
    if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);

    const lines = text
        .split(/\r?\n/)
        .map(l => l.trim())
        .filter(l => l.length);

    if (!lines.length) return [];

    const headerRaw = lines[0].split(',').map(s => s.trim().replace(/^"|"$/g, ''));
    const header = headerRaw.map(h => h.toLowerCase());

    const stimCols = [];
    taskConfig.stimFieldNames.forEach(name => {
        const idx = header.indexOf(name.toLowerCase());
        if (idx >= 0) stimCols.push(idx);
    });

    const paths = new Set(taskConfig.alwaysResources || []);
    for (let r = 1; r < lines.length; r++) {
        const cols = lines[r].split(',').map(s => s.trim().replace(/^"|"$/g, ''));
        for (const ci of stimCols) {
            const raw = cols[ci];
            if (raw && raw.toLowerCase() !== 'default.png') {
                paths.add(normPath(raw));
            }
        }
    }
    return Array.from(paths);
}

// ========== PsychoJS + globals ==========
let psychoJS;                  // was const; now created in runTask
let flowScheduler;             // created in runTask
let dialogCancelScheduler;     // created in runTask

let expInfo = {
    workerId: '',
    mturkLink: ''
};

let PILOTING = util.getUrlParameters().has('__pilotToken');

// globalish state
let currentLoop;
let frameDur;
let globalClock;
let routineTimer;
let frameIdx;
let stimPaths = [];
let actKeys = [];
let trialHadResponse = false;
let learningSession = false;
let prevSession = null;
let practiceEndedShown = false;
let showLearningEndThisTrial = false;
let currentImageDur = 0.5;
let currentTotalDur = 2.5;
let lastFeedbackMsg = '';
let lastWasScored = false;
let currentTrial = {};  // holds current row from CSV

// components
let WelcomeClock, welcomeText, welcomeKey;
let GlobalsClock;
let SessionIntroClock, sessText, sessKey;
let TrialIntroClock, trialText, trialKey;
let FrameClock, img, resp, frameText, waitText;
let FeedbackClock, feedbackText;
let ITIClock, itiText, itiKey, itiClock;
let LearningSessionEndClock, learningEndText, learningEndKey;
let ThanksClock, thanksText, thanksKey;

// common scheduler locals
let t, frameN, continueRoutine, routineForceEnded;

// ========== PUBLIC ENTRY POINT ==========
// This replaces the old IIFE bootstrap.
export async function runTask(taskConfig) {
    currentTaskConfig = taskConfig;

    // Create PsychoJS + schedulers *per run*
    psychoJS = new PsychoJS({debug: false});
    flowScheduler = new Scheduler(psychoJS);
    dialogCancelScheduler = new Scheduler(psychoJS);

    // Collect resources based on this config
    const csvPathsRaw = await collectImagePathsFromCSV(
        currentTaskConfig.trialsCsv,
        currentTaskConfig
    );
    const allPaths = Array.from(new Set([
        ...(currentTaskConfig.alwaysResources || []),
        ...csvPathsRaw
    ])).filter(p => typeof p === 'string' && p.length > 0);

    const resources = [
        {name: currentTaskConfig.trialsCsv, path: currentTaskConfig.trialsCsv},
        ...allPaths.map(p => ({name: p, path: p}))
    ];

    // open window (you can parameterize fullscreen, bgColor, etc. via config later)
    psychoJS.openWindow({
        fullscr: false,
        color: new util.Color([0, 0, 0]),
        units: 'height',
        waitBlanking: true,
        backgroundImage: '',
        backgroundFit: 'none'
    });

    // Subject info dialog
    psychoJS.schedule(psychoJS.gui.DlgFromDict({
        dictionary: expInfo,
        title: currentTaskConfig.expName
    }));
    psychoJS.scheduleCondition(
        () => (psychoJS.gui.dialogComponent.button === 'OK'),
        flowScheduler,
        dialogCancelScheduler
    );

    // Scheduling pipeline
    flowScheduler.add(updateInfo);
    flowScheduler.add(experimentInit);

    // Welcome
    flowScheduler.add(WelcomeRoutineBegin());
    flowScheduler.add(WelcomeRoutineEachFrame());
    flowScheduler.add(WelcomeRoutineEnd());

    // Globals
    flowScheduler.add(GlobalsRoutineBegin());
    flowScheduler.add(GlobalsRoutineEachFrame());
    flowScheduler.add(GlobalsRoutineEnd());

    const trialsLoopScheduler = new Scheduler(psychoJS);
    flowScheduler.add(trialsLoopBegin(trialsLoopScheduler));
    flowScheduler.add(trialsLoopScheduler);
    flowScheduler.add(trialsLoopEnd);

    // Thanks & quit
    flowScheduler.add(ThanksRoutineBegin());
    flowScheduler.add(ThanksRoutineEachFrame());
    flowScheduler.add(ThanksRoutineEnd());
    flowScheduler.add(quitPsychoJS, '', true);

    dialogCancelScheduler.add(quitPsychoJS, '', false);

    // Start experiment with this task’s name and resources
    psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.EXP);
    await psychoJS.start({
        expName: currentTaskConfig.expName,
        expInfo,
        resources
    });

    // Optionally return psychoJS if you want access from caller
    return psychoJS;
}

// ========== Update info ==========
async function updateInfo() {
    currentLoop = psychoJS.experiment;
    expInfo['date'] = util.MonotonicClock.getDateStr();
    expInfo['expName'] = currentTaskConfig.expName;
    expInfo['psychopyVersion'] = '2025.1.1';
    expInfo['OS'] = window.navigator.platform;

    expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
    frameDur = (typeof expInfo['frameRate'] !== 'undefined')
        ? 1.0 / Math.round(expInfo['frameRate'])
        : 1.0 / 60.0;

    util.addInfoFromUrl(expInfo);

    return Scheduler.Event.NEXT;
}

// ========== Experiment init ==========
async function experimentInit() {
    GlobalsClock = new util.Clock();
    prevSession = null;
    frameIdx = -1;
    stimPaths = [];
    actKeys = [];
    trialHadResponse = false;

    // Session intro
    SessionIntroClock = new util.Clock();
    sessText = new visual.TextStim({
        win: psychoJS.window,
        name: 'sessText',
        text: 'Press Enter to start',
        font: 'Open Sans',
        pos: [-0.4, 0],
        anchor: 'left',
        alignHoriz: 'left',
        alignVert: 'center',
        height: 0.04,
        wrapWidth: 0.8,
        draggable: false,
        languageStyle: 'LTR',
        color: new util.Color('white'),
        depth: 0.0
    });
    sessKey = new core.Keyboard({psychoJS, clock: new util.Clock(), waitForStart: true});

    // Trial intro
    TrialIntroClock = new util.Clock();
    trialText = new visual.TextStim({
        win: psychoJS.window,
        name: 'trialText',
        text: 'Press Enter to start the trial',
        font: 'Open Sans',
        pos: [-0.4, 0],
        anchor: 'center',
        alignHoriz: 'left',
        alignVert: 'center',
        draggable: false,
        height: 0.05,
        wrapWidth: 1.2,
        languageStyle: 'LTR',
        color: new util.Color('white'),
        depth: 0.0
    });
    trialKey = new core.Keyboard({psychoJS, clock: new util.Clock(), waitForStart: true});

    // Frame
    FrameClock = new util.Clock();
    img = new visual.ImageStim({
        win: psychoJS.window,
        name: 'img',
        image: 'resources/images/157_Chairs.png',
        anchor: 'center',
        pos: [0, 0],
        size: [0.5, 0.5],
        color: new util.Color([1, 1, 1]),
        interpolate: true
    });
    frameText = new visual.TextStim({
        win: psychoJS.window,
        name: 'frameText',
        text: '',
        font: 'Open Sans',
        pos: [0, 0.4],          // top-ish
        draggable: false,
        height: 0.04,
        wrapWidth: 1.2,
        languageStyle: 'LTR',
        color: new util.Color('white'),
        depth: -1.0
    });
    waitText = new visual.TextStim({
        win: psychoJS.window,
        name: 'waitText',
        text: '',
        font: 'Open Sans',
        pos: [0, 0],            // center of the screen
        draggable: false,
        height: 0.08,
        wrapWidth: 1.2,
        languageStyle: 'LTR',
        color: new util.Color('white'),
        depth: 1.0
    });
    resp = new core.Keyboard({psychoJS, clock: new util.Clock(), waitForStart: true});

    // Feedback
    FeedbackClock = new util.Clock();
    feedbackText = new visual.TextStim({
        win: psychoJS.window,
        name: 'feedbackText',
        text: '',
        font: 'Open Sans',
        pos: [0, -0.3],        // near bottom
        draggable: false,
        height: 0.05,
        wrapWidth: 1.2,
        languageStyle: 'LTR',
        color: new util.Color('white'),
        depth: 0.0
    });

    // ITI
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
    itiClock = new util.Clock();

    // Welcome
    WelcomeClock = new util.Clock();
    welcomeText = new visual.TextStim({
        win: psychoJS.window,
        name: 'welcomeText',
        text: currentTaskConfig.welcomeText,
        font: 'Open Sans',
        pos: [-0.4, 0],
        anchor: 'left',
        alignHoriz: 'left',
        alignVert: 'center',
        height: 0.028,
        wrapWidth: 0.8,
        draggable: false,
        color: new util.Color('white'),
        depth: 0.0
    });
    welcomeKey = new core.Keyboard({psychoJS, clock: new util.Clock(), waitForStart: true});

    // Thanks
    ThanksClock = new util.Clock();
    thanksText = new visual.TextStim({
        win: psychoJS.window,
        name: 'thanksText',
        text: currentTaskConfig.thanksText,
        font: 'Open Sans',
        pos: [0, 0], draggable: false, height: 0.05,
        color: new util.Color('white'),
        wrapWidth: 1.2,
        depth: 0.0
    });
    thanksKey = new core.Keyboard({psychoJS, clock: new util.Clock(), waitForStart: true});

    // Learning session end
    LearningSessionEndClock = new util.Clock();
    learningEndText = new visual.TextStim({
        win: psychoJS.window,
        name: 'learningEndText',
        text: currentTaskConfig.learningSessionEndText,
        font: 'Open Sans',
        draggable: false,
        pos: [-0.4, 0],
        anchor: 'left',
        alignHoriz: 'left',
        alignVert: 'center',
        height: 0.04,
        wrapWidth: 0.8,
        languageStyle: 'LTR',
        color: new util.Color('white'),
        depth: 0.0
    });
    learningEndKey = new core.Keyboard({psychoJS, clock: new util.Clock(), waitForStart: true});

    // clocks
    globalClock = new util.Clock();
    routineTimer = new util.CountdownTimer();

    return Scheduler.Event.NEXT;
}

// ========== Globals routine ==========
let GlobalsComponents = [];

function GlobalsRoutineBegin(snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        t = 0;
        frameN = -1;
        continueRoutine = true;
        routineForceEnded = false;

        GlobalsClock.reset();
        routineTimer.reset();

        psychoJS.experiment.addData('Globals.started', globalClock.getTime());
        GlobalsComponents = [];
        return Scheduler.Event.NEXT;
    };
}

function GlobalsRoutineEachFrame() {
    return async function () {
        t = GlobalsClock.getTime();
        frameN += 1;

        if (psychoJS.experiment.experimentEnded ||
            psychoJS.eventManager.getKeys({keyList: ['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }

        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        continueRoutine = GlobalsComponents.some(
            c => ('status' in c) && c.status !== Status.FINISHED
        );

        return continueRoutine ? Scheduler.Event.FLIP_REPEAT : Scheduler.Event.NEXT;
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

// ========== Welcome ==========
let WelcomeComponents, _welcomeKey_allKeys = [];

function WelcomeRoutineBegin(snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        t = 0;
        frameN = -1;
        continueRoutine = true;
        routineForceEnded = false;

        WelcomeClock.reset();
        routineTimer.reset();

        welcomeKey.keys = undefined;
        welcomeKey.rt = undefined;
        _welcomeKey_allKeys = [];

        psychoJS.experiment.addData('Welcome.started', globalClock.getTime());
        WelcomeComponents = [welcomeText, welcomeKey];
        WelcomeComponents.forEach(c => {
            if ('status' in c) c.status = Status.NOT_STARTED;
        });

        return Scheduler.Event.NEXT;
    };
}

function WelcomeRoutineEachFrame() {
    return async function () {
        t = WelcomeClock.getTime();
        frameN += 1;

        if (t >= 0.0 && welcomeText.status === Status.NOT_STARTED) {
            welcomeText.tStart = t;
            welcomeText.frameNStart = frameN;
            welcomeText.setAutoDraw(true);
            welcomeText.status = Status.STARTED;
        }

        if (t >= 0.0 && welcomeKey.status === Status.NOT_STARTED) {
            welcomeKey.tStart = t;
            welcomeKey.frameNStart = frameN;
            psychoJS.window.callOnFlip(() => welcomeKey.clock.reset());
            psychoJS.window.callOnFlip(() => welcomeKey.start());
            psychoJS.window.callOnFlip(() => welcomeKey.clearEvents());
            welcomeKey.status = Status.STARTED;
        }

        if (welcomeKey.status === Status.STARTED) {
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

        if (psychoJS.experiment.experimentEnded ||
            psychoJS.eventManager.getKeys({keyList: ['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }

        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        continueRoutine = WelcomeComponents.some(
            c => ('status' in c) && c.status !== Status.FINISHED
        );

        return continueRoutine ? Scheduler.Event.FLIP_REPEAT : Scheduler.Event.NEXT;
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

// ========== SessionIntro ==========
let SessionIntroComponents, _sessKey_allKeys = [];

function SessionIntroRoutineBegin(snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        t = 0;
        frameN = -1;
        continueRoutine = true;
        routineForceEnded = false;

        SessionIntroClock.reset();
        routineTimer.reset();

        sessKey.keys = undefined;
        sessKey.rt = undefined;
        _sessKey_allKeys = [];

        const showIntro = (prevSession === null || typeof prevSession === 'undefined' || session !== prevSession);
        if (!showIntro) {
            continueRoutine = false;
        } else {
            const isLearning = currentTaskConfig.isLearningSession(session);
            const introText = currentTaskConfig.makeSessionIntro(session, isLearning, practiceEndedShown);
            sessText.setText(introText);
        }

        psychoJS.experiment.addData('SessionIntro.started', globalClock.getTime());
        SessionIntroComponents = [sessText, sessKey];
        SessionIntroComponents.forEach(c => {
            if ('status' in c) c.status = Status.NOT_STARTED;
        });

        return Scheduler.Event.NEXT;
    };
}

function SessionIntroRoutineEachFrame() {
    return async function () {
        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        t = SessionIntroClock.getTime();
        frameN += 1;

        if (t >= 0.0 && sessText.status === Status.NOT_STARTED) {
            sessText.tStart = t;
            sessText.frameNStart = frameN;
            sessText.setAutoDraw(true);
            sessText.status = Status.STARTED;
        }

        if (t >= 0.0 && sessKey.status === Status.NOT_STARTED) {
            sessKey.tStart = t;
            sessKey.frameNStart = frameN;
            psychoJS.window.callOnFlip(() => sessKey.clock.reset());
            psychoJS.window.callOnFlip(() => sessKey.start());
            psychoJS.window.callOnFlip(() => sessKey.clearEvents());
            sessKey.status = Status.STARTED;
        }

        if (sessKey.status === Status.STARTED) {
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

        if (psychoJS.experiment.experimentEnded ||
            psychoJS.eventManager.getKeys({keyList: ['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }

        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        continueRoutine = SessionIntroComponents.some(
            c => ('status' in c) && c.status !== Status.FINISHED
        );

        return continueRoutine ? Scheduler.Event.FLIP_REPEAT : Scheduler.Event.NEXT;
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
        routineTimer.reset();
        if (currentLoop === psychoJS.experiment) psychoJS.experiment.nextEntry(snapshot);
        return Scheduler.Event.NEXT;
    };
}

// ========== TrialIntro ==========
let TrialIntroComponents, _trialKey_allKeys = [];

function TrialIntroRoutineBegin(snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        t = 0;
        frameN = -1;
        continueRoutine = true;
        routineForceEnded = false;

        TrialIntroClock.reset();
        routineTimer.reset();

        trialKey.keys = undefined;
        trialKey.rt = undefined;
        _trialKey_allKeys = [];

        frameIdx = -1;
        trialHadResponse = false;

        const low = s => (((typeof s) === 'string') || (s instanceof String)) ? s.toLowerCase() : s;

        // build stimPaths & actKeys from currentTrial (set by importConditions)
        stimPaths = currentTaskConfig.stimFieldNames.map(name => currentTrial[name]);
        actKeys = currentTaskConfig.actFieldNames.map(name => {
            if (!name) return null;
            return low(currentTrial[name]);
        });

        learningSession = currentTaskConfig.isLearningSession(session);

        psychoJS.experiment.addData('TrialIntro.started', globalClock.getTime());
        TrialIntroComponents = [trialText, trialKey];
        TrialIntroComponents.forEach(c => {
            if ('status' in c) c.status = Status.NOT_STARTED;
        });

        return Scheduler.Event.NEXT;
    };
}

function TrialIntroRoutineEachFrame() {
    return async function () {
        t = TrialIntroClock.getTime();
        frameN += 1;

        if (t >= 0.0 && trialText.status === Status.NOT_STARTED) {
            trialText.tStart = t;
            trialText.frameNStart = frameN;
            trialText.setAutoDraw(true);
            trialText.status = Status.STARTED;
        }

        if (t >= 0.0 && trialKey.status === Status.NOT_STARTED) {
            trialKey.tStart = t;
            trialKey.frameNStart = frameN;
            psychoJS.window.callOnFlip(() => trialKey.clock.reset());
            psychoJS.window.callOnFlip(() => trialKey.start());
            psychoJS.window.callOnFlip(() => trialKey.clearEvents());
            trialKey.status = Status.STARTED;
        }

        if (trialKey.status === Status.STARTED) {
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

        if (psychoJS.experiment.experimentEnded ||
            psychoJS.eventManager.getKeys({keyList: ['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }

        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        continueRoutine = TrialIntroComponents.some(
            c => ('status' in c) && c.status !== Status.FINISHED
        );

        return continueRoutine ? Scheduler.Event.FLIP_REPEAT : Scheduler.Event.NEXT;
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

// ========== Frame routine ==========
let FrameComponents, _resp_allKeys = [];
let frameRemains;

function FrameRoutineBegin(snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        t = 0;
        frameN = -1;
        continueRoutine = true;
        routineForceEnded = false;

        // timings
        if (learningSession) {
            currentImageDur = currentTaskConfig.timings.learning.image;
            currentTotalDur = currentTaskConfig.timings.learning.total;
        } else {
            currentImageDur = currentTaskConfig.timings.main.image;
            currentTotalDur = currentTaskConfig.timings.main.total;
        }

        FrameClock.reset();
        routineTimer.reset();
        routineTimer.add(currentTotalDur);

        resp.keys = undefined;
        resp.rt = undefined;
        _resp_allKeys = [];
        lastFeedbackMsg = '';
        lastWasScored = false;

        waitText.setAutoDraw(false);
        frameText.setAutoDraw(false);
        feedbackText.setAutoDraw(false);

        frameIdx += 1;

        if (frameIdx >= stimPaths.length) {
            continueRoutine = false;
        } else {
            const rawStim = stimPaths[frameIdx];
            const fallbackStim = (currentTaskConfig.alwaysResources && currentTaskConfig.alwaysResources[0]) ||
                'resources/images/157_Chairs.png';
            const currStim = normPath(rawStim) || fallbackStim;
            img.setImage(currStim);

            const instr = currentTaskConfig.makeFrameInstruction(frameIdx, learningSession);
            frameText.setText(instr || '');
            psychoJS.eventManager.clearEvents();
        }

        psychoJS.experiment.addData('Frame.started', globalClock.getTime());

        FrameComponents = [img, resp, frameText, waitText, feedbackText];
        FrameComponents.forEach(c => {
            if ('status' in c) c.status = Status.NOT_STARTED;
        });

        return Scheduler.Event.NEXT;
    };
}

function FrameRoutineEachFrame() {
    return async function () {
        t = FrameClock.getTime();
        frameN += 1;

        // show image
        if (t >= 0.0 && img.status === Status.NOT_STARTED) {
            img.tStart = t;
            img.frameNStart = frameN;
            img.setAutoDraw(true);
            img.status = Status.STARTED;
        }
        frameRemains = currentImageDur - psychoJS.window.monitorFramePeriod * 0.75;
        if (img.status === Status.STARTED && t >= frameRemains) {
            img.tStop = t;
            img.frameNStop = frameN;
            img.setAutoDraw(false);
            img.status = Status.FINISHED;
        }

        // frameText
        if (learningSession && frameText.text) {
            if (t >= 0.0 && frameText.status === Status.NOT_STARTED) {
                frameText.tStart = t;
                frameText.frameNStart = frameN;
                frameText.setAutoDraw(true);
                frameText.status = Status.STARTED;
            }
        }

        // wait / respond text during delay
        if (learningSession && frameIdx >= 0 && frameIdx < currentTaskConfig.numFrames) {
            const frameEnd = currentTotalDur - psychoJS.window.monitorFramePeriod * 0.75;

            if (t >= currentImageDur && waitText.status === Status.NOT_STARTED) {
                const msg = currentTaskConfig.makeDelayText(frameIdx, learningSession);
                if (msg) {
                    waitText.setText(msg);
                    waitText.tStart = t;
                    waitText.frameNStart = frameN;
                    waitText.setAutoDraw(true);
                    waitText.status = Status.STARTED;
                }
            }

            if (waitText.status === Status.STARTED && t >= frameEnd) {
                waitText.tStop = t;
                waitText.frameNStop = frameN;
                waitText.setAutoDraw(false);
                waitText.status = Status.FINISHED;
            }
        }

        const isResponseFrame = currentTaskConfig.responseFrames.includes(frameIdx);

        if (isResponseFrame) {
            // start resp collection after image
            if (t >= currentImageDur && resp.status === Status.NOT_STARTED) {
                resp.tStart = t;
                resp.frameNStart = frameN;
                psychoJS.window.callOnFlip(() => resp.clock.reset());
                psychoJS.window.callOnFlip(() => resp.start());
                psychoJS.window.callOnFlip(() => resp.clearEvents());
                resp.status = Status.STARTED;
            }

            // stop at frame end
            const frameEnd = currentTotalDur - psychoJS.window.monitorFramePeriod * 0.75;
            if (resp.status === Status.STARTED && t >= frameEnd) {
                resp.tStop = t;
                resp.frameNStop = frameN;
                resp.status = Status.FINISHED;
            }

            if (resp.status === Status.STARTED) {
                let theseKeys = resp.getKeys({keyList: currentTaskConfig.responseKeys, waitRelease: false});

                if (theseKeys.length > 0 && typeof resp.keys === 'undefined') {
                    const last = theseKeys[theseKeys.length - 1];
                    resp.keys = last.name;
                    resp.rt = last.rt;
                    resp.duration = last.duration;

                    trialHadResponse = true;

                    if (learningSession) {
                        const expected = actKeys[frameIdx];
                        const key = resp.keys ? resp.keys.toLowerCase() : null;
                        let fb = '';

                        if (!key) {
                            fb = 'Too slow! Please try to respond within the time window.';
                        } else if (expected && key === expected) {
                            fb = 'Correct!';
                            lastWasScored = true;
                        } else if (expected) {
                            fb = 'Incorrect.';
                            lastWasScored = true;
                        }

                        lastFeedbackMsg = fb;
                        if (fb) {
                            feedbackText.setText(fb);
                            feedbackText.setAutoDraw(true);
                            feedbackText.status = Status.STARTED;
                        }
                    }

                    resp.stop();
                    resp.status = Status.FINISHED;
                }
            }
        }

        if (psychoJS.experiment.experimentEnded ||
            psychoJS.eventManager.getKeys({keyList: ['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }

        if (!continueRoutine || routineTimer.getTime() <= 0) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        continueRoutine = FrameComponents.some(
            c => ('status' in c) && c.status !== Status.FINISHED
        );

        return continueRoutine ? Scheduler.Event.FLIP_REPEAT : Scheduler.Event.NEXT;
    };
}

function FrameRoutineEnd(snapshot) {
    return async function () {
        FrameComponents.forEach(c => {
            if (typeof c.setAutoDraw === 'function') c.setAutoDraw(false);
        });
        waitText.setAutoDraw(false);
        frameText.setAutoDraw(false);

        psychoJS.experiment.addData('Frame.stopped', globalClock.getTime());

        if (currentTaskConfig.responseFrames.includes(frameIdx)) {
            const expected = actKeys[frameIdx];
            const key = resp.keys ? resp.keys.toLowerCase() : null;

            let correct = null;
            let fb = '';

            if (!key) {
                correct = 0;
                fb = 'Too slow! Please try to respond within the time window.';
            } else if (expected && key === expected) {
                correct = 1;
                fb = 'Correct!';
            } else if (expected) {
                correct = 0;
                fb = 'Incorrect.';
            } else {
                correct = null;
                fb = '';
            }

            psychoJS.experiment.addData('resp.keys', resp.keys);
            if (typeof resp.keys !== 'undefined') {
                psychoJS.experiment.addData('resp.rt', resp.rt);
                psychoJS.experiment.addData('resp.duration', resp.duration);
            }

            psychoJS.experiment.addData(`resp_frame${frameIdx + 1}_correct`, correct);

            lastWasScored = currentTaskConfig.shouldShowFeedback(frameIdx, learningSession) && (fb !== '');
            lastFeedbackMsg = fb;
        } else {
            lastWasScored = false;
            lastFeedbackMsg = '';
        }

        resp.stop();
        routineTimer.reset();

        if (currentLoop === psychoJS.experiment) psychoJS.experiment.nextEntry(snapshot);
        return Scheduler.Event.NEXT;
    };
}

// ========== Feedback ==========
let FeedbackComponents;

function FeedbackRoutineBegin(snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        t = 0;
        frameN = -1;
        continueRoutine = true;
        routineForceEnded = false;

        FeedbackClock.reset();
        routineTimer.reset();

        psychoJS.experiment.addData('Feedback.started', globalClock.getTime());

        if (learningSession && lastWasScored && lastFeedbackMsg) {
            feedbackText.setText(lastFeedbackMsg);
        } else {
            continueRoutine = false;
        }

        FeedbackComponents = [feedbackText];
        FeedbackComponents.forEach(c => {
            if ('status' in c) c.status = Status.NOT_STARTED;
        });

        routineTimer.add(currentTaskConfig.timings.feedbackDuration);

        return Scheduler.Event.NEXT;
    };
}

function FeedbackRoutineEachFrame() {
    return async function () {
        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        t = FeedbackClock.getTime();
        frameN += 1;

        if (t >= 0.0 && feedbackText.status === Status.NOT_STARTED) {
            feedbackText.tStart = t;
            feedbackText.frameNStart = frameN;
            feedbackText.setAutoDraw(true);
            feedbackText.status = Status.STARTED;
        }

        if (routineTimer.getTime() <= 0) {
            continueRoutine = false;
        }

        if (psychoJS.experiment.experimentEnded ||
            psychoJS.eventManager.getKeys({keyList: ['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }

        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        continueRoutine = FeedbackComponents.some(
            c => ('status' in c) && c.status !== Status.FINISHED
        );

        return continueRoutine ? Scheduler.Event.FLIP_REPEAT : Scheduler.Event.NEXT;
    };
}

function FeedbackRoutineEnd(snapshot) {
    return async function () {
        FeedbackComponents.forEach(c => {
            if (typeof c.setAutoDraw === 'function') c.setAutoDraw(false);
        });
        psychoJS.experiment.addData('Feedback.stopped', globalClock.getTime());

        routineTimer.reset();
        lastWasScored = false;

        if (currentLoop === psychoJS.experiment) psychoJS.experiment.nextEntry(snapshot);
        return Scheduler.Event.NEXT;
    };
}

// ========== ITI ==========
let ITIComponents, _itiKey_allKeys = [];

function ITIRoutineBegin(snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        t = 0;
        frameN = -1;
        continueRoutine = true;
        routineForceEnded = false;

        ITIClock.reset();
        routineTimer.reset();
        itiClock.reset();

        itiKey.keys = undefined;
        itiKey.rt = undefined;
        _itiKey_allKeys = [];

        psychoJS.experiment.addData('trialHasResponse', trialHadResponse ? 1 : 0);
        const showITIText = !learningSession;

        itiText.setText(currentTaskConfig.makeITIText(trialHadResponse));

        psychoJS.experiment.addData('ITI.started', globalClock.getTime());
        ITIComponents = [itiText, itiKey];
        ITIComponents.forEach(c => {
            if ('status' in c) c.status = Status.NOT_STARTED;
        });

        return Scheduler.Event.NEXT;
    };
}

function ITIRoutineEachFrame() {
    return async function () {
        t = ITIClock.getTime();
        frameN += 1;

        if (t >= 0.0 && itiText.status === Status.NOT_STARTED) {
            itiText.tStart = t;
            itiText.frameNStart = frameN;
            itiText.setAutoDraw(true);
            itiText.status = Status.STARTED;
        }

        if (t >= 0.0 && itiKey.status === Status.NOT_STARTED) {
            itiKey.tStart = t;
            itiKey.frameNStart = frameN;
            psychoJS.window.callOnFlip(() => itiKey.clock.reset());
            psychoJS.window.callOnFlip(() => itiKey.start());
            psychoJS.window.callOnFlip(() => itiKey.clearEvents());
            itiKey.status = Status.STARTED;
        }

        if (itiKey.status === Status.STARTED) {
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

        if (itiClock.getTime() >= currentTaskConfig.timings.itiDuration) {
            continueRoutine = false;
        }

        if (psychoJS.experiment.experimentEnded ||
            psychoJS.eventManager.getKeys({keyList: ['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }

        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        continueRoutine = ITIComponents.some(
            c => ('status' in c) && c.status !== Status.FINISHED
        );

        return continueRoutine ? Scheduler.Event.FLIP_REPEAT : Scheduler.Event.NEXT;
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

// ========== LearningSessionEnd ==========
let LearningSessionEndComponents, _learningEndKey_allKeys = [];

function LearningSessionEndRoutineBegin(snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        t = 0;
        frameN = -1;
        continueRoutine = true;
        routineForceEnded = false;

        LearningSessionEndClock.reset();
        routineTimer.reset();

        learningEndKey.keys = undefined;
        learningEndKey.rt = undefined;
        _learningEndKey_allKeys = [];

        LearningSessionEndComponents = [learningEndText, learningEndKey];
        LearningSessionEndComponents.forEach(c => {
            if ('status' in c) c.status = Status.NOT_STARTED;
        });

        if (!showLearningEndThisTrial) {
            continueRoutine = false;
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }
        psychoJS.experiment.addData('LearningSessionEnd.started', globalClock.getTime());

        return Scheduler.Event.NEXT;
    };
}

function LearningSessionEndRoutineEachFrame() {
    return async function () {
        if (!showLearningEndThisTrial || !continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        t = LearningSessionEndClock.getTime();
        frameN += 1;

        if (t >= 0.0 && learningEndText.status === Status.NOT_STARTED) {
            learningEndText.tStart = t;
            learningEndText.frameNStart = frameN;
            learningEndText.setAutoDraw(true);
            learningEndText.status = Status.STARTED;
        }

        if (t >= 0.0 && learningEndKey.status === Status.NOT_STARTED) {
            learningEndKey.status = Status.STARTED;
            psychoJS.window.callOnFlip(() => learningEndKey.clock.reset());
            psychoJS.window.callOnFlip(() => learningEndKey.start());
            psychoJS.window.callOnFlip(() => learningEndKey.clearEvents());
        }

        let theseKeys = learningEndKey.getKeys({keyList: ['enter', 'return', 'space'], waitRelease: false});
        _learningEndKey_allKeys = _learningEndKey_allKeys.concat(theseKeys);
        if (_learningEndKey_allKeys.length > 0) {
            continueRoutine = false;
        }

        if (psychoJS.experiment.experimentEnded ||
            psychoJS.eventManager.getKeys({keyList: ['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }

        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        continueRoutine = LearningSessionEndComponents.some(
            c => c.status !== Status.FINISHED
        );

        return continueRoutine ? Scheduler.Event.FLIP_REPEAT : Scheduler.Event.NEXT;
    };
}

function LearningSessionEndRoutineEnd(snapshot) {
    return async function () {
        if (!showLearningEndThisTrial) {
            continueRoutine = false;
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }
        LearningSessionEndComponents.forEach(c => {
            if (typeof c.setAutoDraw === 'function') c.setAutoDraw(false);
        });
        learningEndKey.stop();

        psychoJS.experiment.addData('LearningSessionEnd.stopped', globalClock.getTime());

        showLearningEndThisTrial = false;
        if (currentLoop === psychoJS.experiment) psychoJS.experiment.nextEntry(snapshot);
        return Scheduler.Event.NEXT;
    };
}

// ========== Thanks ==========
let ThanksComponents, _thanksKey_allKeys = [];

function ThanksRoutineBegin(snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        t = 0;
        frameN = -1;
        continueRoutine = true;
        routineForceEnded = false;

        ThanksClock.reset();
        routineTimer.reset();

        thanksKey.keys = undefined;
        thanksKey.rt = undefined;
        _thanksKey_allKeys = [];

        psychoJS.experiment.addData('Thanks.started', globalClock.getTime());
        ThanksComponents = [thanksText, thanksKey];
        ThanksComponents.forEach(c => {
            if ('status' in c) c.status = Status.NOT_STARTED;
        });

        return Scheduler.Event.NEXT;
    };
}

function ThanksRoutineEachFrame() {
    return async function () {
        t = ThanksClock.getTime();
        frameN += 1;

        if (t >= 0.0 && thanksText.status === Status.NOT_STARTED) {
            thanksText.tStart = t;
            thanksText.frameNStart = frameN;
            thanksText.setAutoDraw(true);
            thanksText.status = Status.STARTED;
        }

        if (t >= 0.0 && thanksKey.status === Status.NOT_STARTED) {
            thanksKey.tStart = t;
            thanksKey.frameNStart = frameN;
            psychoJS.window.callOnFlip(() => thanksKey.clock.reset());
            psychoJS.window.callOnFlip(() => thanksKey.start());
            psychoJS.window.callOnFlip(() => thanksKey.clearEvents());
            thanksKey.status = Status.STARTED;
        }

        if (thanksKey.status === Status.STARTED) {
            let theseKeys = thanksKey.getKeys({keyList: ['return', 'enter', 'space'], waitRelease: false});
            _thanksKey_allKeys = _thanksKey_allKeys.concat(theseKeys);
            if (_thanksKey_allKeys.length > 0) {
                const last = _thanksKey_allKeys[_thanksKey_allKeys.length - 1];
                thanksKey.keys = last.name;
                thanksKey.rt = last.rt;
                thanksKey.duration = last.duration;
                thanksKey.stop();
                thanksKey.status = Status.FINISHED;
                continueRoutine = false;
            }
        }

        if (psychoJS.experiment.experimentEnded ||
            psychoJS.eventManager.getKeys({keyList: ['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }

        if (!continueRoutine) {
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
        }

        continueRoutine = ThanksComponents.some(
            c => ('status' in c) && c.status !== Status.FINISHED
        );

        return continueRoutine ? Scheduler.Event.FLIP_REPEAT : Scheduler.Event.NEXT;
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

// ========== Trials & frames loops ==========
let trials, frames;

function trialsLoopBegin(trialsLoopScheduler, snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        trials = new TrialHandler({
            psychoJS,
            nReps: 1,
            method: TrialHandler.Method.SEQUENTIAL,
            extraInfo: expInfo,
            originPath: undefined,
            trialList: currentTaskConfig.trialsCsv,
            seed: undefined,
            name: 'trials'
        });
        psychoJS.experiment.addLoop(trials);
        currentLoop = trials;

        trials.forEach(function () {
            snapshot = trials.getSnapshot();
            trialsLoopScheduler.add(importConditions(snapshot));

            trialsLoopScheduler.add(async function () {
                if (!practiceEndedShown &&
                    String(prevSession) === currentTaskConfig.learningSessionCode &&
                    session !== prevSession
                ) {
                    practiceEndedShown = true;
                    showLearningEndThisTrial = true;
                } else {
                    showLearningEndThisTrial = false;
                }
                return Scheduler.Event.NEXT;
            });

            trialsLoopScheduler.add(LearningSessionEndRoutineBegin(snapshot));
            trialsLoopScheduler.add(LearningSessionEndRoutineEachFrame());
            trialsLoopScheduler.add(LearningSessionEndRoutineEnd(snapshot));

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
            trialsLoopScheduler.add(ITIRoutineBegin(snapshot));
            trialsLoopScheduler.add(ITIRoutineEachFrame());
            trialsLoopScheduler.add(ITIRoutineEnd(snapshot));
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
            prevSession = session;

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

function framesLoopBegin(framesLoopScheduler, snapshot) {
    return async function () {
        TrialHandler.fromSnapshot(snapshot);

        frames = new TrialHandler({
            psychoJS,
            nReps: currentTaskConfig.numFrames,
            method: TrialHandler.Method.SEQUENTIAL,
            extraInfo: expInfo,
            originPath: undefined,
            trialList: undefined,
            seed: undefined,
            name: 'frames'
        });
        psychoJS.experiment.addLoop(frames);
        currentLoop = frames;

        frames.forEach(function () {
            snapshot = frames.getSnapshot();

            framesLoopScheduler.add(importConditions(snapshot));
            framesLoopScheduler.add(FrameRoutineBegin(snapshot));
            framesLoopScheduler.add(FrameRoutineEachFrame());
            framesLoopScheduler.add(FrameRoutineEnd(snapshot));

            framesLoopScheduler.add(FeedbackRoutineBegin(snapshot));
            framesLoopScheduler.add(FeedbackRoutineEachFrame());
            framesLoopScheduler.add(FeedbackRoutineEnd(snapshot));

            framesLoopScheduler.add(framesLoopEndIteration(framesLoopScheduler, snapshot));
        });

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

// ========== Utilities ==========
function importConditions(loopSnapshot) {
    return async function () {
        const trial = loopSnapshot.getCurrentTrial();
        currentTrial = trial || {};
        psychoJS.importAttributes(trial);
        return Scheduler.Event.NEXT;
    };
}

async function quitPsychoJS(message, isCompleted) {
    if (psychoJS.experiment.isEntryEmpty()) {
        psychoJS.experiment.nextEntry();
    }

    try {
        const csv = buildCsvFromExperiment(psychoJS);
        const meta = {
            workerId: expInfo?.workerId || '',
            assignmentId: expInfo?.assignmentId || '',
            hitId: expInfo?.hitId || '',
            participant: expInfo?.participant || '',
            timestamp: new Date().toISOString()
        };
        await uploadCsvToSheets(csv, meta);
    } catch (e) {
        console.error('Sheets upload error:', e);
    }

    psychoJS.window.close();
    psychoJS.quit({message, isCompleted});
    return Scheduler.Event.QUIT;
}
