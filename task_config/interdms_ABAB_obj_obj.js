// Configuration object for the interdms ABAB obj obj task

export const TASK_CONFIG = {
    // ----- Basic experiment metadata -----
    expName: 'interdms_ABAB_obj_obj',

    // Which CSV to use for trials
    // trialsCsv: 'resources/interdms_debug.csv',
    trialsCsv: 'resources/interdms_ABAB_obj_obj_trials.csv',

    // Always-preload fallback images (and any other always-needed assets)
    alwaysResources: [
        'resources/images/157_Chairs.png'
    ],

    // How to extract stim / action fields from each trial row
    // These should match column names in the CSV
    stimFieldNames: ['stim1', 'stim2', 'stim3', 'stim4'],
    actFieldNames: [null, null, 'act3', 'act4'],

    // Frames with responses (0-based index)
    numFrames: 4,
    responseFrames: [2, 3],
    responseKeys: ['x', 'b'],

    // Timings (in seconds)
    timings: {
        learning: {
            image: 1.5,
            total: 1.5 + 2.0
        },
        main: {
            image: 0.5,
            total: 0.5 + 2.0
        },
        feedbackDuration: 1.0,
        itiDuration: 5.0
    },

    // Session logic
    learningSessionCode: '0',

    // Texts
    welcomeText: `
• You will perform an Interleaved Delayed Match-to-Sample ABAB Object–Object task.

• Each trial consists of a sequence of 4 images. You will make two comparisons:

  1) Identity match (BB):
     – Compare the 3rd frame with the 1st frame.
     – If they are the same, press 'X'.
     – Otherwise, press 'B'.
     – This is the _B_B in ABAB.
  2) Identity match (AA):
     – Compare the 4th frame with the 2nd frame.
     – If they are the same, press 'X'.
     – Otherwise, press 'B'.
     – This is the A_A_ in ABAB.

• Each image is shown for 500 ms, followed by a 2-second response window in which you can press 'X' or 'B'.

• There are 5 sessions of 20 trials each. You can take short breaks between sessions.

Press Enter to start.`,

    learningSessionEndText:
        `You have completed the learning session.

From now on, the real sessions will begin.
The task will run at normal speed, without extra
frame-by-frame instructions or feedback.

Press Enter to continue to the real sessions.`,

    thanksText:
        `Thank you for participating!

Please submit the following survey code on the MTurk page before submit: AWDR

Press Enter to finish.

Important: after you press Enter, do not close this page until the data file has
finished uploading (this may take a few seconds / you might have to press Enter multiple times). Please wait until you see next page`,

    // --------- TEXT HOOKS / HELPERS ---------
    isLearningSession(session) {
        return String(session) === this.learningSessionCode;
    },

    makeSessionIntro(session, isLearning, practiceEndedShown) {
        const sessStr = String(session);

        if (isLearning && !practiceEndedShown) {
            return `Learning Session (Practice)\n
In this first session, the task will run more slowly.
You will:

• See on-screen instructions for each frame.
• Have more time to view the images.
• Receive feedback after you respond on the comparison frames.

This learning session is slower than the normal sessions
you will perform afterward.

Press Enter to begin the learning session.`;
        }

        return `Session: ${sessStr}\nPress Enter to start`;
    },

    makeFrameInstruction(frameIdx, isLearning) {
        if (!isLearning) return '';
        if (frameIdx === 0) {
            return 'Remember the IDENTITY of this object.';
        } else if (frameIdx === 1) {
            return 'Remember the IDENTITY of this object.';
        } else if (frameIdx === 2) {
            return 'Compare IDENTITY with the FIRST image.\n' +
                'Press X if same, B if different.';
        } else if (frameIdx === 3) {
            return 'Compare IDENTITY with the THIRD image.\n' +
                'Press X if same, B if different.';
        } else {
            return '';
        }
        return '';
    },

    makeDelayText(frameIdx, isLearning) {
        if (!isLearning) return '';
        if (frameIdx === 2 || frameIdx === 3) {
            return 'Respond';
        }
        return 'Wait...';
    },

    shouldShowFeedback(frameIdx, isLearning) {
        return isLearning && this.responseFrames.includes(frameIdx);
    },

    makeITIText(trialHadResponse) {
        if (!trialHadResponse) {
            return 'We did not detect any responses in the last trial.\n' +
                'Please press X or B on the comparison images.\n\n' +
                'Next trial in 5s – press Enter to start now.';
        }
        return 'Next trial in 5s - press Enter to start now';
    }
};
