// Configuration for the 1-back position task

export const TASK_CONFIG = {
    // ----- Basic experiment metadata -----
    expName: '1back_pos',

    // Which CSV to use for trials
    // trialsCsv: 'resources/1back_debug.csv',
    trialsCsv: 'resources/1back_pos_trials.csv',

    // Always-preload fallback images (and any other always-needed assets)
    alwaysResources: [
        'resources/images/157_Chairs.png'
    ],

    // How to extract stim / action fields from each trial row
    // These should match column names in the CSV
    stimFieldNames: ['stim1', 'stim2', 'stim3', 'stim4', 'stim5', 'stim6'],
    actFieldNames: [null, 'act2', 'act3', 'act4', 'act5', 'act6'],

    // Frames with responses (0-based index)
    numFrames: 6,
    responseFrames: [1, 2, 3, 4, 5],
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
Task instructions:
• You will complete a 1-back Position task.

• Each trial contains 6 images (frames). Starting from the 2nd frame, press:
    - 'X' if the location of the current image matches the previous frame,
    - 'B' if it DOES NOT match.

• Each frame is shown for 500 ms, followed by a 2-second interval.

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
            return 'Remember the POSITION of this object.';
        }
        if (frameIdx >= 1 && frameIdx <= 5) {
            return 'Compare with the PREVIOUS image.\nPress X if same POSITION, B if different.';
        }
        return '';
    },

    makeDelayText(frameIdx, isLearning) {
        if (!isLearning) return '';
        if (frameIdx === 1 || frameIdx === 2 || frameIdx === 3 || frameIdx === 4 || frameIdx === 5) {
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
