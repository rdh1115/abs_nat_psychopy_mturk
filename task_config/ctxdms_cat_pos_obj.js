// Configuration object for the ctxdms category position identity task

export const TASK_CONFIG = {
    // ----- Basic experiment metadata -----
    expName: 'ctxdms_category_position_identity',

    // Which CSV to use for trials
    // trialsCsv: 'resources/interdms_debug.csv',
    trialsCsv: 'resources/ctxdms_category_pos_obj_trials.csv',

    // Always-preload fallback images (and any other always-needed assets)
    alwaysResources: [
        'resources/images/157_Chairs.png'
    ],

    // How to extract stim / action fields from each trial row
    // These should match column names in the CSV
    stimFieldNames: ['stim1', 'stim2', 'stim3'],
    actFieldNames: [null, null, 'act3'],

    // Frames with responses (0-based index)
    numFrames: 3,
    responseFrames: [2],
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

• You will perform a Contextual Delayed Match-to-Sample task.

• Each trial consists of a sequence of 3 images. You will make two comparisons, the second comparison depends on the result of the first:

  1) Category match:
     – Compare the 2nd frame with the 1st frame.
     – Remember if they are from the same category (for example, both planes).
  2) Position/Object match:
     – Compare the 3rd frame with the 2nd frame.
     – If first comparison was a match, then compare their POSITION:
     – Otherwise, compare their IDENTITY:
     – If they match according to the relevant feature, press 'X'.
     – Otherwise, press 'B'.

• The categories include Cars, Planes, Boats, Fruits, Faces, Animals, Chairs, and Tables.

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
            return 'Remember the CATEGORY of this object.';
        } else if (frameIdx === 1) {
            return 'Compare CATEGORY with the FIRST image.\n' +
                'Also remember the relevant feature.';
        } else if (frameIdx === 2) {
            return 'If the POSITION matched,\n' +
            'then compare CATEGORY with the SECOND image.\n' +
            'Otherwise, compare IDENTITY with the SECOND image.\n' +
            'Press X if match, B if different.';
        } else {
            return '';
        }
        return '';
    },

    makeDelayText(frameIdx, isLearning) {
        if (!isLearning) return '';
        if (frameIdx === 2) {
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
