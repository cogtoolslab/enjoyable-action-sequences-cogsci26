function setupGame() {
    var urlParams = new URLSearchParams(window.location.search);
    try {
        gs.prolific_info.prolificID = urlParams.get('PROLIFIC_PID');
        gs.prolific_info.prolificStudyID = urlParams.get('STUDY_ID');
        gs.prolific_info.prolificSessionID = urlParams.get('SESSION_ID');
    } catch (error) {
        console.error('Error obtaining prolific URL parameters:', error);
    }

    // Request one stimulus packet for this participant.
    socket.emit("getStims",
        {
            db_name: "stimuli",
            exp_name: gs.study_metadata.experiment,
        }
    );

    socket.on('stims', function (d) {
        gs.session_info.gameID = d.gameid;  // gameId assigned by app.js
        gs.session_info.participantID = d.participantID;
        gs.session_info.questionCondition = d.questionCondition;
        gs.session_info.agentOrderCondition = d.agentOrderCondition;
        gs.session_info.trajectoryPair = d.trajectoryPair;
        gs.session_info.colorOrderCondition = d.colorOrderCondition;
        gs.session_info.trials = d.trials;

        // Data are streamed to Mongo throughout the experiment.
        gs.session_info.send_data = function (data, is_final_edit = false, experiment_finished = false){
            var json = _.extend({},
                { study_metadata: gs.study_metadata },
                { experiment_finished: experiment_finished},
                { finished_trial_flag: is_final_edit},
                { session_info: _.omit(gs.session_info, 'on_finish', 'stimuli') },
                { session_timing: gs.session_timing },
                { prolific: gs.prolific_info },
                data);
            socket.emit('currentData', json,
                gs.study_metadata.project, //dbname
                gs.study_metadata.experiment, //colname
                gs.session_info.gameID);
            };


    gs.session_timing.experiment_start = Date.now();

    // ====================================================================
    // INITIALIZE JSPSYCH
    // ====================================================================

    const jsPsych = initJsPsych({
        on_finish: function(data) {
            // Only called once when trial finishes
            gs.session_info.send_data(data, false, false); // true = final response
        },
        override_safe_mode: true,
        });


    // ====================================================================
    // VIDEO CONFIGURATION
    // ====================================================================

    // agentOrderCondition is an array of agent numbers [1, 2, 3, 4] in the order they appear
    // Each agent number (1-4) represents a specific risk/difficulty combination
    // colorOrderCondition maps to the displayed player order (colors shown to participant)

    let videoBlocks = gs.session_info.agentOrderCondition.map(agentNumber => ({
        player: agentNumber,  // Agent number (1-4)
        videos: []
    }));

    const allVideos = [];
    gs.session_info.trials.forEach(trial => {
        allVideos.push(trial.videoUrl);
        const block = videoBlocks.find(b => b.player === trial.agentNumber);
        if (block) {
            // Store the full trial data, not just the URL
            block.videos.push({
                url: trial.videoUrl,
                data: trial  // Store complete trial data for later use
            });
        } else {
            console.warn(`No videoBlock found for agentNumber ${trial.agentNumber}`);
        }
    });
    allVideos.push(gs.EXPERIMENT_CONFIG.EXAMPLE_VIDEO_PATH);

    // ====================================================================
    // VIDEO TRIAL CREATION FUNCTION
    // ====================================================================

    /**
     * Creates a video trial with rating questions
     * @param {string} videoPath - Path to the video file
     * @param {number} videoIndex - Index within the block
     * @param {number} actualPlayerNumber - Actual player type (1-4, from videoBlocks)
     * @param {number} displayedPlayerNumber - Player number shown to participant (1-4 in order)
     * @param {number} blockNumber - Block number (1-4)
     * @param {Object} ratingSliderConfig - Configuration for rating sliders
     * @returns {Array} - Array of trial objects
     */
    function createVideoTrial(videoPath, videoIndex, actualPlayerNumber, displayedPlayerNumber, blockNumber, ratingSliderConfig, trialData) {
        const trials = [];
        
        // Video viewing trial
        const videoTrial = {
            type: jsPsychVideoButtonResponse,
            stimulus: [videoPath],
            choices: ['Next'],
            prompt: '',
            response_ends_trial: true,
            trial_ends_after_video: false,
            on_load: function() {
                // Find the button and disable it initially
                const buttons = document.querySelectorAll('.jspsych-btn');
                buttons.forEach(function(btn) {
                    btn.disabled = true;
                    btn.style.opacity = '0.5';
                    btn.style.cursor = 'not-allowed';
                });

                // Enable button when video ends
                const video = document.querySelector('video');
                if (video) {
                    video.addEventListener('ended', function() {
                        buttons.forEach(function(btn) {
                            btn.disabled = false;
                            btn.style.opacity = '1';
                            btn.style.cursor = 'pointer';
                        });
                    });
                }
            },
            data: {
                study_phase: 'main_task',
                trial_phase: 'video_viewing',
                displayed_player_number: displayedPlayerNumber,
                actual_player_number: actualPlayerNumber,
                agent_number: trialData.agentNumber,
                risk_level: trialData.riskLevel,
                difficulty_level: trialData.difficultyLevel,
                model_name: trialData.modelName,
                amplitude: trialData.amplitude,
                color: trialData.color,
                question: trialData.question,
                block_number: blockNumber,
                video_index: videoIndex,
                video_file: videoPath
            }
        };
        trials.push(videoTrial);

        // Rating questions - using shared configuration
        const questions = [
            { id: 'risk', text: 'How much danger did you think Player ' + displayedPlayerNumber + ' was in overall?', subtext: 'Please think about how likely it seemed that the run could have ended in a crash at any point.', labels: ['Very unlikely to crash', 'Very likely to crash'] },
            { id: 'map_difficulty', text: 'How difficult did you think the obstacle course Player ' + displayedPlayerNumber + ' navigated was?', subtext: 'Please think about how likely players would be to crash on this obstacle course, regardless of how well this player performed this time.', labels: ['Very unlikely to crash', 'Very likely to crash'] },
            { id: 'enjoyment', text: 'How much did you enjoy watching this video?', subtext: 'Please tell us how fun it was to watch, compared to other videos you might watch of someone playing this game.', labels: ['Much less enjoyable than typical', 'Much more enjoyable than typical'] },
        ];

        var question = questions.find( q => q.id === gs.session_info.questionCondition);

        const ratingTrial = {
            type: jsPsychHtmlSliderResponse,
            stimulus: `<p style="font-size: 18px;"><strong>${question.text}</strong> <br> <i>${question.subtext}</i></p>`,
            labels: question.labels,
            ...ratingSliderConfig,
            data: {
                study_phase: 'main_task',
                trial_phase: 'rating',
                question: question.id,
                displayed_player_number: displayedPlayerNumber,
                actual_player_number: actualPlayerNumber,
                agent_number: trialData.agentNumber,
                risk_level: trialData.riskLevel,
                difficulty_level: trialData.difficultyLevel,
                model_name: trialData.modelName,
                amplitude: trialData.amplitude,
                color: trialData.color,
                block_number: blockNumber,
                video_index: videoIndex,
                video_file: videoPath
            },
            on_load: function() {
                const slider = document.querySelector('input[type="range"]');
                if (slider) {
                    slider.addEventListener('input', (event) => {
                        // Calculate time_elapsed manually for intermediate updates
                        // (jsPsych automatically adds time_elapsed when trial finishes)
                        const time_elapsed = gs.session_timing.experiment_start
                            ? Date.now() - gs.session_timing.experiment_start
                            : 0;
                        const intermediateData = {
                            ...ratingTrial.data,
                            response: event.target.value,
                            time_elapsed: time_elapsed,
                            is_slider_input: true
                        };
                        gs.session_info.send_data(intermediateData, false, false);
                    });
                }
            },
            on_finish: function(data) {
                gs.session_info.send_data(data, true, false); // true = final response
            }
        };
        trials.push(ratingTrial);

        return trials;
    }

    // ====================================================================
    // BUILD TIMELINE
    // ====================================================================

    const timeline = [];

    // ====================================================================
    // EXIT SURVEY DEFINITION
    // ====================================================================


    // Form validation functions are in utils.js

    const exitSurveyTimeline = [];

    const survey_questions = [
        { id: 'risk', text: 'What influenced how much danger you thought players were in?', label_id: 'risk_explanations'},
        { id: 'map_difficulty', text: 'What influenced how difficult you thought the obstacle courses were?', label_id: 'map_difficulty_explanations'},
        { id: 'enjoyment', text: 'What influenced how much you enjoyed watching each video?', label_id: 'enjoyment_explanations'},
        { id: 'competence', text: 'What influenced how well you thought players navigated the courses?', label_id: 'competence_explanations'},
        ];

    var survey_question = survey_questions.find( q => q.id === gs.session_info.questionCondition);

    // Page 1a: Your experience
    const experiencePageA = {
    type: jsPsychSurveyHtmlForm,
    preamble: "<h2>Your Experience</h2>",
    html: '<div class="instruction-text">' +
            SurveyHelpers.formContainer(
                SurveyHelpers.question(survey_question.text) +
                SurveyHelpers.textarea(survey_question.label_id, 4, true)
            ),
    data: { study_phase: "exit_survey", survey_page: "experience", question: "explanations" },
    on_load: setupFormValidation('jspsych-survey-html-form'),
    on_finish: function(data) {
        jsPsych.data.addProperties(data.response);

        gs.session_info.send_data(data, false, false);
        }
    };

    exitSurveyTimeline.push(experiencePageA);

    // Page 1b: perceived differences between players
    const experiencePageB = {
        type: jsPsychHtmlSliderResponse,
        data: {
            study_phase: "exit_survey",
            survey_page: "experience_difference",
            question: "playing_style_difference"
        },
        stimulus: '<p style="font-size: 18px;"><strong>How differently did you think the players navigated the trials?</strong></p>',
        labels: ['Not differently at all', 'Very differently'],
        min: 0,
        max: 100,
        start: 50,
        slider_width: 500,
        require_movement: true,
        on_finish: function(data) {
            gs.session_info.send_data(data, true, false);
        }
    };
    exitSurveyTimeline.push(experiencePageB);

    // Page 2: Gaming Experience
    const gamingExperiencePage = {
    type: jsPsychSurveyHtmlForm,
    data: {
        study_phase: "exit_survey",
        survey_page: "gaming_experience"
    },
    preamble: "<h2>Your Experience</h2>",
    html: `
        <div class="instruction-text" style="text-align:left; max-width:600px; margin:0 auto;">
            <p><strong>How familiar are you with Flappy Bird (the game that inspired this study)?</strong></p>
            <div style="margin: 0.5em 0;">
                <label><input type="radio" name="flappy_bird_familiarity" value="Never heard of it" required> Never heard of it</label><br>
                <label><input type="radio" name="flappy_bird_familiarity" value="Heard of it but never played" required> Heard of it but never played</label><br>
                <label><input type="radio" name="flappy_bird_familiarity" value="Played it a few times" required> Played it a few times</label><br>
                <label><input type="radio" name="flappy_bird_familiarity" value="Played it regularly" required> Played it regularly</label>
            </div>

            <p style="margin-top: 1.5em;"><strong>In the past three months, how often have you watched video game content (e.g., gameplay videos, streams, esports)?</strong></p>
            <div style="margin: 0.5em 0;">
                <label><input type="radio" name="video_game_watching_frequency" value="Never" required> Never</label><br>
                <label><input type="radio" name="video_game_watching_frequency" value="Rarely (less than once a month)" required> Rarely (less than once a month)</label><br>
                <label><input type="radio" name="video_game_watching_frequency" value="Occasionally (1-3 times per month)" required> Occasionally (1-3 times per month)</label><br>
                <label><input type="radio" name="video_game_watching_frequency" value="Regularly (1-3 times per week)" required> Regularly (1-3 times per week)</label><br>
                <label><input type="radio" name="video_game_watching_frequency" value="Very frequently (almost daily or daily)" required> Very frequently (almost daily or daily)</label>
            </div>
        </div>
    `,
    show_clickable_nav: true,
    allow_keys: false,
    button_label_next: 'Next',
    on_load: setupFormValidation('jspsych-survey-html-form'),
    on_finish: function(data) {
        // Add responses to all subsequent trials
        jsPsych.data.addProperties(data.response);

        gs.session_info.send_data(data, false, false);
        }
    };

    exitSurveyTimeline.push(gamingExperiencePage);

    // Page 3: Demographics
    const demographicsPage = {
    type: jsPsychSurveyHtmlForm,
    data: { study_phase: "exit_survey", survey_page: "demographics" },

    preamble: "<h2>Demographic Information</h2>",

    html: `
        <div class="instruction-text" style="text-align: left; max-width: 600px; margin: 0 auto;">
            <p><strong>What is your age?</strong></p>
            <input type="number" name="age" min="18" max="99" required
                style="width: 200px; font-family: inherit; font-size: 14px; padding: 8px;
                border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;">

            <p style="margin-top: 1.5em;"><strong>What is your gender?</strong></p>
            <label><input type="radio" name="gender" value="Male" required> Male</label><br>
            <label><input type="radio" name="gender" value="Female" required> Female</label><br>
            <label><input type="radio" name="gender" value="Non-binary" required> Non-binary</label><br>
            <label><input type="radio" name="gender" value="Other" required> Other</label>

            <p style="margin-top: 1.5em;"><strong>What is your race/ethnicity? (Select all that apply)</strong></p>
            <label><input type="checkbox" name="race[]" value="American Indian or Alaska Native"> American Indian or Alaska Native</label><br>
            <label><input type="checkbox" name="race[]" value="Asian"> Asian</label><br>
            <label><input type="checkbox" name="race[]" value="Black or African American"> Black or African American</label><br>
            <label><input type="checkbox" name="race[]" value="Hispanic or Latino"> Hispanic or Latino</label><br>
            <label><input type="checkbox" name="race[]" value="Native Hawaiian or Other Pacific Islander"> Native Hawaiian or Other Pacific Islander</label><br>
            <label><input type="checkbox" name="race[]" value="White"> White</label><br>
            <label><input type="checkbox" name="race[]" value="Other"> Other</label>
        </div>
    `,

    show_clickable_nav: true,
    allow_keys: false,
    button_label_next: "Next",

    // Custom validation for the race field
    on_load: setupFormValidation('jspsych-survey-html-form', function(form) {
        // Check that at least one race checkbox is selected
        const raceChecked = form.querySelectorAll('input[name="race[]"]:checked').length > 0;
        if (!raceChecked) {
            alert('Please select at least one option for race/ethnicity.');
            return false;
        }
        return true;
    }),
    on_finish: function(data) {
        // jsPsychSurveyHtmlForm automatically collects values, including checkboxes.
        // For checkboxes, the result is an ARRAY (ideal!)
        const resp = data.response;

        const age = resp.age;
        const gender = resp.gender;

        // Note: checkboxes with name="race[]" will create a key "race[]" in the response
        let participantRace = [];
        const raceData = resp['race[]'] || resp.race; // Handle both 'race[]' and 'race' keys
        if (raceData) {
            // If raceData exists (it's a string or an array), ensure it's an array.
            participantRace = Array.isArray(raceData) ? raceData : [raceData];
        }
        const demographicData = {
            age: age,
            gender: gender,
            race: participantRace
        };

        // Update the trial's stored response
        const newResponse = { ...resp, ...demographicData };

        jsPsych.finishTrial({
            ...data,
            response: newResponse
        });

        jsPsych.data.addProperties(newResponse);

        // Merge survey_response into data before sending
        const dataWithResponse = _.extend({}, data, { survey_response: newResponse });
        gs.session_info.send_data(dataWithResponse, false, false);
        }
    };

    exitSurveyTimeline.push(demographicsPage);

    // Page 4: Additional comments
    const commentsPage = {
    type: jsPsychSurveyHtmlForm,
    data: { study_phase: "exit_survey", survey_page: "comments" },
    preamble: "<h2>Additional Feedback</h2>",
    html: `
        <div class="instruction-text">
            <div style="text-align: left; max-width: 600px; margin: 0 auto;">
                <p><strong>Do you have any other comments or feedback to share with us?</strong></p>
                <p style="font-size: 0.9em; color: #666;">If you encountered any technical difficulties, please describe the issue below.</p>

                <textarea 
                    name="additional_feedback" 
                    rows="6"
                    style="width: 100%; font-family: inherit; font-size: 14px; padding: 8px; 
                           border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;"
                ></textarea>
            </div>
        </div>
    `,
    show_clickable_nav: true,
    button_label_next: "Next",

    on_finish: function(data) {
        const feedback = data.response.additional_feedback || "";

        const newResponse = {
            ...data.response,
            additional_feedback: feedback
        };
        // Update current trial before saving
        jsPsych.finishTrial({
            ...data,
            response: newResponse
        });
        // Also add properties globally for later trials.
        jsPsych.data.addProperties(newResponse);

        // Merge survey_response into data before sending
        const dataWithResponse = _.extend({}, data, { survey_response: newResponse });
        gs.session_info.send_data(dataWithResponse, false, false);
        }
    };

    exitSurveyTimeline.push(commentsPage);

    const exitSurvey= {
        timeline: exitSurveyTimeline,
        on_timeline_start: function() {
            // Record when main task completes and exit survey begins
            gs.session_timing.main_task_complete = Date.now();
            gs.session_timing.exit_survey_start = Date.now();
        }
    };

    // ====================================================================
    // 1. CONSENT PAGE
    // ====================================================================
    const consent = {
        data: { study_phase: "consent" },
        type: jsPsychHtmlButtonResponse,
        stimulus:
            "<div class='instruction-text'>" +
            '<h2>Rate That Video</h2>' +
            '<div style="text-align: left">' +
            "<p>Welcome! In this study, you will watch and rate short videos of individuals playing a simple game. The session should take approximately <b>5 minutes</b>.</p>" +
            "<p><i> Note: We recommend completing the study in Chrome. It has not been tested in other browsers.</i></p>" +
            "<div class='consent'>" +
            "<p>By clicking below, you are agreeing to take part in a study being conducted by cognitive scientists in the <b>Department of Psychology at Stanford University</b>. If you have questions about this research, please contact us at <a href='mailto:cogtoolslab.requester@gmail.com?subject=Flappy Bird 2 Study'>cogtoolslab.requester@gmail.com</a>. We will do our best to respond promptly and professionally.</p>" +
            "<ul>" +
            "<li>You must be at least 18 years old to participate.</li>" +
            "<li>Your participation is voluntary.</li>" +
            "<li>You may decline to answer any question or stop the study at any time without penalty.</li>" +
            "<li>Your responses are anonymous and will be analyzed only in aggregate form.</li>" +
            "</ul>" +
            "</div>" +
            "<p>Do you consent to participate in this study as described above?</p>" +
            "</div>" +
            "</div>",
        choices: ["Yes, I agree to participate"],
        margin_vertical: "30px",
        on_start: function () {
                gs.session_timing.consent_start = Date.now();
            },
        enable_button_after: 3000,
        on_finish: function() {
            // Record consent completion time
            gs.session_timing.consent_complete = Date.now();
            
            // Send consent data immediately
            // sendDataToServer(data, 'consent');
            
            // Enter fullscreen immediately after consent
            const element = document.documentElement;
            if (element.requestFullscreen) {
                element.requestFullscreen();
            } else if (element.mozRequestFullScreen) {
                element.mozRequestFullScreen();
            } else if (element.webkitRequestFullscreen) {
                element.webkitRequestFullscreen();
            } else if (element.msRequestFullscreen) {
                element.msRequestFullscreen();
            }
        }
    };
    timeline.push(consent);

    // ====================================================================
    // 2. MAIN EXPERIMENT
    // ====================================================================
    const mainExperimentTimeline = {
        timeline: []
    };

    // ====================================================================
    // 2a. PRELOAD VIDEOS
    // ====================================================================
    const preload = {
        data: { study_phase: "preload" },
        type: jsPsychPreload,
        video: allVideos,
        message: 'Loading videos. This may take a moment depending on your internet connection...',
        error_message: '<p>The experiment failed to load. Please try restarting your browser.</p>' +
                    '<p>If this error persists after 2-3 tries, please contact the experimenter.</p>',
        continue_after_error: false,
        show_progress_bar: true,
        max_load_time: gs.EXPERIMENT_CONFIG.MAX_PRELOAD_TIME,
        on_finish: function() {
            gs.session_timing.preload_complete = Date.now();
        }
    };
    mainExperimentTimeline.timeline.push(preload);

    // ====================================================================
    // 2b. INSTRUCTIONS
    // ====================================================================

    // Define instruction questions (generic versions without player number)
    const instruction_questions = [
        { id: 'risk', text: 'How much danger did you think the Player was in overall?', subtext: 'Please think about how likely it seemed that the run could have ended in a crash at any point.' },
        { id: 'map_difficulty', text: 'How difficult did you think the obstacle course the Player navigated was?', subtext: 'Please think about how likely players would be to crash on this obstacle course, regardless of how well this player performed this time.'},
        { id: 'enjoyment', text: 'How much did you enjoy watching this video?', subtext: 'Please tell us how fun it was to watch, compared to other videos you might watch of someone playing this game.' },
    ];

    var instruction_question = instruction_questions.find( q => q.id === gs.session_info.questionCondition);

    const instructions = {
        data: { study_phase: "instructions" },
        type: jsPsychInstructions,
        pages: [
            // --- PAGE 1: Overview ---
            "<div class='instruction-text'>" +
            "<h2>~ Welcome! ~</h2>" +
            "<p>In this study, you will watch and rate short videos of players navigating underwater obstacle courses in the game <b>Splashy Trials</b>.</p>" +
            "</div>",

            // --- PAGE 2: How the game works ---
            "<div class='instruction-text'>" +
            "<h2>~ How Splashy Trials work ~</h2>" +
            "<p>In Splashy Trials, players try to navigate obstacle courses without any collisions.</p>" +
            "<p>If a player hits a pipe or the ground, they lose immediately and the game ends.</p>" +
            "<div style='text-align: center; margin: 20px 0;'>" +
            "<video width='250' height='350' style='display: block; margin: 0 auto;' autoplay loop muted>" +
            "<source src='" + gs.EXPERIMENT_CONFIG.PATH_TO_VIDEOS + gs.EXPERIMENT_CONFIG.EXAMPLE_VIDEO_PATH + "' type='video/mp4'>" +
            "Your browser does not support the video tag." +
            "</video>" +
            "</div>" +
            "</div>",

            // --- PAGE 3: Questions ---
            "<div class='instruction-text'>" +
            "<h2>~ What you will be doing ~</h2>" +
            "<p>After watching each video, you will answer the following question:</p>" +
            "<p><b>" + instruction_question.text + "</b> <br> <i>" + instruction_question.subtext + "</i></p>" +
            "</div>",

            // --- PAGE 4: Four players ---
            "<div class='instruction-text'>" +
            "<h2>~ Meet the Splashers! ~</h2>" +
            "<p>You will watch <b>one video</b> from each of these four players. Each of them will be navigating a different obstacle course.</p>" +
            "<div style='display: flex; justify-content: center; align-items: center; gap: 30px; margin: 30px 0;'>" +
            "<div style='text-align: center;'>" +
            "<img src='assets/bird-upflap-" +  gs.session_info.colorOrderCondition[0] + ".png' style='width: 60px; height: 45px;'>" +
            "<p style='margin-top: 10px; font-weight: bold;'>Player 1</p>" +
            "</div>" +
            "<div style='text-align: center;'>" +
            "<img src='assets/bird-upflap-" +  gs.session_info.colorOrderCondition[1] + ".png' style='width: 60px; height: 45px;'>" +
            "<p style='margin-top: 10px; font-weight: bold;'>Player 2</p>" +
            "</div>" +
            "<div style='text-align: center;'>" +
            "<img src='assets/bird-upflap-" +  gs.session_info.colorOrderCondition[2] + ".png' style='width: 60px; height: 45px;'>" +
            "<p style='margin-top: 10px; font-weight: bold;'>Player 3</p>" +
            "</div>" +
            "<div style='text-align: center;'>" +
            "<img src='assets/bird-upflap-" +  gs.session_info.colorOrderCondition[3] + ".png' style='width: 60px; height: 45px;'>" +
            "<p style='margin-top: 10px; font-weight: bold;'>Player 4</p>" +
            "</div>" +
            "</div>" +
            "<p>Ready to dive in? Let's begin!</p>" +
            "</div>",
        ],
        show_clickable_nav: true,
        allow_keys: false,
        allow_backward: true,
        button_label_next: 'Next',
        button_label_previous: 'Previous',
        on_load: function() {
            // Initialize tracking set for pages that have been viewed for 5+ seconds
            if (!instructions.pagesCompleted) {
                instructions.pagesCompleted = new Set();
            }

            // Apply timing constraint to the first page
            const applyTimingConstraint = function(pageIndex) {
                // Use a small delay to ensure DOM is updated
                setTimeout(function() {
                    if (!instructions.pagesCompleted.has(pageIndex)) {
                        const nextButton = document.querySelector('#jspsych-instructions-next');
                        if (nextButton) {
                            nextButton.disabled = true;
                            nextButton.style.opacity = '0.5';
                            nextButton.style.cursor = 'not-allowed';

                            // Enable after 5 seconds
                            setTimeout(function() {
                                nextButton.disabled = false;
                                nextButton.style.opacity = '1';
                                nextButton.style.cursor = 'pointer';
                                instructions.pagesCompleted.add(pageIndex);
                            }, 2500);
                        }
                    }
                }, 50);
            };

            // Apply to the initial page (page 0)
            applyTimingConstraint(0);
        },
        on_page_change: function(current_page) {
            // This callback fires every time the page changes
            // Check if this page needs the timing constraint
            if (!instructions.pagesCompleted.has(current_page)) {
                // Small delay to ensure button is in the DOM
                setTimeout(function() {
                    const nextButton = document.querySelector('#jspsych-instructions-next');
                    if (nextButton) {
                        nextButton.disabled = true;
                        nextButton.style.opacity = '0.5';
                        nextButton.style.cursor = 'not-allowed';

                        // Enable after 5 seconds
                        setTimeout(function() {
                            nextButton.disabled = false;
                            nextButton.style.opacity = '1';
                            nextButton.style.cursor = 'pointer';
                            instructions.pagesCompleted.add(current_page);
                        }, 5000);
                    }
                }, 50);
            }
        },
        on_finish: function() {
            gs.session_timing.instructions_complete = Date.now();
        }
    };
    mainExperimentTimeline.timeline.push(instructions);

    // ====================================================================
    // 2c. BLOCKS WITH VIDEO TRIALS AND RATING QUESTIONS
    // ====================================================================

    videoBlocks.forEach((block, blockIndex) => {
        // Block introduction page
        // Note: Participants see "Player 1, 2, 3, 4" in order, but actual player type is tracked in data
        const blockIntro = {
            type: jsPsychHtmlButtonResponse,
            stimulus: '<h2>Our next Splasher is Player ' + (blockIndex +1) + '</h2>' +
                '<div style="text-align: center; margin: 20px 0;">' +
                '<img src="assets/bird-upflap-' +  gs.session_info.colorOrderCondition[blockIndex] + '.png" style="width: 80px; height: 60px;">' +
                '</div>' +
                '<p>Click "Next" when you are ready!</p>',
            choices: ['Next'],
            data: {
                study_phase: 'main_task',
                trial_phase: 'block_intro',
                block_number: blockIndex + 1,
                displayed_player_number: blockIndex + 1,  // What participant sees (1-4 in order)
                actual_player_number: block.player        // Actual player type (randomized)
            },
            on_start: function() {
                // Record main task start time on first block only
                if (blockIndex === 0 && !gs.session_timing.main_task_start) {
                    gs.session_timing.main_task_start = Date.now();
                }
            },
            on_load: function() {
                // Disable the "Next" button for 2 seconds
                setTimeout(function() {
                    const nextButton = document.querySelector('.jspsych-btn');
                    if (nextButton) {
                        nextButton.disabled = true;
                        nextButton.style.opacity = '0.5';
                        nextButton.style.cursor = 'not-allowed';

                        // Enable after 2 seconds
                        setTimeout(function() {
                            nextButton.disabled = false;
                            nextButton.style.opacity = '1';
                            nextButton.style.cursor = 'pointer';
                        }, 1000);
                    }
                }, 50);
            }
        };
        mainExperimentTimeline.timeline.push(blockIntro);

        // Add video trials for this block
        block.videos.forEach((video, videoIndex) => {
            const videoTrials = createVideoTrial(
                video.url,                         // video URL
                videoIndex,
                block.player,                      // actual player number (agent type)
                blockIndex + 1,                    // displayed player number (1-4 in order)
                blockIndex + 1,                    // block number
                gs.RATING_SLIDER_CONFIG,
                video.data                         // trial data from MongoDB
            );
            mainExperimentTimeline.timeline.push(...videoTrials);
        });

        // After Player 2 (blockIndex === 1), add comparison for Players 1 vs 2
        if (blockIndex === 1) {
            // Define comparison questions based on question condition
            const comparisonQuestions = {
                'risk': {
                    text: 'Which player was in the most danger overall?',
                    subtext: 'Please think about which run was more likely to have ended in a crash at any point.'
                },
                'map_difficulty': {
                    text: 'Which player had the most difficult obstacle course?',
                    subtext: 'Please think about which obstacle course was more likely to cause crashes for players, regardless of how well each player performed.'
                },
                'enjoyment': {
                    text: 'Which player was more enjoyable to watch?',
                    subtext: 'Please tell us which was more fun to watch.'
                },
            };

            const comparisonObj = comparisonQuestions[gs.session_info.questionCondition];

            // Comparison: Players 1 & 2
            const comparison_12 = {
                type: jsPsychHtmlButtonResponse,
                stimulus: '<div class="instruction-text">' +
                        '<h2>' + comparisonObj.text + '</h2>' +
                        // '<p><i>' + comparisonObj.subtext + '</i></p>' +
                        '<div style="display: flex; justify-content: center; gap: 50px; margin: 40px 0;">' +
                        '<div style="text-align: center;">' +
                        '<img src="assets/bird-upflap-' +  gs.session_info.colorOrderCondition[0] + '.png" style="width: 80px; height: 60px;">' +
                        '<p style="margin-top: 15px; font-weight: bold; font-size: 18px;">Player 1</p>' +
                        '</div>' +
                        '<div style="text-align: center;">' +
                        '<img src="assets/bird-upflap-' +  gs.session_info.colorOrderCondition[1] + '.png" style="width: 80px; height: 60px;">' +
                        '<p style="margin-top: 15px; font-weight: bold; font-size: 18px;">Player 2</p>' +
                        '</div>' +
                        '</div>' +
                        '</div>',
                choices: ['Player 1', 'Player 2'],
                data: {
                    study_phase: 'pairwise_comparison',
                    comparison_type: gs.session_info.questionCondition,
                    pair: '1_vs_2',
                    displayed_players: [1, 2],
                    actual_agent_types: [gs.session_info.agentOrderCondition[0],  gs.session_info.agentOrderCondition[1]],
                    player_mapping:  gs.session_info.agentOrderCondition
                },
                on_finish: function(data) {
                    gs.session_info.send_data(data, true, false);
                }
            };
            mainExperimentTimeline.timeline.push(comparison_12);
            
            // Attention Check: After comparing players 1 and 2
            const attentionCheck = {
                type: jsPsychSurveyHtmlForm,
                data: {
                    study_phase: 'attention_check',
                    check_type: 'midpoint_comprehension'
                },
                html: `
                    <div class="instruction-text" style="text-align: left; max-width: 600px; margin: 0 auto;">
                        <p><strong> Great job! Which of these tasks have you been performing in this study? </strong></p>
                        <label><input type="checkbox" name="attention_check[]" value="play_games"> Playing Splashy Trials.</label><br>
                        <label><input type="checkbox" name="attention_check[]" value="watch_gameplays"> Watching videos of players playing Splashy Trials.</label><br>
                    </div>
                `,
                button_label: "Continue",
                on_finish: function(data) {
                    // Extract the attention check responses
                    const resp = data.response;

                    // Handle checkbox responses (can be array or single value)
                    let selectedOptions = [];
                    if (resp.attention_check) {
                        selectedOptions = Array.isArray(resp.attention_check) ? resp.attention_check : [resp.attention_check];
                    }

                    // Store the selected options
                    data.attention_check_selected = selectedOptions;

                    // Determine which options were selected
                    data.selected_play_games = selectedOptions.includes('play_games');
                    data.selected_watch_gameplays = selectedOptions.includes('watch_gameplays');

                    // Correct answer is: watch_gameplays
                    const correctAnswers = ['watch_gameplays'];
                    const correctSelections = correctAnswers.every(ans => selectedOptions.includes(ans));
                    const noIncorrectSelections = !selectedOptions.includes('play_games');

                    // Mark if they passed the attention check
                    data.attention_check_passed = correctSelections && noIncorrectSelections;

                    // Send data to server
                    gs.session_info.send_data(data, true, false);
                }
            };
            mainExperimentTimeline.timeline.push(attentionCheck);
        }
        
        // After Player 4 (blockIndex === 3), add comparison for Players 3 vs 4
        if (blockIndex === 3) {
            // Define comparison questions based on question condition
            const comparisonQuestions = {
                'risk': 'Which player was in the most danger?',
                'map_difficulty': 'Which player had the most difficult obstacle course?',
                'enjoyment': 'Which player was more enjoyable to watch?',
                'competence': 'Which player navigated the course better?'
            };

            const comparisonQuestion = comparisonQuestions[gs.session_info.questionCondition];

            // Comparison: Players 3 & 4
            const comparison_34 = {
                type: jsPsychHtmlButtonResponse,
                stimulus: '<div class="instruction-text">' +
                        '<h2>' + comparisonQuestion + '</h2>' +
                        '<div style="display: flex; justify-content: center; gap: 50px; margin: 40px 0;">' +
                        '<div style="text-align: center;">' +
                        '<img src="assets/bird-upflap-' +  gs.session_info.colorOrderCondition[2] + '.png" style="width: 80px; height: 60px;">' +
                        '<p style="margin-top: 15px; font-weight: bold; font-size: 18px;">Player 3</p>' +
                        '</div>' +
                        '<div style="text-align: center;">' +
                        '<img src="assets/bird-upflap-' +  gs.session_info.colorOrderCondition[3] + '.png" style="width: 80px; height: 60px;">' +
                        '<p style="margin-top: 15px; font-weight: bold; font-size: 18px;">Player 4</p>' +
                        '</div>' +
                        '</div>' +
                        '</div>',
                choices: ['Player 3', 'Player 4'],
                data: {
                    study_phase: 'pairwise_comparison',
                    comparison_type: gs.session_info.questionCondition,
                    pair: '3_vs_4',
                    displayed_players: [3, 4],
                    actual_agent_types: [gs.session_info.agentOrderCondition[2], gs.session_info.agentOrderCondition[3]],
                    player_mapping: gs.session_info.agentOrderCondition
                },
                on_finish: function(data) {
                    gs.session_info.send_data(data, true, false);
                }
            };
            mainExperimentTimeline.timeline.push(comparison_34);
        }
    });

    // Add the main experiment timeline to the overall timeline
    timeline.push(mainExperimentTimeline);

    // ====================================================================
    // 3. EXIT SURVEY
    // ====================================================================
    
    timeline.push(exitSurvey);

    // ====================================================================
    // 4. CONCLUSION
    // ====================================================================
    const exitFullscreen = {
        type: jsPsychFullscreen,
        fullscreen_mode: false,
        data: { study_phase: "exit_fullscreen" }
    };
    timeline.push(exitFullscreen);

    const conclusion = {
        data: { 
            study_phase: "conclusion"
        },
        type: jsPsychHtmlButtonResponse,
        stimulus: '<h1>Thanks for participating in our experiment!</h1>' +
            '<p>Please click the <em>Submit</em> button to complete the study and return to prolific.</p>',
        choices: ['Submit'],
        on_finish: function(data) {
                // Only called once when trial finishes
                gs.session_timing.experiment_complete = Date.now();
                gs.session_info.send_data(data, false, true); // true = experiment finished
                window.onbeforeunload = null; // prevent warning message on redirect (erikb)                
                window.open('https://app.prolific.com/submissions/complete?cc=CSUVIWY7', '_self');
            }
    };
    timeline.push(conclusion);

    // ====================================================================
    // 5. RUN THE EXPERIMENT
    // ====================================================================
    jsPsych.run(timeline);

}); // close socket
} // close setupGame