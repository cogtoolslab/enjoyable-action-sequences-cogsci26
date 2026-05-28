gs = {
    study_metadata: {
        project: "engaging-movies", // Mongo database for participant data
        experiment: "engaging-movies-cogsci-55", // Mongo collection for this release
        iteration: "supplement55_2",
        version: '1.0.0',
        study_description: 'Video gameplay ratings',
        date: new Date().toISOString().split('T')[0],
    },
    session_timing: {
        experiment_start: undefined,
        consent_start: undefined,
        consent_complete: undefined,
        instructions_start: undefined,
        instructions_complete: undefined,
        preload_complete: undefined,
        main_task_start: undefined,
        main_task_complete: undefined,
        exit_survey_start: undefined,
        experiment_complete: undefined,
    },
    session_info: {
        gameID: undefined,
        participantID: undefined,
        numTrials: undefined,
        questionCondition: undefined,
        agentOrderCondition: undefined,
        trajectoryPair: undefined,
        colorOrderCondition: undefined,
        trials: undefined,
        send_data: undefined
    },
    EXPERIMENT_CONFIG: {
        VIDEOS_PER_PLAYER: 1,
        MAX_PRELOAD_TIME: 120000,
        PATH_TO_VIDEOS: '',
        EXAMPLE_VIDEO_PATH: 'https://engaging-movies.s3.us-east-2.amazonaws.com/cogsci-stimuli/example.mp4'
    },
    RATING_SLIDER_CONFIG: {
        min: 0,
        max: 100,
        start: 50,
        slider_width: 500,
        require_movement: true,
    },
    comprehensionAttempts: 0,
    prolific_info: {
        prolificPID: undefined,
        prolificStudyID: undefined,
        prolificSessionID: undefined
    }
}
