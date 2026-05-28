# Watch-and-Rate Experiment

This folder contains the browser experiment and Node services for the Splashy Trials watch-and-rate study. Participants watch short gameplay videos, rate each video on one assigned question condition, complete two pairwise comparisons, answer an exit survey, and are redirected back to Prolific.

The experiment is not a static web page. The browser connects to `app.js` over Socket.IO, and `app.js` requests stimulus assignments from `store.js`, which reads and writes MongoDB.

## Experiment Flow

1. Consent and fullscreen entry.
2. Video preload and instructions.
3. Four player blocks, with one video per player in the current release.
4. Rating after each video. The rating prompt is determined by `questionCondition` from the stimulus packet.
5. Pairwise comparisons after players 1 vs. 2 and players 3 vs. 4.
6. Attention check, exit survey, fullscreen exit, and Prolific completion redirect.

## Services

Run two Node processes from this directory:

```bash
npm install
npm run start:store
```

In a second terminal:

```bash
npm run start:app
```

Then open:

```text
http://localhost:8864/
```

`app.js` serves the experiment and listens on port `8864` by default. Use `--gameport` to choose another port:

```bash
node app.js --gameport 9000
```

`store.js` listens on port `6002`. The app server expects the store server at `http://localhost:6002`.

## MongoDB Credentials

`store.js` loads credentials from `../../auth.json`, relative to this folder. From the repository root, copy the template and fill in local MongoDB credentials:

```bash
cp auth.json.template auth.json
```

`auth.json` is ignored by git and should not be committed.

The current Mongo connection string is:

```text
mongodb://<user>:<password>@localhost:27017/
```

## Stimulus Packets

When the browser starts, it emits `getStims` with:

```js
{
  db_name: "stimuli",
  exp_name: gs.study_metadata.experiment
}
```

`store.js` selects the least-served document from that collection by sorting on `numGames`, marks it as served, and sends it to the browser. Each stimulus document should include:

- `participantID`
- `questionCondition`
- `agentOrderCondition`
- `trajectoryPair`
- `colorOrderCondition`
- `trials`

Each item in `trials` should include the fields used by `setup.js`, including `videoUrl`, `agentNumber`, `riskLevel`, `difficultyLevel`, `modelName`, `amplitude`, `color`, and `question`.

## Data Writes

The browser streams data throughout the experiment via the `currentData` Socket.IO event. `app.js` forwards each payload to `store.js`, which inserts it into:

```js
{
  dbname: gs.study_metadata.project,
  colname: gs.study_metadata.experiment
}
```

Study metadata, session timing, Prolific URL parameters, session assignment fields, and trial or survey responses are included in these inserted documents.

## Prolific

The experiment reads Prolific URL parameters from `PROLIFIC_PID`, `STUDY_ID`, and `SESSION_ID`. At the end of the study, the participant is redirected to the Prolific completion URL in `js/setup.js`.

Update the completion code in `js/setup.js` before deploying a new Prolific study.

## Assets And Videos

The checked-in player sprites live in `assets/` and are referenced by `colorOrderCondition`. The instruction example video is configured in `js/game_settings.js` as a public S3 URL.

Trial videos are not stored in this folder. They are loaded from the `videoUrl` fields in the MongoDB stimulus packet, so public replication requires those URLs to remain available or to be replaced with self-hosted URLs in the stimulus documents.

## Files

- `index.html`: browser entry point and script includes.
- `js/setup.js`: jsPsych timeline, Socket.IO data flow, ratings, comparisons, surveys, and completion redirect.
- `js/game_settings.js`: study metadata and client-side configuration.
- `app.js`: Express and Socket.IO app server.
- `store.js`: MongoDB-backed stimulus assignment and data insert service.
- `run_tmux.sh`: optional helper for running a Node file inside a tmux session.