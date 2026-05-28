var
    argv = require('minimist')(process.argv.slice(2)),
    fs = require('fs'),
    app = require('express')(),
    _ = require('lodash'),
    sendPostRequest = require('request').post;

////////// EXPERIMENT GLOBAL PARAMS //////////

var gameport;

if (argv.gameport) {
    gameport = argv.gameport;
    console.log('using port ' + gameport);
} else {
    gameport = 8864;
    console.log('no gameport specified: using 8864\nUse the --gameport flag to change');
}

try {
    var privateKey = fs.readFileSync('/etc/letsencrypt/live/cogtoolslab.org/privkey.pem'),
        certificate = fs.readFileSync('/etc/letsencrypt/live/cogtoolslab.org/cert.pem'),
        intermed = fs.readFileSync('/etc/letsencrypt/live/cogtoolslab.org/chain.pem'),
        options = { key: privateKey, cert: certificate, ca: intermed },
        server = require('https').createServer(options, app).listen(gameport),
        io = require('socket.io')(server, { allowEIO3: true }); // to support socket.io v2 clients
} catch (err) {
    console.log("cannot find SSL certificates; falling back to http");
    var server = app.listen(gameport),
        io = require('socket.io')(server);
}

app.get('/*', (req, res) => {
    console.log("REQUEST:", req.method, req.url);
    serveFile(req, res);
});

io.engine.on("connection_error", (err) => {
    console.log(err.req);      // the request object
    console.log(err.code);     // the error code, for example 1
    console.log(err.message);  // the error message, for example "Session ID unknown"
    console.log(err.context);  // some additional error context
});

io.on('connection', function (socket) {
    console.log('Client connected.');

    // Upon stimuli request, serve one stimulus packet.
    socket.on('getStims', function (data) {
        initializeWithTrials(socket, data.db_name, data.exp_name);
    });

    // Upon getting session data from client, write data to db.
    socket.on('currentData', function (data, db_name, exp_name, gameid) {
        console.log(gameid + ' currentData received: ' + JSON.stringify(data).substring(100, 200));
        writeDataToMongo(data, db_name, exp_name);
    });

});

// Do not serve local credential files.
var FORBIDDEN_FILES = ["auth.json"];

var serveFile = function (req, res) {
    var fileName = req.params[0] || 'index.html';
    if (FORBIDDEN_FILES.includes(fileName)) {
        // Don't serve files that contain secrets
        console.log("Forbidden file requested: " + fileName);
        return res.sendStatus(403);
    }
    console.log('\t :: Express :: file requested: ' + fileName);
    return res.sendFile(fileName, { root: __dirname });
};

function initializeWithTrials(socket, db_name, exp_name) {
    // Request one entry from the stimuli database to initialize this session.
    var gameid = UUID();
    sendPostRequest('http://localhost:6002/db/getstims', {
        json: {
            dbname: db_name,
            colname: exp_name,
            gameid: gameid
        }
    }, (error, res, body) => {
        if (!error && res.statusCode === 200 && typeof body !== 'undefined') {
            // send trial list (and id) to client
            var packet = {
                gameid: gameid, 
                participantID: body.participantID, 
                agentOrderCondition: body.agentOrderCondition,
                trajectoryPair: body.trajectoryPair,
                colorOrderCondition: body.colorOrderCondition,
                questionCondition: body.questionCondition,
                trials: body.trials
            };
            socket.emit('stims', packet);
            console.log("INITIALIZED EXPERIMENT ", exp_name, " :: ", gameid);
        } else {
            console.log(db_name);
            console.log(exp_name);
            console.log(`error getting stims: ${error} ${body}`);
        }
    });
}

var UUID = function () {
    var baseName = (Math.floor(Math.random() * 10) + '' +
        Math.floor(Math.random() * 10) + '' +
        Math.floor(Math.random() * 10) + '' +
        Math.floor(Math.random() * 10));
    var template = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx';
    var id = baseName + '-' + template.replace(/[xy]/g, function (c) {
        var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
    return id;
};

var writeDataToMongo = function (data, db_name, exp_name) {
    sendPostRequest(
        'http://localhost:6002/db/insert',
        {
            json: _.extend(
                {
                    dbname: db_name,
                    colname: exp_name
                },
                data)
        },
        (error, res, body) => {
            if (!error && res.statusCode === 200) {
                console.log(`sent data to store`);
            } else {
                console.log(`error sending data to store: ${error} ${body}`);
            }
        }
    );
};