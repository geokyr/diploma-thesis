# Platform

## Architecture
### Components Diagram
```mermaid
flowchart TD
  subgraph UI[Frontend]
    A1[Admin Tab: Metrics, Drift Status, Notifications]
    A2[User Tab: Map + Predictions]
  end

  subgraph ORCH[Orchestrator]
    O1[Simulation Clock]
    O2[Trip Feeder]
    O3[Error Calculator]
    O4[Drift Engine]
    O5[Drift Status]
  end

  subgraph MS[Model Service]
    M1[/predict/]
    M2[/retrain/]
    M3[/load/]
    M4[Preprocessing]
  end

  subgraph FS[File Store]
    D1[(parquet: train/test/rain)]
    D2[(models)]
  end

  User["👤 User"] --> UI
  ExternalDataStore["External Data Store"] -.-> FS
  UI -- REST poll --> ORCH
  UI -- REST /start --> ORCH
  UI -- REST /pause --> ORCH
  ORCH -- REST /predict(batch) --> MS
  ORCH -- REST /retrain --> MS
  ORCH -- REST /load --> MS
  MS -- writes/reads --> D2
  ORCH -- reads --> D1
  ORCH -- reads current model meta --> D2
  UI -- REST /user_predict --> ORCH
```

### Sequence Diagrams
#### Timelapse Prediction Loop
```mermaid
sequenceDiagram
  autonumber
  participant UI as Frontend
  participant ORCH as Orchestrator
  participant MS as Model Service
  participant FS as File Store

  UI->>ORCH: POST /start
  ORCH->>FS: Load test.parquet metadata
  loop Every 5 minutes simulation time or 500ms real time
    ORCH->>ORCH: Fetch next batch of trips
    ORCH->>MS: POST /predict {task, features[]}
    MS-->>ORCH: predictions[]
    ORCH->>ORCH: join with ground truth → abs errors
    ORCH->>ORCH: update rolling MAE, push to Drift Engine
    ORCH-->>UI: poll metrics_update{MAE_eta, MAE_fuel, MAE_stops}
    ORCH->>ORCH: if test→rain boundary: set "day change"
    ORCH-->>UI: notification{day_transition}
  end
```

#### Drift Detection → Data Collection → Retrain → Swap
```mermaid
sequenceDiagram
  autonumber
  participant ORCH as Orchestrator
  participant MS as Model Service
  participant UI as Frontend

  ORCH->>ORCH: errors[model] (smoothed stream)
  ORCH-->>ORCH: detectors fire, majority vote=drift at t
  ORCH->>ORCH: state[model]=confirmed, start "collecting" window
  ORCH-->>UI: notification{model, drift_detected, t}

  Note over ORCH: Continue predicting with old model to accumulate rain data

  ORCH->>ORCH: after N trips/time collected → prepare retrain dataset
  ORCH-->>UI: notification{model, retrain_started}
  ORCH->>MS: POST /retrain {task, data_index, params}
  MS-->>ORCH: 202 Accepted
  loop until done
    ORCH->>MS: GET /status?task=...
    MS-->>ORCH: {"progress": p%}
    ORCH-->>UI: retrain_status{model, p%}
  end
  MS-->>ORCH: retrain_done{artifact=...}
  ORCH->>MS: POST /load {task, version=new}
  ORCH->>ORCH: state[model]=swapped
  ORCH-->>UI: notification{model, swapped_to_vN}
  ORCH-->>UI: metrics show lower error vs drifted regime
```

#### User Query Map Origin/Destination at Current Sim Time
```mermaid
sequenceDiagram
  autonumber
  participant UI as Frontend
  participant ORCH as Orchestrator
  participant MS as Model Service

  UI->>ORCH: POST /user_predict {origin, dest, sim_time}
  ORCH->>ORCH: derive features (distance, hour_bin, etc.)
  ORCH->>MS: POST /predict {task, features}
  MS-->>ORCH: {eta, fuel, stops}
  ORCH-->>UI: prediction payload + current weather/drift badge
```

## Prompt
We want to make a platform based around the concept of a drift detection and mitigation process.

We have 3 different machine learning models: estimated time of arrival prediction, fuel consumption prediction, number of stops prediction. All three of them will most likely end up being XGBoost/CatBoost/LightGBM, but with different preprocessing steps, features and hyperparameters.

We have also generated some SUMO traffic data from Athens center map. One dataset is 10 hours of train data (which we used to train the models), one dataset is 10 hours of test data (which we will use on the platform to test the models), and one dataset is 10 hours of rain data (which is basically a concept drift scenario simulation, where the road friction is reduced from 1.0 to 0.4 to simulate rain, and will be used on the platform to test the models and detect drift).

The data is FCD with fuel and waiting time from emission output from SUMO in CSV format (all columns are merged in one file), for every timestep for every vehicle. We do some preprocessing to get the data in a trip format, where we only keep the source and destination coordinates, the hour bin where the trip started, the distance traveled and some more features, different per model. There are around 55-60k trips in each dataset, and each model has around 10-20 features.

We have hourly varied traffic generation periods for realistic traffic patterns. We add some noise to each dataset so it has some variance, and also use a different random seed for trip generation.

We will use the test data and then the rain data, one after the other and have the models predict on those trips. This will be done in a sped up fashion, in something like a timelapse, for example the whole 20 hours of data will be compressed into 2-3 mins of timelapse. This is for demonstration purposes, to show the concept drift and the mitigation process in a reasonable time.

Other than that, we will have graphs to monitor the performance of the models, and if possible these can be dynamically filled in, as the timelapse progresses, every second or so, and accompanied by labels with the metrics written out, everytime a value is calculated and added to the graphs. For example, since we have 3 different models/ML tasks, we can have a total of 3 graphs that show the MAE of each model, plus some labels for the latest metrics calculated.

This will be the admin tab of our frontend, and we will also have a user tab, where when clicked, the timelapse and simulation will pause. A map of central Athens will appear, where the user will be able to select a source and destination for a trip, and ask for predictions. We will then predict on that state, based on the models we have loaded and for that specific time, and return the predictions, sort of like a Google Maps type of application. For this to happen, other than the source and destination pairs, we will need the start time for the trip, which the backend will have, and we will also need some more information like distance of the route (might be able to calculate it using sumolib or traci on the fly, otherwise we will fix a list of 10 common routes with distances precalculated), or number of edges traversed (again might be able to calculate it using sumolib or traci on the fly).

At some point after the 10 hours of base test data ends, the day will change, and the rain scenario will begin. This should send a notification to the user, and if we then swap to the user tab, a rain effect can be added to it. Then, as the predictions continue with the base models on the drifted rain data, the graphs will show higher errors than before. Then, we will have a drift detection mechanism, that will be receiving the errors for each model as they are being sent to the graphs as well from the backend, and will detect drift based on the errors. This will happen independently for each model, and for each drift detection, there should be a notification that drift has been detected for that model. 

The drift detector works as follows. Feed detectors a smoothed‑error stream – turn each trip’s absolute prediction error into a short rolling‑average series so the signal is steady enough for statistical tests. Run four parallel detectors on that stream – ADWIN, Page‑Hinkley, KSWIN, plus a home‑built SPC control‑chart rule that flags when the signal stays above its baseline mean + k·σ for a sustained run. Declare drift on majority vote – log a single “ensemble” timestamp only when at least three detectors fire, filtering out lone false alarms.

After the drift detection happens for a model, and the relevant notifications/visuals are shown (for example, background or some overlap with red for drift detected, yellow for collecting, blue for retraining and green for swapped), we will wait for some time so that the graphs have enough high errors calculated and shown, and in order to "collect" more drifted rain data, so that we can retrain the model with some old test data and the new "collected" rain data. However, bare in mind, as is the case with the whole pipeline and platform, we already have all the data we want, we just want to make this somewhat realistic, so in real life, we would have to wait to collect more data before retraining, and that is what we are doing here as well.

After enough data has been collected for a model to retrain, a notification should go out that the model is being retrained as part of the drift mitigation process. The errors, on the meantime should remain high, as we are still predicting with the base model. When retraining is done for a model and it is added to the model registry, it should be swapped in the system, a notification should go out that the retrained model is ready to be used, and it should then continue with the predictions with this new model. The errors should start to fall down a bit, not to the base level, but to a level that is better than the first drifted errors with the base model.

We want to keep this in general simple, and not too complex. We don't know if there is a better way to do this, or any other ideas to suggest for improvement or changes. We are mainly using Python, and we will probably use FastAPI for the Backend, Dash Plotly or any other capable framework for the Frontend that integrates well with Python, River library for the Drift Component together and a FastAPI if needed for the communication. We will try not to set up a database, as we can do our job for now with filesystem models and data (maybe parquet the csvs for speed). We are open to ideas and to changes to make this more feasible and not extremely complex on an engineering aspect. Also, the errors can be stored at some point, like a file or in memory for the frontend.

The Backend probably is the one responsible for driving the simulation clock of the timelapse, for sending feature batches to model services in simulation order to predict, for receiving the predictions, matching them to the ground truth, computing the errors, sending them to the Frontend and to the Drift Component, managing the drift state for the 3 models (stable > confirmed > collecting > retraining > swapped), emitting websocket events to Frontend (tick, errors update, drift event, retrain status, day transition for drift, etc), providing rest endpoints for run control (start, pause, restart), etc. These are all ideas and thoughts and can be reduced, modified or removed if not necessary.

The models will be served on the Model Serving Components running FastAPI and Uvicorn or any other capable framework, that will load the model weights from the Database and also run the code for the preprocessing steps, and serve some endpoints, like /predict and /retrain, and maybe /status for training progress, or /load to load model weights. The Backend will be orchestrating the whole process, and the Frontend will be a simple dashboard with the graphs and metrics, the notifications, and the user tab to request a prediction for a specific trip. The notifications can be timestamped and include a descriptive message.

Finally, the ground truth should be considered as available immediately, and not in the future, so we can use it instantly to calculate the errors, even though this is not realistic.

For a general orchestration, using Docker is a good idea, with various containers for all the components, and possibly a docker compose file to run the whole platform. But if there is another better or simper way to do this, let me know.

Give me a good plan for this, the architecture, the components, their roles, what communications will go on, how it will be orchestrated, what tools to actually use, since I'm open to suggestions, and if its feasible to do this with the tools suggested, other ideas for changes or improvements, etc. Get technical because I need it to make sure this is possible to do, and there are tools to do it, so be detailed, without becoming too complex or over engineering it.

This is an overview of a platform I wrote. I need you to read it, understand it, process it, solve any problems/questions etc, and then help me present this to my supervisors. I'd possibly like some diagrams, like a component diagram with the communications noted, some sequence diagrams, a general plan for the architecture and even written text as an overview or I dont know what. You know better. Give me what Im asking for, but keep in mind that the goal is to do something that is worth it, isn't extremely complex since we are undergraduate students doing this for our master or diploma thesis, but keeps a level of realism with some room for future improvements.
