# Notes

## Platform

### Overview
We want to make a platform based around the concept of a drift detection and mitigation process.

We have 3 different machine learning models: estimated time of arrival prediction, fuel consumption prediction, number of stops prediction. All three of them will be  XGBoost or LightGBM, but with different preprocessing steps, features and hyperparameters.

We have also generated some SUMO traffic data from Athens center map. One dataset is 10 hours of train data (which we used to train the models), one dataset is 10 hours of test data (which we will use on the platform to test the models), and one dataset is 10 hours of rain data (which is basically a concept drift scenario simulation, where the road friction is reduced from 1.0 to 0.4 to simulate rain, and will be used on the platform to test the models and detect drift).

The data is FCD with fuel and waiting time from emission output from SUMO in CSV format (all columns are merged in one file), for every timestep for every vehicle. We do some preprocessing to get the data in a trip format, where we only keep the source and destination coordinates, the time the trip started, the distance traveled and some more features, different per model. There are around 55-60k trips in each dataset, and for example one of the models has 55 features.

Following is the description of the dataset from the Zenodo release.

  This release provides 3 trace files containing Floating Car Data (FCD) from 10-hour microscopic traffic simulations of central Athens, generated with the SUMO (Simulation of Urban MObility) toolkit.

  Each simulation reflects distinct conditions and configurations.

  - train-fcd - Training simulation on the base network
  - test-fcd - Test simulation on the base network
  - rain-fcd - Simulation on the rain-modified network (with reduced friction on all lanes)

  These files are now available in both CSV and Parquet format. The output is organized by timestep and for each vehicle that is in the simulation at that timestep a row of its id, x, y, speed, lane, odometer, fuel, and waiting time is logged.

  Each simulation has a duration of 10 hours, spanning from morning to afternoon. There is a traffic pattern applied, with morning and afternoon peaks and lower mid-day activity. This traffic pattern is described by a list of per-hour traffic generation periods. On that list, some noise is added for each simulation, using a different random seed for each one. The same random seed is also used together with the traffic generation periods to generate the random trips used for each simulation. The goal was to introduce realistic variability, without going completely off a common traffic pattern.

  For the generation of this synthetic dataset, the following tools were used.

  - osmGet – Extract OpenStreetMap data from a bounding box over central Athens
  - osmBuild – Convert OSM data into a SUMO-compatible network
  - randomTrips – Generate random trips based on per-hour traffic generation periods
  - sumo – Run the microscopic simulation and export FCD data

We will use the test data and then the rain data, one after the other and have the models predict on those trips. This will be done in a sped up fashion, in something like a timelapse, for example the whole 20 hours of data will be compressed into 2-3 mins of timelapse. This is for demonstration purposes, to show the concept drift and the mitigation process in a reasonable time frame.

Other than that, we will have graphs to monitor the performance of the models, and if possible these can be dynamically filled in, as the timelapse progresses, every second or so, and accompanied by labels with the latest metrics written out, everytime a value is calculated and added to the graphs. For example, since we have 3 different models/ML tasks, we can have a total of 3 graphs that show the MAE of each model, plus the latest calculated metrics labels.

This will be the admin tab of our frontend, and we will also have a user tab, where when clicked, the timelapse and simulation will pause. A map of central Athens will appear, where the user will be able to select a source and destination for a trip, and ask for predictions. We will then predict on that state, based on the models we have loaded and for that specific time, and return the predictions, sort of like a Google Maps type of application. For this to happen, other than the source and destination pairs, we will need the start time for the trip, which the backend will have, and we will also need some more information like distance of the route (might be able to calculate it using sumolib or traci on the fly, otherwise we will fix a list of 10 common routes with distances precalculated), or number of edges traversed (again might be able to calculate it using sumolib or traci on the fly).

At some point after the 10 hours of base test data ends, the day will change, and the rain scenario will begin. This should send a notification to the user, and if we then swap to the user tab, a rain effect can be added to it. Then, as the predictions continue with the base models on the drifted rain data, the graphs will show higher errors than before. Then, we will have a drift detection mechanism, that will be receiving the errors for each model as they are being sent to the graphs as well from the backend, and will detect drift based on the errors. This will happen independently for each model, and for each drift detection, there should be a notification that drift has been detected for that model.

The drift detector works as follows. Feed detectors a smoothed‑error stream – turn each trip’s absolute prediction error into a short rolling‑average series so the signal is steady enough for statistical tests. Run four parallel detectors on that stream – ADWIN, Page‑Hinkley, KSWIN, plus a home‑built SPC control‑chart rule that flags when the signal stays above its baseline mean + k·σ for a sustained run. Declare drift on majority vote – log a single “ensemble” timestamp only when at least three detectors fire, filtering out lone false alarms. This is subject to change, but this might not matter in terms of the rest of the platform architecture.

After the drift detection happens for a model, and the relevant notifications/visuals are shown (for example, background or some overlap with red for drift detected, yellow for collecting, blue for retraining and green for swapped), we will wait for some time so that the graphs have enough high errors calculated and shown, and in order to "collect" more drifted rain data, so that we can retrain the model with some old test data and the new "collected" rain data. However, bare in mind, as is the case with the whole pipeline and platform, we already have all the data we want, we just want to make this somewhat realistic, so in real life, we would have to wait to collect more data before retraining, and that is what we are doing here as well.

After enough data has been collected for a model to retrain, a notification should go out that the model is being retrained as part of the drift mitigation process. The errors, on the meantime should remain high, as we are still predicting with the base model. When retraining is done for a model and it is added to the model registry, it should be swapped in the system, a notification should go out that the retrained model is ready to be used, and it should then continue with the predictions with this new model. The errors should start to fall down a bit, not to the base level, but to a level that is better than the first drifted errors with the base model.

We want to keep this in general simple, and not too complex. We don't know if there is a better way to do this, or any other ideas to suggest for improvement or changes. We are mainly using Python, and we will probably use FastAPI for the Backend, Dash Plotly or any other capable framework for the Frontend that integrates well with Python, River library for the Drift Component together and a FastAPI if needed for the communication. We will try not to set up a database, as we can do our job for now with filesystem models and data (parquet for speed). We are open to ideas and to changes to make this more feasible and not extremely complex on an engineering aspect. Also, the errors can be stored at some place, like a file or in memory for the frontend.

The Backend probably is the one responsible for driving the simulation clock of the timelapse, for sending feature batches to the predictors in simulation order to predict, for receiving the predictions, matching them to the ground truth, computing the errors, sending them to the Frontend and to the Drift Component, managing the drift state for the 3 models (stable > confirmed > collecting > retraining > swapped), polling events to Frontend (tick, errors update, drift event, retrain status, day transition for drift, etc), providing rest endpoints for run control (start, pause, restart), etc. These are all ideas and thoughts and can be reduced, modified or removed if not necessary.

The models will be served on the Model Serving Components running FastAPI and Uvicorn or any other capable framework, that will load the model weights from the filesystem and also run the code for the preprocessing steps, and serve some endpoints, like /predict and /retrain, and maybe /status for training progress, or /load to load model weights. The Backend will be orchestrating the whole process, and the Frontend will be a simple dashboard with the graphs and metrics, the notifications, and the user tab to request a prediction for a specific trip. The notifications can be timestamped and include a descriptive message.

Finally, the ground truth should be considered as available immediately, and not in the future, so we can use it instantly to calculate the errors, even though this is not realistic.

For a general orchestration, using Docker is a good idea, with various containers for all the components, and possibly a docker compose file to run the whole platform. But if there is another better or simper way to do this, let me know.

Give me a good plan for this, the architecture, the components, their roles, what communications will go on, how it will be orchestrated, what tools to actually use, since I'm open to suggestions, and if its feasible to do this with the tools suggested, other ideas for changes or improvements, etc. Get technical because I need it to make sure this is possible to do, and there are tools to do it, so be detailed, without becoming too complex or over engineering it.

Give me what I'm asking for, but keep in mind that the goal is to do something that is worth it, isn't extremely complex since we are undergraduate students doing this for a master or diploma thesis, but keeps a level of realism with some room for future improvements.

Here's an overview of what I'm planning to build. Take the overall plan as a guide, and the more technical details can be adjusted to fit whatever we decide to work with, so it's not an exact spec, but rough guidelines. Can you help me start with this and provide an MVP of this architecture, so that I can work on it iteratively later and refine it to match the final wanted product? Also make sure to plan on the iterations, and how this can be built in steps, starting from a working core and having laid out a proper skeleton that will be easy to extend and modify later down the road.

The diagrams and the rest of the information are just possible options, you can use them as a guide, but not stick to them, feel free to change them to fit whatever you consider is better.

I'd like to use docker and docker compose. I also use uv and a pyproject.toml to handle the envs, with Python 3.12.11. I already have one on my root for the env I was using on my experiments, and I also have the thesis/ folder which contains all my library code that I used on the experiments. This is installed editable on the environment so I can do global imports with safety. We can add folders for the backend, UI, and whatever else is needed inside the thesis/ folder.

I'm not sure how we should handle the different environments for the different containers/components, but use uv and pyproject.toml if possible. If it makes sense for them to be separate, and different from the root environment, then do that.Also make the dockerfiles efficient, so that they don't have to be rebuilt every time, and only the changed files are rebuilt.

I'm also not sure how we should handle the data and models, but I think using volumes and passing them to each container that needs them is a good idea.

Also keep in mind, the drift service will be done by a colleague, keep it as a blackbox for now, but design around it and handle the communication with it, and my colleague can later reform his code to fit our needs. The UI will also be designed by a colleague, so keep it simple and efficient, but being able to see the interactions and replies from the backend.

Keep it simple. I need you to plan this and explain everything in detail before proceeding with the implementation. Then we can start building on top of that.

### Components Diagram
```mermaid
flowchart TD
  subgraph UI[Frontend]
    A1[Admin Tab: Metrics, Drift Status, Notifications]
    A2[User Tab: Map + Predictions]
  end

  subgraph BE[Backend]
    O1[Simulation Clock]
    O2[Trip Feeder]
    O3[Error Calculator]
    O4[Drift Engine]
    O5[Drift Status]
  end

  subgraph PR[Predictor]
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
  UI -- REST poll --> BE
  UI -- REST /start --> BE
  UI -- REST /pause --> BE
  BE -- REST /predict(batch) --> PR
  BE -- REST /retrain --> PR
  BE -- REST /load --> PR
  PR -- writes/reads --> D2
  BE -- reads --> D1
  BE -- reads current model meta --> D2
  UI -- REST /user_predict --> BE
```

### Sequence Diagrams

#### Timelapse Prediction Loop
```mermaid
sequenceDiagram
  autonumber
  participant UI as Frontend
  participant BE as Backend
  participant PR as Predictor
  participant FS as File Store

  UI->>BE: POST /start
  BE->>FS: Load test.parquet metadata
  loop Every 5 minutes simulation time or 500ms real time
    BE->>BE: Fetch next batch of trips
    BE->>PR: POST /predict {task, features[]}
    PR-->>BE: predictions[]
    BE->>BE: join with ground truth → abs errors
    BE->>BE: update rolling MAE, push to Drift Engine
    BE-->>UI: poll metrics_update{MAE_eta, MAE_fuel, MAE_stops}
    BE->>BE: if test→rain boundary: set "day change"
    BE-->>UI: notification{day_transition}
  end
```

#### Drift Detection → Data Collection → Retrain → Swap
```mermaid
sequenceDiagram
  autonumber
  participant BE as Backend
  participant PR as Predictor
  participant UI as Frontend

  BE->>BE: errors[model] (smoothed stream)
  BE-->>BE: detectors fire, majority vote=drift at t
  BE->>BE: state[model]=confirmed, start "collecting" window
  BE-->>UI: notification{model, drift_detected, t}

  Note over BE: Continue predicting with old model to accumulate rain data

  BE->>BE: after N trips/time collected → prepare retrain dataset
  BE-->>UI: notification{model, retrain_started}
  BE->>PR: POST /retrain {task, data_index, params}
  PR-->>BE: 202 Accepted
  loop until done
    BE->>PR: GET /status?task=...
    PR-->>BE: {"progress": p%}
    BE-->>UI: retrain_status{model, p%}
  end
  PR-->>BE: retrain_done{artifact=...}
  BE->>PR: POST /load {task, version=new}
  BE->>BE: state[model]=swapped
  BE-->>UI: notification{model, swapped_to_vN}
  BE-->>UI: metrics show lower error vs drifted regime
```

#### User Query Map Origin/Destination at Current Sim Time
```mermaid
sequenceDiagram
  autonumber
  participant UI as Frontend
  participant BE as Backend
  participant PR as Predictor

  UI->>BE: POST /user_predict {origin, dest, sim_time}
  BE->>BE: derive features (distance, hour_bin, etc.)
  BE->>PR: POST /predict {task, features}
  PR-->>BE: {eta, fuel, stops}
  BE-->>UI: prediction payload + current weather/drift badge
```

### Timelapse Design
- Time warp: 20h -> 180s (400x)
- Tick: 150ms real time so 1m simulation time
- On each tick, pull all trips within the window (1m)
- Send batch of features to predictor /predict
- Compute rolling metrics every 1 second and update the UI (1 Hz)

### Datasets
```
data/
  test-fcd.parquet
  rain-fcd.parquet
```

### Model Registry
```
models/
  eta/
    stable/
      model.joblib
      metadata.json
    retrain/
      model.joblib
      metadata.json
  fuel/
    stable/
      model.joblib
      metadata.json
    retrain/
      model.joblib
      metadata.json
  stops/
    stable/
      model.joblib
      metadata.json
    retrain/
      model.joblib
      metadata.json
```

## Archive

### Environment Setup
To construct the environment with uv, the following commands were used.
```bash
uv init --bare --package --python 3.12.9
uv python pin 3.12.9
echo "" >> pyproject.toml
echo "[tool.hatch.build.targets.wheel]" >> pyproject.toml
echo 'packages = ["thesis"]' >> pyproject.toml
uv add catboost==1.2.8 eclipse-sumo==1.24.0 ipykernel==6.29.5 lightgbm==4.6.0 matplotlib==3.10.1 numpy==2.2.3 optuna==4.4.0 pandas==2.2.3 pyarrow==21.0.0 requests==2.32.4 scikit-learn==1.6.1 scipy==1.15.2 seaborn==0.13.2 xgboost==2.1.4
```

### Closure Drift
When trying out the closure drift scenario, there were a few problems with the implementation and the quality of the data generated. Two options were tried, and both of them had their own problems, leading to the conclusion that it was not possible to create a realistic closure scenario that would be strong enough to be detected by the drift detector, while also not completely messing up with the traffic patterns.

Option one was to use a rerouter on some lanes that would make the act as closed. This was done by using the [closingLaneReroute](https://sumo.dlr.de/docs/Simulation/Rerouter.html#closing_a_lane) rerouter. This however meant that cars would be inserted on the network at the time they were supposed to leave based on the routes file, calculate a route and then while the car was following the route, if it had a closed lane on it, it would be blocked and not move. This could be observed on the gui, where cars would be first at green traffic lights and would not move, up until the point where 300 seconds would pass and the car would get teleported to the next lane. This was caused by the fact that cars didn't have a rerouter device on them, but adding one could possibly interfere with the whole simulation, as other cars would also change their, calculated at insertion time, routes and alter the network traffic behavior, when compared to the base scenario.

Option two was to generate a new network where the closed lanes or edges would be simply removed, and the junctions would be recalculated automatically, using the [netedit](https://sumo.dlr.de/docs/Netedit/index.html) tool. However, because this was changing the network capacity, the vehicles were getting inserted at different times than before, because of the reduced capacity and higher traffic generated by the missing lanes or edges. It was therefore hard to keep a similar traffic behavior, and when trying to tune the traffic to have a similarity to the base scenario, while also introducing a drift because of closed lanes or edges, it was hard to find a combination of lanes or edges to close. In some cases, closing whole roads, even main ones like Panepistimiou, would not lead to much change. In other cases, closing some smaller edges/lanes would completely bottleneck the network traffic flow. It was therefore hard to find a realistic scenario of closed edges that made sense, while also being strong enough to be detected by the drift detector, while not completely messing up with the traffic patterns. Betweeness centrality and other similar network metrics were utilized to give a better idea on what lanes or edges would be a good fit for closure, but it was again hard to find a realistic combination, like an event happening around an area, a whole edge/road closed for works or metro works around a block/square etc.

Finally, in almost every closure scenario that was simulated, the results from the models were not as expected, since frequently the drifted scenario would return better results than the scenario where we had retrained the models on the drift data. Therefore, it was decided not to include this drift scenario on this project.
