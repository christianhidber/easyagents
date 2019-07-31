# Persistenz

## Aktueller Stand
```python
ppoAgent = PpoAgent(    gym_env_name = 'Berater-v1', fc_layers=(500,500,500), learning_rate=1e-4,
                        training=Training( num_iterations = 2500, num_episodes_per_iteration = 10,
                                           max_steps_per_episode = 50, num_epochs_per_iteration = 5 )

ppoAgent.train()
````
## Speichern via Callbacks: Olli's Vorschlag (Keras Style)
```python
ppoAgent = PpoAgent(    gym_env_name = 'Berater-v1', fc_layers=(500,500,500), leaarning_rate=1e-4 )

checkPointCallback = ModelCheckpoint('models', monitor='rewards|loss|steps', period=10, save_best_only=True) 

ppoAgent.train( config=TrainingConfig( num_iterations = 2500, num_episodes_per_iteration = 10,
                                       max_steps_per_episode = 50, num_epochs_per_iteration = 5 ),
                callbacks=[checkPointCallback] )
````
### Denkbare weitere Callbacks (Keras inspiriert):
* ModelCheckpoint: Speichern (diverse Strategien wie "bestes Modell" gemessen an reward / loss / steps)
* Jupyter: matplotlib Statistik Grafiken & Image State Rendering (wie jetzt)
* Mp4Logger: erzeugt MP4 Film ueber alle besuchten States in einer Episode
* AnsiLogger: erzeugt log fuer ANSI rendering
* CSVLogger: CSV von Statistik-Daten
* ReduceLROnPlateau, LearningRateScheduler
* EarlyStopping: dynamische Trainingsdauer (abhängig von statistik-werten)
* TensorBoard

# Das könnte heissen (Ansatz):
```python
ppoAgent = PpoAgent( gym_env_name = 'Berater-v1', fc_layers=(500,500,500) )

ppoAgent.train( train_callbacks= [PpoConfig(...), TrainingSchedule(...), ModelCheckPoint(...)],
                eval_callbacks=[PlaySchedule(...), JupyterVisualize(...)],
                api_callback=[LogAgentCalls, LogEnvCalls] )
 
ppoAgent.play( [PlaySchedule(...), JupyterMovie(...), Mp4Write(...)] ) 
```

auch möglich: laden eines CheckPoints und fortsetzen des Trainings von diesem Punkt an mit einem anderen 
TrainingSchedule / LearningRate:
```python
ppoAgent = PpoAgent( gym_env_name = 'Berater-v1', fc_layers=(500,500,500) )
ppoAgent.load(...)
ppoAgent.train( train_callbacks= [PpoConfig(...), TrainingSchedule(...), ModelCheckPoint(...)] )
```

### TrainingCallback:
* on_train_begin / on_train_end
* on_iteration_begin / on_iteration_end

### PlayCallback
* on_play_begin / on_play_end
* on_episode_begin / on_episode_end
* on_step_begin / on_step_end

### ApiCallback
* on_agent_apicall_begin / on_agent_apicall_end
* on_env_apicall_begin / on_env_apicall_end


### Agent
#### Vom Agent zu implementieren
* _play_episode(...)
* _train(...)
* _save(...) / _load(...)

#### Agent BaseClass (EasyAgent)
* train(...)
* play(...)
* save(...) / load(...)
* training_config
* training_statistics 
* play_config
* play_statistics

##### TrainingConfig
Kann von TrainingCallbacks gelesen / modifiziert werden => dynamische TrainingSchedules / LearningRate Adaption
* learning_rate
* reward_discount_gamma
* num_iterations
* num_episodes_per_iteration
* max_steps_per_episode
* num_epochs_per_iteration
* num_iterations_between_eval
* current_iteration
* training_done
* Algorithmus spezifische (aber nicht implementations-spezifische) ableitungen: zB PpoConfig mit
  zusaetzlichem Property num_training_steps_in_replay_buffer

##### TrainingStatistics
Kann von Callbacks gelesen / befüllt werden (hier können Callbacks weitere Statistiken erzeugen, 
und diese untereinander austauschen. Loss / rewards / steps werden immer befuellt)
* dictionary key : [(iteration,value)] (wird vor jedem train geloescht)
* loss => ['loss']
* rewards => ['rewards'] = [(iteration,(min,r,max)]
* steps => ['steps'] = [(iteration,(min,s,max)]

##### PlayConfig
Kann von PlayCallbacks gelesen / modifiziert werden
* num_episodes
* current_episode
* playing_done

##### PlayStatistics
Kann von Callbacks gelesen / befüllt werden
* dictionary key : [(episode,value)] (wird vor jedem play geloescht)
* rewards => ['rewards'] 
* steps => ['steps'] 

# Allgemeine Design Goals
### Leuchttürme
* Fokus: Trainieren, debuggen & evaluieren von Policies fuer (eigene) Gym Environments (nicht neue Algorithmen entwickeln)
* einfach & konsistent vor flexibel & mächtig
* "Keras für RL": 
    * Einbinden bestehender Agent Implementationen (TfAgents, OpenAI baselines, Huskarl, ...)
    * Gleiches Api / UI über alle Agents

#### Weitere Ziele
* "Keras für RL": 
    * Einfache Integration weiterer Libraries: einzelne Agents, gemeinsame Basisklasse fuer Library spezifische 
        Funktionen wie speichern oder eine Episode spielen.
    * Möglichst viel Code Library unabhängig / übergreifend verwendbar
* Trainieren, Ausfuehren, Persistenz von Modellen (& Konfigurationen) 
* Dynamische, Grafik-Updates während Training
* Lauffähig unter Jupyter (Colab) und command line
* Perspektivisch: Heuristiken fuer 
    * Wahl der Netzwerk Architektur
    * Wahl des Agents
    * Wahl der Training Strategie
    * Wahl der Evaluations Strategie
   
# Issues
* Default Callbacks ?
* Wie kann der Benutzer die NN-Architektur definieren jenseits von fc_layers ?
* Persistenz von Training- & Log-Configuration: Können wir Trainings-Konfiguration speichern / festhalten 
  (ie die Parametrisierung der Callbacks) ? So dass wir das gleiche Training reproduzieren können ?
   
