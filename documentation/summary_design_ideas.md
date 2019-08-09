# Design Ideen für EasyAgents vNext 
Stand: 190809

## Leuchttürme
* "Trainieren, evaluieren & debuggen von Policies fuer (eigene) Gym Environments" vor 
  "neue Algorithmen entwickeln" 
* "einfach & konsistent" vor "flexibel & mächtig"
* "Keras für RL": 
    * Gleiches Api / UI über alle Agents
    * Einbinden bestehender Agent Implementationen (TfAgents, OpenAI baselines, Huskarl, ...)

## Szenario
* Simpel
````
agent = PpoAgent( "LineWorld-v0" )
agent.train( SingleEpisode() )
agent.train()
agent.save('...')
agent.load('...')
agent.play()
````
* Advanced
````
agent = PpoAgent( "LineWorld-v0", fc_layers=(500,250,50) )
agent.train( train=[Fast(), ModelCheckPoint(), ReduceLROnPlateau(), TensorBoard()],
             play=[JupyterStatistics(), JupyterRender(), Mp4()],
             api=[AgentApi()] )
````
    
## Leitideen

#### Frontend & Backend (scikit learn, matplotlib): 
  Trennung von "User Api" und "Implementations Api" mit Hilfe von getrennten Frontend
  und Backend Klassen-Hierarchien für die Agents. 
  Mögliche Klassenhierarchie:
  * Frontend: EasyAgent ( train(), play(), save(), load() )
    * PpoAgent
    * DqnAgent
    * ReinforceAgent
    * ...
  * Backend: BackendAgent
    * TfAgent
        * PpoTfAgent
        * DqnTfAgent
        * ...
    * BaselineAgent
    * HuskarlAgent
  * Konstruktor von FrontendAgent erhält als Argument den Namen des Gym-Environments.
    (Dies ist gut weil 1. Modell & Env voneinander abhängen, 2. Save das Env nicht
     speichert, 3. bei Load kann das Modell gegen ein allenfalls anders 
     instanziiertes Env geladen werden)
     
#### Alles ist ein Callback (Keras): 
  Features & Erweiterungen mittels Callbacks implementieren (analog Keras). 
  Folgende Callback Typen, jeweils mit einer NoOp Implementation.
  * Callback Types:
      * TrainCallback: 
        werden waehrend dem Training ausgefuehrt und können den Trainings-Verlauf beeinflussen
        * on _[train | iteration]_begin / _end
      * PlayCallback: 
        werden waehrende der Evaluation ausgefuehrt, kein Einfluss auf Training
        * on _[play | episode | step]_begin / _end
      * ApiCallback: 
        werden waehrend Training & Play ausgefuehrt, kein Einfluss auf Training.
        * on_apicall_begin / on_apicall_end
        * on_gym_[init | step | reset | render]_begin / _end
  * Callback-Ideen
    * Train Callbacks
        * ModelCheckpoint: Speichern (diverse Strategien wie "bestes Modell" gemessen an reward / loss / steps)
        * ReduceLROnPlateau, 
        * LearningRateScheduler
        * EarlyStopping: dynamische Trainingsdauer (abhängig von statistik-werten)
    * Play Callbacks
        * Jupyter: matplotlib Statistik Grafiken & Image State Rendering (wie jetzt)
        * Mp4Logger: erzeugt MP4 Film ueber alle besuchten States in einer Episode
        * AnsiLogger: erzeugt log fuer ANSI rendering
        * CSVLogger: CSV von Statistik-Daten
        * TensorBoard
    * Api Callbacks
#### Train Loop:
  * BackendAgent.train implementiert einen Basis Loop, 
  * Loop is parametrisiert durch TrainConfig
  * TrainConfig kann durch Callbacks verändert werden => erlaubt dynamisches stoppen
  * **TrainConfig**:
    * num_iterations / num_iterations_between_eval / num_episodes_per_iteration / num_epochs_per_iteration
    * learning_rate
    * reward_discount
  * Pro Algorithmus Ableitungen von TrainConfig möglich, enthält Algorithmus spezifische Werte die von 
    Algorithmus spezifischen Callbacks gelesen verändert werden könnten:
    * PpoTrainConfig
    * DqnTrainConfig
        * replay_buffer_size
        * num_steps_replay_buffer_preload
  * Keine Backend-spezifischen TrainConfig Ableitungen
  
#### Pre-Configuration [TrainConfig, PlayConfig, ApiConfig] (oz, chh):
* Thematisch zusammengehoerende Konfigurations-Parameter werden in einer Klasse zusammengefasst
* Unterstützt Kommunikation zwischen Frontend & Backend
* Jeweils gleiche Implementation für Frontend & Backend
* Mittels Ableitungen werden Default-Konfigurationen für verschiedene Szenarion definiert:
    * TrainConfig mit Ableitungen TrainSingleEpisode, TrainFast, TrainNormal,...
    * PlayConfig
        * num_episodes
        * max_steps_per_episode
    * ApiConfig mit Ableitungen AgentApi, GymApi, AllApis  
        
#### Statistics (...):
  * Dictionary zum Austausch von Daten zwischen den Callbacks. 
  * Kann von Callbacks gelesen / befüllt werden (hier können Callbacks weitere Statistiken erzeugen, 
    und diese untereinander austauschen. 
  * Erlaubt die Entwicklung von allgemeinen Visualisierungs-Callbacks fuer Training und Playback
  * CallbackType spezifisch
    * TrainStatistics: (Loss / rewards / steps werden von Train Loop immer befuellt)
        * dictionary key : [(iteration,value)] (wird vor jedem train geloescht)
        * loss => ['loss']
        * rewards => ['rewards'] = [(iteration,(min,r,max)]
        * steps => ['steps'] = [(iteration,(min,s,max)]   
    * PlayStatistics: (rewards, steps wird von PlayLoop immer befuelt)
        * dictionary key : [(episode,value)] (wird vor jedem play geloescht)
        * rewards => ['rewards'] 
        * steps => ['steps']  
  
#### Backend Factory (Keras ?):
  * Eine Library (OpenAI, TfAgents, Huskarl,...) wird durch die Implementation einer BackendAgentFactory 
    in EasyAgents integriert.
  * Eine Default Implementation von BackendAgentFactory waehlt aus "bekannten" Backends die jeweils
    "beste" Implementation
  * Die BackendFactory kann bei der Instanziierung des Agents definiert werden.

## Open Issues:
  * wie kann ein andere Netz-Architektur definiert werden (als fully connected) ?

  
