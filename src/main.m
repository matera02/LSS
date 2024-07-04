%%
% Caricamento dei dati di training
trainData = readtable('data/DailyDelhiClimateTrain.csv');

% Caricamento dei dati di test
testData = readtable('data/DailyDelhiClimateTest.csv');

isNormalizzato = 0; %1 è true, 0 è false 
target = 'meantemp';

fprintf("TARGET: %s\n", target);

fprintf("Modello di regressione lineare\n");
d = 1;
Util.stampa_modelli(trainData, testData, target, isNormalizzato, d);

fprintf("Modello di regressione polinomiale, d: %d\n", 2);
d = 2;
Util.stampa_modelli(trainData, testData, target, isNormalizzato, d);

fprintf("Modello di regressione polinomiale, d: %d\n", 3);
d = 3;
Util.stampa_modelli(trainData, testData, target, isNormalizzato, d);

fprintf("Modello di regressione polinomiale, d: %d\n", 4);
d = 4;
Util.stampa_modelli(trainData, testData, target, isNormalizzato, d);
%%

target = 'mean_pressure';
fprintf("TARGET: %s\n", target);

fprintf("Modello di regressione lineare\n");
d = 1;
Util.stampa_modelli(trainData, testData, target, isNormalizzato, d);

fprintf("Modello di regressione polinomiale, d: %d\n", 2);
d = 2;
Util.stampa_modelli(trainData, testData, target, isNormalizzato, d);

fprintf("Modello di regressione polinomiale, d: %d\n", 3);
d = 3;
Util.stampa_modelli(trainData, testData, target, isNormalizzato, d);

fprintf("Modello di regressione polinomiale, d: %d\n", 4);
d = 4;
Util.stampa_modelli(trainData, testData, target, isNormalizzato, d);



%%
target = 'meantemp';
d = 1;
isNormalizzato = 0;
maxit = 100;

fprintf("TARGET: %s\n", target)
fprintf("DATI NON NORMALIZZATI\n")
fprintf("Modello di regressione lineare\n")
Util.stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit);

d = 2;
fprintf("Modello di regressione polinomiale, d = %d\n", d)
Util.stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit);

d = 3;
fprintf("Modello di regressione polinomiale, d = %d\n", d)
Util.stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit);

d = 4;
fprintf("Modello di regressione polinomiale, d = %d\n", d)
Util.stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit);

isNormalizzato = 1;
d = 1;
fprintf("TARGET: %s\n", target)
fprintf("DATI NORMALIZZATI\n")
fprintf("Modello di regressione lineare\n")
Util.stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit);


d = 2;
fprintf("Modello di regressione polinomiale, d = %d\n", d)
Util.stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit);


d = 3;
fprintf("Modello di regressione polinomiale, d = %d\n", d)
Util.stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit);

d = 4;
fprintf("Modello di regressione polinomiale, d = %d\n", d)
Util.stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit);



target = 'mean_pressure';
d = 1;
isNormalizzato = 0;
maxit = 100;

fprintf("TARGET: %s\n", target)
fprintf("DATI NON NORMALIZZATI\n")
fprintf("Modello di regressione lineare\n")
Util.stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit);

d = 2;
fprintf("Modello di regressione polinomiale, d = %d\n", d)
Util.stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit);

d = 3;
fprintf("Modello di regressione polinomiale, d = %d\n", d)
Util.stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit);

d = 4;
fprintf("Modello di regressione polinomiale, d = %d\n", d)
Util.stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit);

isNormalizzato = 1;
d = 1;
fprintf("TARGET: %s\n", target)
fprintf("DATI NORMALIZZATI\n")
fprintf("Modello di regressione lineare\n")
Util.stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit);


d = 2;
fprintf("Modello di regressione polinomiale, d = %d\n", d)
Util.stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit);


d = 3;
fprintf("Modello di regressione polinomiale, d = %d\n", d)
Util.stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit);

d = 4;
fprintf("Modello di regressione polinomiale, d = %d\n", d)
Util.stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit);












