classdef Modello
    
    properties
        trainData;
        testData;
        XTrain;
        yTrain;
        XTest;
        yTest;
        isNormalizzato;
        metodo;
        d; %grado del polinomio
        yPred;
        rmse;
        condizionamento;
        tempo;
    end

    methods (Access = private)
        % Vado a settare la variabile target
        function [yTrain, yTest] = setVariabileTarget(obj, idx_target)
            yTrain = obj.trainData(:, idx_target);
            yTest = obj.testData(:, idx_target);
            yTrain = table2array(yTrain);
            yTest = table2array(yTest);
        end

        function [XTrainNorm, yTrainNorm, XTestNorm, yTestNorm] = normalizzaDati(obj, XTrain, XTest)
            % Normalizzo minMax in base ai dati di training
            XMin = min(XTrain);
            XMax = max(XTrain);
            yMin = min(obj.yTrain);
            yMax = max(obj.yTrain);

            XTrainNorm = Util.normalizza_minMax(XTrain, XMin, XMax);
            yTrainNorm = Util.normalizza_minMax(obj.yTrain, yMin, yMax);
            XTestNorm = Util.normalizza_minMax(XTest, XMin, XMax);
            yTestNorm = Util.normalizza_minMax(obj.yTest, yMin, yMax);

            % Ho provato anche con normalizzazione standard
            %XTrainNorm = Util.normalizza_standard(XTrain);
            %yTrainNorm = Util.normalizza_standard(obj.yTrain);
            %XTestNorm = Util.normalizza_standard(XTest);
            %yTestNorm = Util.normalizza_standard(obj.yTest);

        end

        % Aggiungo una colonna di 1 nella matrice di train e di test per
        % costruire la matrice di Vandermonde
        function [XTrain, XTest] = getMatriceVandermonde(obj, XTrain, XTest, dim1, dim2)
            XTrain = [ones(dim1, 1), XTrain];
            XTest = [ones(dim2, 1), XTest];
        end




        % dim1 è la lunghezza della colonna della target nel training
        % dim2 è quella del test
        function [XTrain, yTrain, XTest, yTest] = getFeature(obj, dim1, dim2)
            yTrain = obj.yTrain;
            yTest = obj.yTest;
            % a seconda della dimensione specificata mi ricavo le feature
            % di input necessarie per la costruzione del modello
            switch obj.d
                case 1
                    XTrain = (1:dim1)';
                    XTest = (dim1 + 1 : dim1 + dim2)';
                    % normalizzo i dati se questo viene specificato
                    % nell'inizializzazione del modello
                    if(obj.isNormalizzato)
                        %disp("vengono normalizzati");
                        [XTrain, yTrain, XTest, yTest] = obj.normalizzaDati(XTrain, XTest);
                    end
                    % Costruisco la matrice di Vandermonde
                    [XTrain, XTest] = obj.getMatriceVandermonde(XTrain, XTest, dim1, dim2);
                case 2
                    XTrain1 = (1:dim1)';
                    XTrain2 = XTrain1.^2;
                    XTrain = [XTrain1, XTrain2];
                    XTest1 = (dim1+1:dim1+dim2)';
                    XTest2 = XTest1.^2;
                    XTest = [XTest1, XTest2];
                    if(obj.isNormalizzato)
                        [XTrain, yTrain, XTest, yTest] = obj.normalizzaDati(XTrain, XTest);
                    end
                    [XTrain, XTest] = obj.getMatriceVandermonde(XTrain, XTest, dim1, dim2);
                case 3
                    XTrain1 = (1:dim1)';
                    XTrain2 = XTrain1.^2;
                    XTrain3 = XTrain1.^3;
                    XTrain = [XTrain1, XTrain2, XTrain3];
                    XTest1 = (dim1+1:dim1+dim2)';
                    XTest2 = XTest1.^2;
                    XTest3 = XTest1.^3;
                    XTest = [XTest1, XTest2, XTest3];
                    if(obj.isNormalizzato)
                        [XTrain, yTrain, XTest, yTest] = obj.normalizzaDati(XTrain, XTest);
                    end
                    [XTrain, XTest] = obj.getMatriceVandermonde(XTrain, XTest, dim1, dim2);
                case 4
                    XTrain1 = (1:dim1)';
                    XTrain2 = XTrain1.^2;
                    XTrain3 = XTrain1.^3;
                    XTrain4 = XTrain1.^4;
                    XTrain = [XTrain1, XTrain2, XTrain3, XTrain4];
                    XTest1 = (dim1+1:dim1+dim2)';
                    XTest2 = XTest1.^2;
                    XTest3 = XTest1.^3;
                    XTest4 = XTest1.^4;
                    XTest = [XTest1, XTest2, XTest3, XTest4];
                    if(obj.isNormalizzato)
                        [XTrain, yTrain, XTest, yTest] = obj.normalizzaDati(XTrain, XTest);
                    end
                    [XTrain, XTest] = obj.getMatriceVandermonde(XTrain, XTest, dim1, dim2);
                otherwise
                    disp('d non specificata');
            end
        end

        function [N, D, x, v, pb, up, low, tol, c1, c2, maxit, f] = getParametriPSO(obj)
            % Parametri per PSO
            N = 30; % Numero di particelle
            D = size(obj.XTrain, 2); % Dimensione dello spazio dei parametri (numero di colonne di X)
            x = rand(D, N); % Posizioni iniziali delle particelle
            v = rand(D, N); % Velocità iniziali delle particelle
            pb = x; % Posizioni personali migliori iniziali (uguali alle posizioni iniziali)
            up = 10; % Limite superiore dei parametri
            low = -10; % Limite inferiore dei parametri
            tol = 1e-5; % Tolleranza per la convergenza
            c1 = .5; % Parametro di apprendimento c1
            c2 = .9; % Parametro di apprendimento c2
            maxit = 100; % Numero massimo di iterazioni

            % Funzione obiettivo
            f = @(theta) Swarm.funzione_obiettivo(obj.XTrain, obj.yTrain, theta);
        end

        % Ottengo i coefficienti per la regressione a seconda del metodo
        % selezionato, oltre ai coefficienti ottengo anche il
        % condizionamento e il tempo di esecuzione del metodo
        function [coeff, condizionamento, tempo] = getCoefficienti(obj)
            pcr = PCR(obj.XTrain, obj.yTrain);
            [N, D, posParticelle, v, pb, up, low, tol, c1, c2, maxit, f] = obj.getParametriPSO();
            switch obj.metodo
                case 'normali'
                    tic;
                    [coeff, condizionamento] = lss_normali(obj.XTrain, obj.yTrain);
                    tempo = toc;
                case 'thin_qr'
                    tic;
                    [coeff, condizionamento] = lss_thin_qr(obj.XTrain, obj.yTrain);
                    tempo = toc;
                case 'thin_qr_pivoting'
                    tic;
                    [coeff, condizionamento] = lss_qr_pivoting(obj.XTrain, obj.yTrain);
                    tempo = toc;
                case 'svd_scree_plot_cattel'
                    [coeff,tempo, condizionamento] = pcr.scree_Plot_Cattel();
                case 'svd_guttman_keiser'
                    tic;
                    [coeff, k, condizionamento] = pcr.guttman_keiser(1);
                    tempo = toc;
                    %fprintf('K selezionato da Guttman-Keiser: %d\n', k);
                case 'svd_energia'
                    tic;
                    [coeff, condizionamento] = pcr.criterio_Energia(90);
                    tempo = toc;
                case 'svd_entropia'
                    tic;
                    [coeff, condizionamento] = pcr.criterio_Entropia(0.95);
                    tempo = toc;
                case 'swarm'
                    tic;
                    % global best come coefficienti
                    [posParticelle, v, pb, coeff, nit, radius, iters] = Swarm.min_swarmND(N, posParticelle, pb, v, f, up, low, tol, c1, c2, maxit);
                    condizionamento = cond(posParticelle);
                    tempo = toc;
                % swarm con velocità pesate secondo aggiustamento randomico
                case 'swarm_rand'
                    tic;
                    [posParticelle, v, pb, coeff, nit, radius, iters] = Swarm.min_swarmND_Aggiustamento_Randomico(N, posParticelle, pb, v, f, up, low, tol, c1, c2, maxit);
                    %fprintf("Coefficienti swarm random");
                    %disp(coeff);
                    condizionamento = cond(posParticelle);
                    tempo = toc;
                case 'swarm_d_lineare'
                    tic;
                    [posParticelle, v, pb, coeff, nit, radius, iters] = Swarm.min_swarmND_Decremento_Lineare(N, posParticelle, pb, v, f, up, low, tol, c1, c2, maxit);
                    %fprintf("Coefficienti swarm con decremento lineare");
                    %disp(coeff);
                    condizionamento = cond(posParticelle);
                    tempo = toc;
                case 'swarm_d_non_lineare'
                    tic;
                    [posParticelle, v, pb, coeff, nit, radius, iters] = Swarm.min_swarmND_Decremento_Non_Lineare(N, posParticelle, pb, v, f, up, low, tol, c1, c2, maxit);
                    %fprintf("Coefficienti swarm con decremento non lineare");
                    %disp(coeff);
                    condizionamento = cond(posParticelle);
                    tempo = toc;
            end
        end

        function [predizione] = getPredizione(obj, coeff)
            predizione = obj.XTest * coeff;
        end

    end   
    
    methods
        % Inizializzazione e costruzione del modello
        function obj = Modello(trainData, testData, metodo, target, isNormalizzato, d)
            obj.metodo = metodo;
            obj.trainData = trainData;
            obj.testData = testData;
            %Setto la variabile target a seconda di quella specificata
            switch target
                case 'meantemp'
                    [obj.yTrain, obj.yTest] = obj.setVariabileTarget(2);
                case 'humidity'
                    [obj.yTrain, obj.yTest] = obj.setVariabileTarget(3);
                case 'wind_speed'
                    [obj.yTrain, obj.yTest] = obj.setVariabileTarget(4);
                case 'mean_pressure'
                    [obj.yTrain, obj.yTest] = obj.setVariabileTarget(5);
                otherwise
                    disp('Target non valida');
            end
            obj.isNormalizzato = isNormalizzato;

            % d indica il grado del polinomio per il modello
            obj.d = d;

            % Entrambe le dimensioni servono per poter definire le feature
            % di input
            dim1 = size(obj.yTrain, 1);
            dim2 = size(obj.yTest, 1);
            
            % Inizializzo tutte le feature necessarie per la costruzione
            % del modello
            [obj.XTrain, obj.yTrain, obj.XTest, obj.yTest] = obj.getFeature(dim1, dim2);

            % Ricavo i coefficienti, a seconda del metodo specificato, per
            % poter ottenere le predizioni
            [coeff, obj.condizionamento, obj.tempo] = obj.getCoefficienti();
            
            %disp(coeff);

            % Calcolo la predizione
            obj.yPred = obj.getPredizione(coeff);

            % Calcolo l'RMSE 
            obj.rmse = Util.rmse(obj.yTest, obj.yPred);

        end
    end
end

