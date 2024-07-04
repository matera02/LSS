classdef Util
    
    
    methods (Static)

        function errore = rmse(valore_esatto, predizione)
            errore = sqrt(mean((valore_esatto - predizione).^2));
        end

        function val = normalizza_minMax(X, min, Max)
            val = (X - min) ./ (Max - min);
        end

        function val = normalizza_standard(X)
            media = mean(X);
            deviazione_standard = std(X);
            val = (X - media) ./ deviazione_standard;
        end

        
        function stampa_modelli(trainData, testData, target, isNormalizzato, d)
            x = Modello(trainData, testData, 'normali', target, isNormalizzato, d);
            y = Modello(trainData, testData, 'thin_qr', target, isNormalizzato, d);
            z = Modello(trainData, testData, 'thin_qr_pivoting', target, isNormalizzato, d);
            sc = Modello(trainData, testData, 'svd_scree_plot_cattel', target, isNormalizzato, d);
            sgk = Modello(trainData, testData, 'svd_guttman_keiser', target, isNormalizzato, d);
            senrg = Modello(trainData, testData, 'svd_energia', target, isNormalizzato, d);
            sentr = Modello(trainData, testData, 'svd_entropia', target, isNormalizzato, d);
            swarm = Modello(trainData, testData, 'swarm', target, isNormalizzato, d);
            swarm_rand = Modello(trainData, testData, 'swarm_rand', target, isNormalizzato, d);
            swarm_d_lineare = Modello(trainData, testData, 'swarm_d_lineare', target, isNormalizzato, d);
            swarm_d_non_lineare = Modello(trainData, testData, 'swarm_d_non_lineare', target, isNormalizzato, d);


            fprintf("Rango della matrice di training: %d", rank(x.XTrain));
            %disp(x.XTrain);
            fprintf('\n\nRMSE Normali: %.4f\t Tempo trascorso: %.4f\tCondizionamento: %.4f\n', x.rmse, x.tempo, x.condizionamento);
            fprintf('RMSE THIN QR: %.4f\t Tempo trascorso: %.4f\tCondizionamento: %.4f\n', y.rmse, y.tempo, y.condizionamento);
            fprintf('RMSE THIN QR PIVOTING: %.4f\t Tempo trascorso: %.4f\tCondizionamento: %.4f\n', z.rmse, z.tempo, z.condizionamento);
            fprintf('RMSE SVD SCREE-PLOT CATTEL: %.4f\t Tempo trascorso: %.4f\tCondizionamento: %.4f\n', sc.rmse, sc.tempo, sc.condizionamento);
            fprintf('RMSE SVD GUTTMAN KEISER: %.4f\t Tempo trascorso: %.4f\tCondizionamento: %.4f\n', sgk.rmse, sgk.tempo, sgk.condizionamento);
            fprintf('RMSE SVD ENERGIA: %.4f\t Tempo trascorso: %.4f\tCondizionamento: %.4f\n', senrg.rmse, senrg.tempo, senrg.condizionamento);
            fprintf('RMSE SVD ENTROPIA: %.4f\t Tempo trascorso: %.4f\tCondizionamento: %.4f\n\n\n', sentr.rmse, sentr.tempo, sentr.condizionamento);
            fprintf('RMSE SWARM: %.4f\t Tempo trascorso: %.4f\n', swarm.rmse, swarm.tempo);
            fprintf('RMSE SWARM RANDOM: %.4f\t Tempo trascorso: %.4f\n', swarm_rand.rmse, swarm_rand.tempo);
            fprintf('RMSE SWARM DECREMENTO LINEARE: %.4f\t Tempo trascorso: %.4f\n', swarm_d_lineare.rmse, swarm_d_lineare.tempo);
            fprintf('RMSE SWARM DECREMENTO NON LINEARE: %.4f\t Tempo trascorso: %.4f\n', swarm_d_non_lineare.rmse, swarm_d_non_lineare.tempo);
        end


        function tempo_medio = calcola_tempo_medio_esecuzione(trainData, testData, metodo, target, isNormalizzato, d, maxit)
            tempo_totale = 0;
            for i = 1:maxit
                x = Modello(trainData, testData, metodo, target, isNormalizzato, d);
                tempo_totale = tempo_totale + x.tempo;
            end
            tempo_medio = tempo_totale / maxit;
        end


        % NON CONSIDERO LO SCREE_PLOT DIPENDE DAL TRONCAMENTO
        function stampa_tempo_medio_esecuzione(trainData, testData, target, isNormalizzato, d, maxit)
            fprintf("TEMPO MEDIO DI ESECUZIONE: \n");
            tempo_medio_normali = Util.calcola_tempo_medio_esecuzione(trainData, testData, 'normali', target, isNormalizzato, d, maxit);
            tempo_medio_thin_qr = Util.calcola_tempo_medio_esecuzione(trainData, testData, 'thin_qr', target, isNormalizzato, d, maxit);
            tempo_medio_qr_pivoting = Util.calcola_tempo_medio_esecuzione(trainData, testData, 'thin_qr_pivoting', target, isNormalizzato, d, maxit);
            tempo_medio_guttman_keiser = Util.calcola_tempo_medio_esecuzione(trainData, testData, 'svd_guttman_keiser', target, isNormalizzato, d, maxit);
            tempo_medio_energia = Util.calcola_tempo_medio_esecuzione(trainData, testData, 'svd_energia', target, isNormalizzato, d, maxit);
            tempo_medio_entropia = Util.calcola_tempo_medio_esecuzione(trainData, testData, 'svd_entropia', target, isNormalizzato, d, maxit);
            tempo_medio_swarm = Util.calcola_tempo_medio_esecuzione(trainData, testData, 'swarm', target, isNormalizzato, d, maxit);
            tempo_medio_swarm_rand = Util.calcola_tempo_medio_esecuzione(trainData, testData, 'swarm_rand', target, isNormalizzato, d, maxit);
            tempo_medio_swarm_d_lineare = Util.calcola_tempo_medio_esecuzione(trainData, testData, 'swarm_d_lineare', target, isNormalizzato, d, maxit);
            tempo_medio_swarm_d_non_lineare = Util.calcola_tempo_medio_esecuzione(trainData, testData, 'swarm_d_non_lineare', target, isNormalizzato, d, maxit);


            fprintf("NORMALI: %f\n", tempo_medio_normali);
            fprintf("THIN QR: %f\n", tempo_medio_thin_qr);
            fprintf("THIN QR PIVOTING: %f\n", tempo_medio_qr_pivoting);
            fprintf("GUTTMAN KEISER: %f\n", tempo_medio_guttman_keiser);
            fprintf("ENERGIA: %f\n", tempo_medio_energia);
            fprintf("ENTROPIA: %f\n", tempo_medio_entropia);
            fprintf("SWARM: %f\n", tempo_medio_swarm);
            fprintf("SWARM RANDOM: %f\n", tempo_medio_swarm_rand);
            fprintf("SWARM DECREMENTO LINEARE: %f\n", tempo_medio_swarm_d_lineare);
            fprintf("SWARM DECREMENTO NON LINEARE: %f\n", tempo_medio_swarm_d_non_lineare);


        end
    end
end

