classdef PCR
    
    properties
        A
        U
        S
        V
        b
    end


    methods (Access = private)

        function [U_k, S_k, V_k] = seleziona_k_componenti(obj, k)
            U_k = obj.U(:, 1:k);
            S_k = obj.S(1:k, 1:k);
            V_k = obj.V(:, 1:k);
        end

        function soluzioni = risolvi(obj, U_k, S_k, V_k)
            s = diag(S_k);
            soluzioni = V_k * ((U_k' * obj.b) ./ s);
        end

        function condizionamento = calcola_condizionamento(obj, S_k)
            valori_singolari = diag(S_k);
            condizionamento = valori_singolari(1) / valori_singolari(end);
        end


    end


    methods (Access = public)
        function obj = PCR(A, b)
            obj.A = A;
            %PCA Construct an instance of this class
            %   Detailed explanation goes here
            [obj.U, obj.S, obj.V] = svd(A, 'econ');
            obj.b = b;
        end
        
        function [soluzioni, tempo, condizionamento] = scree_Plot_Cattel(obj)
            valori_singolari = diag(obj.S);

            % Genero lo Scree-Plot per determinare il numero ottimale di componenti principali
            figure;
            semilogy(valori_singolari, 'ro-', 'LineWidth', 2);
            title('Scree Plot');
            xlabel('Componenti Principali');
            ylabel('Valori Singolari');
            k = input('Inserisci il numero di componenti principali da mantenere: ');

            % Prendo il tempo dopo aver deciso le k componenti
            tic;
            % Seleziono i primi k componenti principali
            [U_k, S_k, V_k] = obj.seleziona_k_componenti(k);

            condizionamento = obj.calcola_condizionamento(S_k);

            % Risolvo il sistema ridotto
            soluzioni = obj.risolvi(U_k, S_k, V_k);
            tempo = toc;

        end


        function [soluzioni, k, condizionamento] = guttman_keiser(obj, soglia)
            valori_singolari = diag(obj.S);
            % k è il numero di valori singolari > soglia
            k = length(valori_singolari(valori_singolari > soglia));
            [U_k, S_k, V_k] = obj.seleziona_k_componenti(k);
            condizionamento = obj.calcola_condizionamento(S_k);
            soluzioni = obj.risolvi(U_k, S_k, V_k);
        end

        % energia è l'energia da trattenere
        function [soluzioni, condizionamento] = criterio_Energia(obj, energia)
            valori_singolari = diag(obj.S);
            energia_totale = sum(valori_singolari.^2);
            percentuale = energia_totale * energia / 100;

            % Determino il numero minimo di componenti principali
            % per conservare una certa percentuale di energia
            temp = valori_singolari(1)^2;
            i = 1;
            while temp < percentuale && i < length(valori_singolari)
                i = i + 1;
                temp = sum(valori_singolari(1:i).^2);
            end

            % Seleziono i primi k componenti principali
            %fprintf("K selezionato da criterio energia: %d\n", i);
            [U_k, S_k, V_k] = obj.seleziona_k_componenti(i);

            condizionamento = obj.calcola_condizionamento(S_k);

            % Risolvo il sistema ridotto
            soluzioni = obj.risolvi(U_k, S_k, V_k);

        end

        % percentuale si riferisce alla percentuale dell'entropia
        function [soluzioni, condizionamento] = criterio_Entropia(obj, percentuale)
            r = rank(obj.A);
            valori_singolari = diag(obj.S);

            %disp(obj.S);

            %fprintf("Valori singolari: ");
            %disp(valori_singolari);

            fj = valori_singolari.^2 / sum(valori_singolari.^2);

            %fprintf("Fj: ");
            %disp(fj);

            E = -1 / log(r) * sum(fj  .* log(fj));
            %fprintf('SVD Entropia %f\n', E);
            % Calcolo la somma cumulativa delle frazioni di energia
            F = cumsum(fj);
            % Trovo i valori dove la somma cumulativa supera l'entropia
            K = F > E;
            elem = find(K, 1, "first"); %cerco il primo elemento in k a true
            %fprintf('%d\n', elem);
            % Calcolo di k basato su una certa percentuale dell'entropia
            ks = int64(ceil(r * E * percentuale));  % Uso int64 per convertire in intero
            
            %fprintf('SVD Entropia = %f, k suggerito = %d\n', E, ks);


            % Seleziono i primi k componenti principali
            [U_k, S_k, V_k] = obj.seleziona_k_componenti(ks);

            condizionamento = obj.calcola_condizionamento(S_k);

            % Risolvo il sistema ridotto
            soluzioni = obj.risolvi(U_k, S_k, V_k);
        end

    end
end

