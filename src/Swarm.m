classdef Swarm

    methods (Static)
        % Funzione obiettivo che calcola il MSE
        % Per le funzioni convesse è garantito trovare il minimo
        function errore = funzione_obiettivo(X, y, theta)
            predizioni = X * theta; % Previsioni del modello
            errori = y - predizioni; % Errori di previsione
            errore = mean(errori .^ 2); % MSE
        end
        
        function [x, v, pb, gb, nit, radius, iters] = min_swarmND(N, x, pb, v, f, up, low, tol, c1, c2, maxit)
            % INPUT:
            % - N: dimensione dello swarm
            % - x: matrice delle posizioni iniziali per ciascuna particella (DxN)
            % - pb: matrice delle posizioni personali migliori (DxN)
            % - v: matrice delle velocità per ciascuna particella (DxN)
            % - f: funzione obiettivo
            % - up: confine superiore (1xD)
            % - low: confine inferiore (1xD)
            % - tol: threshold di accuratezza
            % - c1, c2: parametri di apprendimento
            % - maxit: numero massimo di iterazioni

            iters = {}; % Usa una cella per memorizzare le iterazioni
            nit = 0;

            % Deduzione della dimensione dello spazio
            D = size(x, 1);

            % Inizializzazione
            f_pb = f(pb);
            [~, idx] = min(f_pb);
            gb = pb(:, idx);
            x_prec = x;
            v_prec = v;
            pb_prec = pb;
            iters{end+1} = x_prec;

            % Calcolo di v in t+1 con strategia "global best"
            r1 = rand(D, N);
            r2 = rand(D, N);
            % v = momento + comp_cognitiva + comp_sociale
            v = v_prec + c1 * r1 .* (pb_prec - x_prec) + c2 * r2 .* (gb - x_prec);
            x = x_prec + v;

            % Controllo sulle particelle che escono dallo spazio di ricerca
            if min(x) < low
                x(x < low) = low;
            end

            if max(x) > up
                x(x>up) = up;
            end

            iters{end+1} = x;

            % Aggiornamento di x e pb
            f_x = f(x);
            ind = f_x < f_pb;
            pb(:, ind) = x(:, ind);
            f_pb(ind) = f_x(ind);
            [~, idx] = min(f_pb);
            gb = pb(:, idx);

            % Calcolo del raggio
            %disp(up);
            %disp(low);
            diameter = norm(up - low);
            radius = max(sqrt(sum((x - gb).^2, 1))) / diameter;

            while (radius > tol) && (nit < maxit)
                x_prec = x;
                v_prec = v;
                pb_prec = pb;

                r1 = rand(D, N);
                r2 = rand(D, N);
                v = v_prec + c1 * r1 .* (pb_prec - x_prec) + c2 * r2 .* (gb - x_prec);
                x = x_prec + v;

                % Controllo sulle particelle che escono dallo spazio di ricerca
                if min(x) < low
                    x(x < low) = low;
                end

                if max(x) > up
                    x(x>up) = up;
                end

                iters{end+1} = x;

                f_x = f(x);
                f_pb = f(pb);
                ind = f_x < f_pb;
                pb(:, ind) = x(:, ind);
                f_pb(ind) = f_x(ind);
                [~, idx] = min(f_pb);
                gb = pb(:, idx);

                % Calcolo del raggio
                radius = max(sqrt(sum((x - gb).^2, 1))) / diameter;
                nit = nit + 1;
            end
        end


        % il peso viene scelto in maniera random 
        % con una normale standard con mu = 0.72 e
        % sigma abbastanza piccolo tale che w < 1
        function [x, v, pb, gb, nit, radius, iters] = min_swarmND_Aggiustamento_Randomico(N, x, pb, v, f, up, low, tol, c1, c2, maxit)
            % INPUT:
            % - N: dimensione dello swarm
            % - x: matrice delle posizioni iniziali per ciascuna particella (DxN)
            % - pb: matrice delle posizioni personali migliori (DxN)
            % - v: matrice delle velocità per ciascuna particella (DxN)
            % - f: funzione obiettivo
            % - up: confine superiore (1xD)
            % - low: confine inferiore (1xD)
            % - tol: threshold di accuratezza
            % - c1, c2: parametri di apprendimento
            % - maxit: numero massimo di iterazioni

            iters = {}; % Usa una cella per memorizzare le iterazioni
            nit = 0;

            % Deduzione della dimensione dello spazio
            D = size(x, 1);

            % Inizializzazione
            f_pb = f(pb);
            [~, idx] = min(f_pb);
            gb = pb(:, idx);
            x_prec = x;
            v_prec = v;
            pb_prec = pb;
            iters{end+1} = x_prec;

            % Calcolo di v in t+1 con strategia "global best"
            r1 = rand(D, N);
            r2 = rand(D, N);

            mu = 0.72;
            sigma = 0.1;
            w = normrnd(mu, sigma);
            
            v = w * v_prec + c1 * r1 .* (pb_prec - x_prec) + c2 * r2 .* (gb - x_prec);
            x = x_prec + v;

            % Controllo sulle particelle che escono dallo spazio di ricerca
            if min(x) < low
                x(x < low) = low;
            end

            if max(x) > up
                x(x>up) = up;
            end

            iters{end+1} = x;

            % Aggiornamento di x e pb
            f_x = f(x);
            ind = f_x < f_pb;
            pb(:, ind) = x(:, ind);
            f_pb(ind) = f_x(ind);
            [~, idx] = min(f_pb);
            gb = pb(:, idx);

            % Calcolo del raggio
            %disp(up);
            %disp(low);
            diameter = norm(up - low);
            radius = max(sqrt(sum((x - gb).^2, 1))) / diameter;

            while (radius > tol) && (nit < maxit)
                x_prec = x;
                v_prec = v;
                pb_prec = pb;

                r1 = rand(D, N);
                r2 = rand(D, N);

                w = normrnd(mu, sigma);

                v = w * v_prec + c1 * r1 .* (pb_prec - x_prec) + c2 * r2 .* (gb - x_prec);
                x = x_prec + v;

                % Controllo sulle particelle che escono dallo spazio di ricerca
                if min(x) < low
                    x(x < low) = low;
                end

                if max(x) > up
                    x(x>up) = up;
                end

                iters{end+1} = x;

                f_x = f(x);
                f_pb = f(pb);
                ind = f_x < f_pb;
                pb(:, ind) = x(:, ind);
                f_pb(ind) = f_x(ind);
                [~, idx] = min(f_pb);
                gb = pb(:, idx);

                % Calcolo del raggio
                radius = max(sqrt(sum((x - gb).^2, 1))) / diameter;
                nit = nit + 1;
            end
        end


        % decremento lineare: ad ogni iterazione w è scelto
        % w(t) = (w(0) - w(nt)) * (nt - n)/nt + w(nt)
        % w(0) > w(nt)
        % Scelta tipica w(0) = 0.9, w(nt) = 0.4
        function [x, v, pb, gb, nit, radius, iters] = min_swarmND_Decremento_Lineare(N, x, pb, v, f, up, low, tol, c1, c2, maxit)
            % INPUT:
            % - N: dimensione dello swarm
            % - x: matrice delle posizioni iniziali per ciascuna particella (DxN)
            % - pb: matrice delle posizioni personali migliori (DxN)
            % - v: matrice delle velocità per ciascuna particella (DxN)
            % - f: funzione obiettivo
            % - up: confine superiore (1xD)
            % - low: confine inferiore (1xD)
            % - tol: threshold di accuratezza
            % - c1, c2: parametri di apprendimento
            % - maxit: numero massimo di iterazioni

            iters = {}; % Usa una cella per memorizzare le iterazioni
            nit = 0;

            % Deduzione della dimensione dello spazio
            D = size(x, 1);

            % Inizializzazione
            f_pb = f(pb);
            [~, idx] = min(f_pb);
            gb = pb(:, idx);
            x_prec = x;
            v_prec = v;
            pb_prec = pb;
            iters{end+1} = x_prec;

            % Calcolo di v in t+1 con strategia "global best"
            r1 = rand(D, N);
            r2 = rand(D, N);

            w_0 = 0.9;
            w_nt = 0.4;

            % Decremento lineare
            w = (w_0-w_nt)*(maxit-nit)/(maxit) + w_nt;

            v = w * v_prec + c1 * r1 .* (pb_prec - x_prec) + c2 * r2 .* (gb - x_prec);
            x = x_prec + v;

            % Controllo sulle particelle che escono dallo spazio di ricerca
            if min(x) < low
                x(x < low) = low;
            end

            if max(x) > up
                x(x>up) = up;
            end

            iters{end+1} = x;

            % Aggiornamento di x e pb
            f_x = f(x);
            ind = f_x < f_pb;
            pb(:, ind) = x(:, ind);
            f_pb(ind) = f_x(ind);
            [~, idx] = min(f_pb);
            gb = pb(:, idx);

            % Calcolo del raggio
            %disp(up);
            %disp(low);
            diameter = norm(up - low);
            radius = max(sqrt(sum((x - gb).^2, 1))) / diameter;

            while (radius > tol) && (nit < maxit)
                x_prec = x;
                v_prec = v;
                pb_prec = pb;

                r1 = rand(D, N);
                r2 = rand(D, N);
                
                % Decremento lineare
                w = (w_0-w_nt)*(maxit-nit)/(maxit) + w_nt;

                v = w * v_prec + c1 * r1 .* (pb_prec - x_prec) + c2 * r2 .* (gb - x_prec);
                x = x_prec + v;

                % Controllo sulle particelle che escono dallo spazio di ricerca
                if min(x) < low
                    x(x < low) = low;
                end

                if max(x) > up
                    x(x>up) = up;
                end

                iters{end+1} = x;

                f_x = f(x);
                f_pb = f(pb);
                ind = f_x < f_pb;
                pb(:, ind) = x(:, ind);
                f_pb(ind) = f_x(ind);
                [~, idx] = min(f_pb);
                gb = pb(:, idx);

                % Calcolo del raggio
                radius = max(sqrt(sum((x - gb).^2, 1))) / diameter;
                nit = nit + 1;
            end
        end

        % Ad ogni iterazione w è scelto come
        % w(t+1) = ((w(t) - 0.4) * (nt - t)) / (nt + 0.4)
        function [x, v, pb, gb, nit, radius, iters] = min_swarmND_Decremento_Non_Lineare(N, x, pb, v, f, up, low, tol, c1, c2, maxit)
            % INPUT:
            % - N: dimensione dello swarm
            % - x: matrice delle posizioni iniziali per ciascuna particella (DxN)
            % - pb: matrice delle posizioni personali migliori (DxN)
            % - v: matrice delle velocità per ciascuna particella (DxN)
            % - f: funzione obiettivo
            % - up: confine superiore (1xD)
            % - low: confine inferiore (1xD)
            % - tol: threshold di accuratezza
            % - c1, c2: parametri di apprendimento
            % - maxit: numero massimo di iterazioni

            iters = {}; % Usa una cella per memorizzare le iterazioni
            nit = 0;

            % Deduzione della dimensione dello spazio
            D = size(x, 1);

            % Inizializzazione
            f_pb = f(pb);
            [~, idx] = min(f_pb);
            gb = pb(:, idx);
            x_prec = x;
            v_prec = v;
            pb_prec = pb;
            iters{end+1} = x_prec;

            % Calcolo di v in t+1 con strategia "global best"
            r1 = rand(D, N);
            r2 = rand(D, N);

            

            % Inizializzo w a 0.9
            w = 0.9;

            v = w * v_prec + c1 * r1 .* (pb_prec - x_prec) + c2 * r2 .* (gb - x_prec);
            x = x_prec + v;

            % Controllo sulle particelle che escono dallo spazio di ricerca
            if min(x) < low
                x(x < low) = low;
            end

            if max(x) > up
                x(x>up) = up;
            end

            iters{end+1} = x;

            % Aggiornamento di x e pb
            f_x = f(x);
            ind = f_x < f_pb;
            pb(:, ind) = x(:, ind);
            f_pb(ind) = f_x(ind);
            [~, idx] = min(f_pb);
            gb = pb(:, idx);

            % Calcolo del raggio
            %disp(up);
            %disp(low);
            diameter = norm(up - low);
            radius = max(sqrt(sum((x - gb).^2, 1))) / diameter;

            while (radius > tol) && (nit < maxit)
                x_prec = x;
                v_prec = v;
                pb_prec = pb;

                r1 = rand(D, N);
                r2 = rand(D, N);
                
                % Decremento non lineare
                w = (w-0.4)*(maxit-nit)/(maxit + 0.4);

                v = w * v_prec + c1 * r1 .* (pb_prec - x_prec) + c2 * r2 .* (gb - x_prec);
                x = x_prec + v;

                % Controllo sulle particelle che escono dallo spazio di ricerca
                if min(x) < low
                    x(x < low) = low;
                end

                if max(x) > up
                    x(x>up) = up;
                end

                iters{end+1} = x;

                f_x = f(x);
                f_pb = f(pb);
                ind = f_x < f_pb;
                pb(:, ind) = x(:, ind);
                f_pb(ind) = f_x(ind);
                [~, idx] = min(f_pb);
                gb = pb(:, idx);

                % Calcolo del raggio
                radius = max(sqrt(sum((x - gb).^2, 1))) / diameter;
                nit = nit + 1;
            end
        end



    end
end

