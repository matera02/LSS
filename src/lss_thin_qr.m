function [soluzioni, cond_thin_qr] = lss_thin_qr(A, b)
    %[Q, T] = qr(A);
    %num_colonne = size(A, 2);
    %R = T(1:num_colonne, :);
    %z = Q' * b; % z è il vettore che scomponiamo in c1 e c2
    %c = z(1:num_colonne); 
    %soluzioni = R \ c;

    [Q, R] = qr(A, 'econ');
    %disp(Q);
    %disp(R);
    
    cond_thin_qr = cond(R);
    soluzioni = R \ (Q' * b);
end

