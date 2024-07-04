function [soluzioni, cond_pivoting_qr] = lss_qr_pivoting(A, b)
    [Q,R,p] = qr(A,'econ', "vector");
    cond_pivoting_qr = cond(R);
    %fprintf("Vettore di permutazione qr pivoting: ");
    %disp(p);
    soluzioni(p,:) = R\(Q\b);
end

