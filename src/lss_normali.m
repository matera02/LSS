function [soluzioni, cond_normali] = lss_normali(A, b)
    cond_normali = cond(A' * A);
    soluzioni = (A' * A) \ (A' * b);
end

