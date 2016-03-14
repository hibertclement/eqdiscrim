function Xout = l2filter(b, a, X)
    [nr, nc] = size(X);
    
    X_01 = filter(b, a, X);
    if nc == 1
        X_02 = filter(b, a, flipud(X_01));
        Xout = flipud(X_02);
    else
        X_02 = filter(b, a, fliplr(X_01));
        Xout = fliplr(X_02);   
end