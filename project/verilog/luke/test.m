W = zeros(64,64);
X = zeros(64,64);
seed = 1234567;
for i=1:64
    for j=1:64
        W(i,j) = mod(seed,2);
        seed = mod(seed * 17, 100007);
        X(i,j) = mod(seed,2);
        seed = mod(seed * 17, 100007);
    end
end
%W(1:16,1:16)
%X(1:16,1:16)
ANS = W(1:16,1:16)^2 * X(1:16,1:16);