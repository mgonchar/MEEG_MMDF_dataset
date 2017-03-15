function d = corr_dist(A,B)

%% Herdin
herdin = 1 - sum(sum(A.*B))/(norm(A,'fro')*norm(B,'fro'));
fprintf('Herdin: %f\n', herdin);

%% Frostner
e = eigs(A,B);
e = e(e>0.01);
lne = log(e); clear e;
frostner = sqrt(lne'*lne); clear lne;
fprintf('Frostner: %f\n', frostner);

%% KL Norm
kl_norm = (sum(sum(inv(A).*B)) - log(det(B)/det(A)))/2;
fprintf('K-L (Normal): %f\n', kl_norm);

d = [herdin frostner kl_norm];

end