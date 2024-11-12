function [xs, idxs, phis_to_go, pAp] = conjgrad(eval_Ap, b, x0, maxiters, miniters, cg_tol)
%% Implementation of the conjugate gradient method. See Martens, 2010.
% phi(x) = 1/2 x' * A * x - b' * x is the quadratic function to be minimized.
gapratio = 0.1;
mingap = 10;
maxgap = max(ceil(maxiters * gapratio), mingap) + 1;
phis = zeros(maxgap,1); % The values of phi are kept for terminate condition.

% These are used for CG backtracking.
idx_last = 1;
idx = 5;
gamma = 1.3;
xs = {};
idxs = [];
phis_to_go = [];
pAp = 0.0;

% Initial values for iteration.
x = x0;
package = eval_Ap(x0);
Ap = package{1};
r = b - Ap;
p = r;
phi = 0.5 * double((-r-b)'*x);

for i = 1:maxiters
    % Compute the matrix-vector product. This is where 95% of the work in hf lies.
    package = eval_Ap(p);
    Ap = package{1};
    pAp = p'*Ap;
    
    % The Gauss-Newton matrix should always be positive definite.
    if pAp <= 0
        disp(['Non-positive curvature found!: pAp = ', num2str(pAp,16)]);
        disp('Bailing...');
        break;
    end
    
    alpha = (r'*r)/pAp;  
    x = x + alpha*p;
    r_new = r - alpha*Ap;
    beta = (r_new'*r_new)/(r'*r);
    p = r_new + beta*p;
    r = r_new;
    
    phi = 0.5 * double((-r-b)'*x);
    phis(mod(i-1,maxgap)+1) = phi;
    
    testgap = max(ceil(i*gapratio), mingap); % This is the k in Martens, 2010.
    phi_pre = phis(mod(i-testgap-1,maxgap)+1);
    
    if i == 1 || i == 3 || i == 5 || i == idx
        xs{end+1} = x;
        idxs(end+1) = i;
        phis_to_go(end+1) = phi;
        
        idx_last = idx;
        idx = ceil(idx*gamma);
    end
    
    % Terminate condition. This may become largely irrelevant after the
    % optimization has passed certain steps. The CG iteration always hits
    % maxiters first.
    if i > testgap && phi < 0 && (phi - phi_pre)/phi < testgap*cg_tol && i >= miniters
        disp(['CG breaking due to relative tolerance condition based on phi at iter: ', num2str(i) '.']);
        break;
    end
end

%% Return.
if i ~= idx_last
    xs{end+1} = x;
    idxs(end+1) = i;
    phis_to_go(end+1) = phi;
end

end

