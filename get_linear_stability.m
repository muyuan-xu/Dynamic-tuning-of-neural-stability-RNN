function [eigs, neigs_uns, Vs, Us] = get_linear_stability(net, n_x0_1, do_return_eigvecs)
% A function to automate the linear stability analysis around the point n_x0_1.
% 
% INPUTS
% net - the matlab structure of the network being analyzed.
%
% n_x0_1 - the point around which the linear stability analysis is done.
%
% do_return_eigvecs - boolean, as to whether or not to return the left and
% right eigenvectors.
%
% OUTPUTS
% eigs - eigenvalues of Jacobian.
%
% neigs_uns - number of unstable eigenvalues for either continuous or
% discrete linearized system.
%
% Vs - right eigenvectors, returned as empty if do_return_eigvecs is false.
%
% Us - left eigenvectors, returned as empty if do_return_eigvecs is false.

N = net.layers(2).npost;
[~, n_Wrr_n, ~, ~, ~, ~] = unpackRNN(net, net.theta);

dt_o_tau = net.dt / net.tau;

n_r0_1 = net.layers(2).transfun(n_x0_1);
n_dr0_1 = net.layers(2).Doperator(n_r0_1);

if dt_o_tau < 1.0
    % Jacobian.
    n_J_F_n = -eye(N) + n_Wrr_n * diag(n_dr0_1);
else
    n_J_F_n = n_Wrr_n * diag(n_dr0_1);
end

if do_return_eigvecs
    [V, D] = eig(n_J_F_n);
    eigs = diag(D);
    if dt_o_tau < 1.0
        [~, idxs] = sort(real(eigs), 'descend');
        eigs = eigs(idxs);
        Vs = V(:,idxs);
        neigs_uns = sum(real(eigs) > 0);
    else
        [~, idxs] = sort(abs(eigs), 'descend');
        eigs = eigs(idxs);
        Vs = V(:,idxs);
        neigs_uns = sum(abs(eigs) > 1);
    end
    Us = pinv(Vs); % The left eigenvectors.
    
    % Consistently orient all the vectors.
    ref_vec = ones(N,1);
    for i = 1:N
        if dot(Vs(:,i), ref_vec) < 0.0
            Vs(:,i) = -Vs(:,i);
            Us(i,:) = -Us(i,:);
        end
    end
else
    eigs = eig(n_J_F_n);
    Vs = [];
    Us = [];
    if dt_o_tau < 1.0
        [~, idxs] = sort(real(eigs), 'descend');
        eigs = eigs(idxs);
        neigs_uns = sum(real(eigs) > 0);
    else
        [~, idxs] = sort(abs(eigs), 'descend');
        eigs = eigs(idxs);
        neigs_uns = sum(abs(eigs) > 1);
    end
end

end