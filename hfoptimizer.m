function [theta_opt, objfun] = hfoptimizer(net, conditions_ntrials, inputs_ntrials, targets_ntrials, varargin)
%% This code is created based on David Sussillo's implementation (https://github.com/sussillo/hfopt-matlab).
% Initial settings.
do_parallel = false;

save_path = './'; % Save snapshots for hf iterations.
save_every = 10; % Save every ... hf iterations. 

size_of_minibatches = NaN; % The size of minibatches used in CG iterations.

lambda_init = 1.0; % The initial weight of damping.
min_lambda = realmin; % To prevent lambda from being ridiculously small.
cg_tol = 5e-4; % This is the value used in Martens, 2010 (ICML). 
init_decay_factor = 0.95; % The initial theta used in CG is the old one multiplies this factor. See Martens, 2010.
% Use the Levenburg-Marquardt style heuristic method to find a proper lambda.
rho_drop_thresh = 0.75; % The value at which lambda is reduced.
rho_boost_thresh = 0.25; % The value at which lambda is increased.
rho_drop_val = 2.0/3.0;
rho_boost_val = 1.0/rho_drop_val;
max_lambda_increases_for_negative_curvature = 100; % Help when CG gives a negative curvature.
min_cg_iter = 20; % Minimal CG iterations.
max_cg_iter = 300; % Maximal CG iterations. This is important, otherwise the optimization flatens out early.
cg_increase_factor = 1.3; % If a solution is found by N CG iterations this time, then try N multiplied by this factor CG iters next time.

% Terminate conditions.
objfun_tol = 0.0; % Usually this is not used. See Martens, 2010.
objfun_diff_tol = 1e-20; % Difference in the objective function values.
grad_norm_tol = 1e-20; % Magnitude of the gradient.

max_hf_iter = 5000; % This value shall never be reached.
max_hf_fail_count = 5000; % Maximum number hf is allowed to fail.
max_hf_consecutive_fail_count = 500; % Maximum number hf is allowed to fail consecutively. Should be related to how fast lambda changes.
max_lambda = 1e20;

optargin = size(varargin,2);
for i = 1:2:optargin
    switch varargin{i}
        case 'do_parallel'
            do_parallel = varargin{i+1};
        case 'save_path'
            save_path = varargin{i+1};
        case 'save_every'
            save_every = varargin{i+1};
        case 'size_of_minibatches'
            size_of_minibatches = varargin{i+1};
        case 'lambda_init'
            lambda_init = varargin{i+1};
        case 'min_cg_iter'
            min_cg_iter = varargin{i+1};
        case 'max_cg_iter'
            max_cg_iter = varargin{i+1};
        case 'cg_tol'
            cg_tol = varargin{i+1};  
        case 'max_hf_iter'
            max_hf_iter = varargin{i+1};
        case 'max_hf_fail_count'
            max_hf_fail_count = varargin{i+1};
        case 'objfun_tol'
            objfun_tol = varargin{i+1};
        case 'objfun_diff_tol'
            objfun_diff_tol = varargin{i+1};        
        otherwise
            assert(false, ['Variable argument ' varargin{i} ' not recognized.']);
    end
end

ntrials = size(inputs_ntrials,2); % The total number of trials.

if do_parallel
    disp('Evaluating in parallel');
else
    disp('Evaluating in serial');
end

if isnan(size_of_minibatches)
    size_of_minibatches = ceil(ntrials/5);
end
assert(size_of_minibatches <= ntrials, 'Invalid size of minibatches.');
assert(size_of_minibatches > 0, 'Invalid size of minibatches.');
disp(['...using minibatch size of ' num2str(size_of_minibatches) ' out of ' num2str(ntrials) ' samples.']);

go = 1;
lambda = lambda_init;
niters_cg = min_cg_iter;
disp(['Initial maximum CG iterations: ' num2str(niters_cg) '.']);
pn_cgstart = zeros(size(net.theta)); % The initial pn used in CG iterations.
ncgiters_constant_decreasing = 0;

hf_iter = 0;
total_hf_fail_count = 0;
total_hf_consecutive_fail_count = 0;
objfun_last = Inf;
grad_norm = Inf;
rho = NaN;

do_resample_data = true;
do_recompute_gradient = true;
do_recompute_rho = true;

hf_iter_time = 0.0;
total_time = 0.0;
stop_string = '';

% First, evaluate the objective function to make sure we are not going backward at the beginning.
forward_passes_ntrials = cell(1,ntrials);
objfuns = cell(1,ntrials);
if ~do_parallel
    for i = 1:ntrials
        package = eval_objfun(net,conditions_ntrials{i},inputs_ntrials{i},targets_ntrials{i},forward_passes_ntrials{i});
        objfuns{i} = package{1};
    end
else
    parfor i = 1:ntrials
        package = eval_objfun(net,conditions_ntrials{i},inputs_ntrials{i},targets_ntrials{i},forward_passes_ntrials{i});
        objfuns{i} = package{1};
    end
end
clear package;
Loss = 0;
for i = 1:ntrials
    Loss = Loss + objfuns{i};
end
Loss = Loss / ntrials;
objfun = Loss;
disp(['Initial objective function: ' num2str(objfun) '.']);
objfun_constant_decreasing = realmax;

%% Main.
while go
    tic;
    hf_iter = hf_iter + 1;
    
    if hf_iter > max_hf_iter
        hf_iter = hf_iter - 1;
        stop_string = ['Stopping because total number of HF iterations is greater than ' num2str(max_hf_iter) '.'];
        break;
    end
    
    disp(['HF iteration: ' num2str(hf_iter) '.']);
    
    if do_resample_data
        % Evaluate forward passes.
        if ~do_parallel
            for i = 1:ntrials
                package = eval_network(net,conditions_ntrials{i},inputs_ntrials{i},targets_ntrials{i});
                forward_passes_ntrials{i} = package{1};
            end
        else
            parfor i = 1:ntrials
                package = eval_network(net,conditions_ntrials{i},inputs_ntrials{i},targets_ntrials{i});
                forward_passes_ntrials{i} = package{1};
            end
        end
        clear package;
    
        % Evaluate objective functions.
        if ~do_parallel
            for i = 1:ntrials
                package = eval_objfun(net,conditions_ntrials{i},inputs_ntrials{i},targets_ntrials{i},forward_passes_ntrials{i});
                objfuns{i} = package{1};
            end
        else
            parfor i = 1:ntrials
                package = eval_objfun(net,conditions_ntrials{i},inputs_ntrials{i},targets_ntrials{i},forward_passes_ntrials{i});
                objfuns{i} = package{1};
            end
        end
        clear package;
        Loss = 0;
        for i = 1:ntrials
            Loss = Loss + objfuns{i};
        end
        Loss = Loss / ntrials;
        objfun_new_data = Loss;
        
        % Sample minibatches.
        if size_of_minibatches < ntrials
            rp = randperm(ntrials);
            random_trial_idxs = rp(1:size_of_minibatches);
        else
            random_trial_idxs = 1:ntrials;
        end
        conditions_nsamples = conditions_ntrials(:,random_trial_idxs);
        inputs_nsamples = inputs_ntrials(:,random_trial_idxs);
        if ~isempty(targets_ntrials)
            targets_nsamples = targets_ntrials(:,random_trial_idxs);
        else
            targets_nsamples = {};
        end
        forward_passes_nsamples = forward_passes_ntrials(random_trial_idxs);
    end
    
    if do_recompute_gradient
        % Note that the gradient b is evaluated on the entire training set.
        grad = zeros(size(net.theta));
        if ~do_parallel
            for i = 1:ntrials
                package = eval_gradient(net,conditions_ntrials{i},inputs_ntrials{i},targets_ntrials{i},forward_passes_ntrials{i});
                grad = grad + package{1};
            end
        else
            parfor i = 1:ntrials
                package = eval_gradient(net,conditions_ntrials{i},inputs_ntrials{i},targets_ntrials{i},forward_passes_ntrials{i});
                grad = grad + package{1};
            end
        end
        clear package;
        grad = grad / ntrials;
        grad_norm = norm(grad);
    end
    
    % Do CG.
    all_pn = {};
    cg_gamma_idxs = [];
    % Protect against accidental negative curvature.
    nlambda_increases_for_negative_curvature = 0;
    while isempty(all_pn) && nlambda_increases_for_negative_curvature < max_lambda_increases_for_negative_curvature
        [all_pn, cg_gamma_idxs, all_cg_phis, pAp] = conjgrad(@(dw) eval_avg_gv(net, conditions_nsamples, inputs_nsamples, targets_nsamples, ...
            dw, lambda, forward_passes_nsamples, 'do_parallel', do_parallel), -grad, pn_cgstart, niters_cg, min_cg_iter, cg_tol);      
        if pAp <= 0.0
            % This means we have gone out of the trustworthy region of the
            % quadratic approximation, so increase lambda and try again.
            nlambda_increases_for_negative_curvature = nlambda_increases_for_negative_curvature + 1;
            if lambda > realmin
                lambda = rho_boost_val * lambda;
            end
            do_resample_data = 0;
            all_pn = {};
            disp(['Found non-positive curvature, increasing lambda to ' num2str(lambda) '.']);
        end
    end
    
    if pAp < 0.0
        stop_string = 'Stopping because non-positive curvature was found.';
        go = 0;
        break;
    end
    
    % CG backtracking.
    pn = all_pn{end};
    cgtheta = net.theta + pn; % net holds the best solution from last hf iteration.
    cgnet = net;
    cgnet.theta = cgtheta;
    if ~do_parallel
        for i = 1:ntrials
            package = eval_objfun(cgnet,conditions_ntrials{i},inputs_ntrials{i},targets_ntrials{i},[]);
            objfuns{i} = package{1};
        end
    else
        parfor i = 1:ntrials
            package = eval_objfun(cgnet,conditions_ntrials{i},inputs_ntrials{i},targets_ntrials{i},[]);
            objfuns{i} = package{1};
        end
    end
    clear package;
    Loss = 0;
    for i = 1:ntrials
        Loss = Loss + objfuns{i};
    end
    Loss = Loss / ntrials;
    objfun_cg = Loss;
    objfun_cg_forward = Loss;
    disp(['CG objfun: ' num2str(objfun_cg) ' at iter: ' num2str(cg_gamma_idxs(length(all_pn))) '.']);
    
    cgbt_min_idx = length(all_pn); % The last CG iteration that decreased.
    last_cgbt_eval_iter = length(all_pn);
    cg_found_better_solution = 0;
    cgbt_did_break = 0;
    cg_did_increase = 0; % Did the objfun increase as CG went forward in time? If not, then we may need to continue going forward.
    for i = length(all_pn)-1:-1:1
        pn = all_pn{i};
        cgtheta = net.theta + pn;
        cgnet = net;
        cgnet.theta = cgtheta;
        if ~do_parallel
            for j = 1:ntrials
                package = eval_objfun(cgnet,conditions_ntrials{j},inputs_ntrials{j},targets_ntrials{j},[]);
                objfuns{j} = package{1};
            end
        else
            parfor j = 1:ntrials
                package = eval_objfun(cgnet,conditions_ntrials{j},inputs_ntrials{j},targets_ntrials{j},[]);
                objfuns{j} = package{1};
            end
        end
        clear package;
        Loss = 0;
        for j = 1:ntrials
            Loss = Loss + objfuns{j};
        end
        Loss = Loss / ntrials;
        objfun_cg = Loss;
        disp(['CG objfun: ' num2str(objfun_cg) ' at iter: ' num2str(cg_gamma_idxs(i)) '.']);
        
        % Note that when the process itself is stochastic, improvement on
        % one instance may not apply to another, causing the objfun to
        % temporarily increase. Therefore one may also want to include '<
        % objfun' in the condition.
        if objfun_cg_forward < objfun_cg && objfun_cg_forward < objfun_new_data % && objfun_cg_forward < objfun % If the forward one was better.
            objfun_cg = objfun_cg_forward;
            cgbt_min_idx = last_cgbt_eval_iter;
            cgbt_did_break = 1;
            cg_found_better_solution = 1;
            break;
        end
        if objfun_cg_forward > objfun_cg
            cg_did_increase = 1;
        end
        cgbt_did_break = 0;
        objfun_cg_forward = objfun_cg;
        last_cgbt_eval_iter = i;
    end
    if ~cgbt_did_break && objfun_cg < objfun_new_data % && objfun_cg < objfun % Check if CG iteration 1 was useful.
        cg_found_better_solution = 1;
        cgbt_min_idx = 1;
    end
    
    % Now decide what to do based on the CG backtracking.
    if cg_found_better_solution
        cg_min_idx = cg_gamma_idxs(cgbt_min_idx);
        
        total_hf_consecutive_fail_count = 0;
        
        % Do a little line search.
        p_cg_good = all_pn{cgbt_min_idx};        
        c_good = 1;
        objfun_ls = objfun_cg;
        objfun_ls_min = objfun_cg;
        objfun_ls_last = realmax;
        
        lsnet = net;
        c = 0.98;
        i = 0;
        min_frac_ls_decrease = 0.01;
        c_decrease_val = 1.5;
        
        while objfun_ls < objfun_ls_last
            i = i+1;
            c = 1 - c_decrease_val*(1-c);
            if c < 0
                break;
            end
            lsnet.theta = net.theta + c*p_cg_good;
            objfun_ls_last = objfun_ls;
            
            if ~do_parallel
                for j = 1:ntrials
                    package = eval_objfun(lsnet,conditions_ntrials{j},inputs_ntrials{j},targets_ntrials{j},[]);
                    objfuns{j} = package{1};
                end
            else
                parfor j = 1:ntrials
                    package = eval_objfun(lsnet,conditions_ntrials{j},inputs_ntrials{j},targets_ntrials{j},[]);
                    objfuns{j} = package{1};
                end
            end
            clear package;
            Loss = 0;
            for j = 1:ntrials
                Loss = Loss + objfuns{j};
            end
            Loss = Loss / ntrials;
            objfun_ls = Loss;
            
            if objfun_ls < objfun_ls_min
                objfun_ls_min = objfun_ls;
                c_good = c;
            end
            
            % disp(['Line search objfun: ' num2str(objfun_ls) ' at iter: ' num2str(i) '.']);
            
            frac_ls_decrease = abs((objfun_ls_last - objfun_ls) / objfun_ls_last);
            if objfun_ls > objfun_ls_last || frac_ls_decrease < min_frac_ls_decrease
                break;
            end
        end
        net.theta = net.theta + c_good*p_cg_good;
        
        objfun_last = objfun;
        objfun = objfun_ls_min;
        
        % Setup conditions for the next hf iteration.
        do_resample_data = true;
        do_recompute_gradient = true;
        do_recompute_rho = true;
        
        pn_cgstart = init_decay_factor * all_pn{end};
        niters_cg = ceil(cg_min_idx * cg_increase_factor);
        
        objfun_constant_decreasing = realmax;
        ncgiters_constant_decreasing = 0;
        
    elseif ~cg_did_increase && ~cgbt_did_break && ~isinf(objfun_cg) && objfun_cg < objfun_constant_decreasing && ...
            ncgiters_constant_decreasing < max_cg_iter % In this case, CG iters were continually decreasing but never decreased below the last hf iteration objfun value.
        % Just need more iterations.
        disp('CG iterations were constantly decreasing, but not less than the objfun at last HF iteration.');
        
        % Setup conditions for the next hf iteration.
        do_resample_data = false;
        do_recompute_gradient = false;
        do_recompute_rho = false;
        
        if ~any(isnan(all_pn{end}))
            pn_cgstart = all_pn{end};
        else
            pn_cgstart = zeros(size(net.theta));
        end
        cg_min_idx = NaN;
        niters_cg = ceil(niters_cg * 2.0);
        cg_tol = cg_tol / 2.0;
        
        objfun_constant_decreasing = objfun_cg;
        ncgiters_constant_decreasing = ncgiters_constant_decreasing + niters_cg;
        
    else
        if ncgiters_constant_decreasing >= max_cg_iter
            disp(['Failed because ncgiters_constant_decreasing >= ' num2str(max_cg_iter) '.']);
        end
        
        disp('Last CG evaluation wasn''t good enough. Trying to increase lambda. Reseting CG start. Resampling data.');
        
        total_hf_fail_count = total_hf_fail_count + 1;
        total_hf_consecutive_fail_count = total_hf_consecutive_fail_count + 1;
        
        % Setup conditions for the next hf iteration.
        do_resample_data = true;
        do_recompute_gradient = true;
        do_recompute_rho = false;
        
        lambda = rho_boost_val * lambda;
        pn_cgstart = zeros(size(net.theta));
        cg_min_idx = NaN;
        niters_cg = min_cg_iter;
        
        objfun_constant_decreasing = realmax;
        ncgiters_constant_decreasing = 0;
    end
    
    % Hard limits for CG iterations.
    if niters_cg < min_cg_iter
        niters_cg = min_cg_iter;
    elseif niters_cg > max_cg_iter
        niters_cg = max_cg_iter;
    end
    
    if do_recompute_rho && hf_iter > 1
        % rho = (f(theta+p) - f(theta)) / (phi_theta(p) - phi_theta(0)) is
        % used to heuristically compute lambda, a crucial parameter
        % denoting the trustworthy region of the quadratic approximation of
        % f: phi = phi_theta(p) = 1/2 p' * A * p - b' * p, which is
        % minimized by CG. It is often found to be more effective to have
        % the numerator computed on the same subset of data that is used to
        % compute the Gv products, as this lets lambda be reduced more
        % aggressively, but one should also try to compute the numerator on
        % a larger dataset like the whole training set.

        cg_phi = all_cg_phis(cgbt_min_idx);
        rho_numer = objfun - objfun_last;
        rho_denom = cg_phi;
        rho = rho_numer / rho_denom;        
        if rho > 0.0
            if rho < rho_boost_thresh
                lambda = rho_boost_val * lambda;
            elseif rho > rho_drop_thresh
                lambda = rho_drop_val * lambda;
            end
        end
        if lambda ~= 0.0
            if lambda < min_lambda
                lambda = min_lambda;
            end
        end
        
        disp(['CG phi: ' num2str(cg_phi) ', rho: ' num2str(rho), ', lambda: ' num2str(lambda,6) '.']);
        
    else
        disp(['Lambda: ' num2str(lambda,6) '.']);
    end
    
    % Display the result of current hf iteration.
    disp(['Objective function : ' num2str(objfun), '.']);
    if cg_found_better_solution
        if hf_iter > 1
            disp(['Objective function - objective function last: ', num2str(objfun - objfun_last) '.']);
        end
    end
    
    % Terminate conditions.
    if grad_norm < grad_norm_tol
        stop_string = ['Stopping because the magnitude of gradient was less than ' num2str(grad_norm_tol) '.'];
        go = 0;
    end
    
    if abs(objfun - objfun_last) < objfun_diff_tol
        stop_string = ['Stopping because the difference in objective function fell below: ' num2str(objfun_diff_tol) '.'];
        go = 0;
    end
    
    if objfun < objfun_tol
        stop_string = ['Stopping because the objective function fell below: ' num2str(objfun_tol) '.'];
        go = 0;
    end
    
    if total_hf_fail_count > max_hf_fail_count
        stop_string = ['Stopping because the total number of HF iteration failures was greater than ' num2str(max_hf_fail_count) '.'];
        go = 0;
    end
    
    if total_hf_consecutive_fail_count > max_hf_consecutive_fail_count
        stop_string = ['Stopping because HF iteration failed ' num2str(max_hf_consecutive_fail_count) ' times in a row.'];
        go = 0;
    end
    
    if lambda >= max_lambda
        stop_string = ['Stopping because lambda was greater than the max lambda: ' num2str(lambda) ' > ' num2str(max_lambda) '.'];
        go = 0;
    end
    
    % Timing.
    hf_iter_time = toc;
    total_time = total_time + hf_iter_time;
    net.total_time = total_time;
    disp(['Elapsed time: ' num2str(hf_iter_time) ' seconds.']);

    % Take snapshots.
    if mod(hf_iter, save_every) == 0
        save([save_path '/hfopt_' num2str(hf_iter) '.mat'], 'net');
    end
end

%% Exit.
save([save_path '/hfopt_' num2str(hf_iter) '.mat'], 'net');
disp(stop_string);
hours = floor(total_time/3600);
minutes = floor((total_time - hours*3600)/60);
seconds = total_time - hours*3600 - minutes*60;
disp(['Elapsed time in total: ' num2str(hours) ' hours, ' num2str(minutes) ' minutes, and ' num2str(seconds) ' seconds.']);
theta_opt = net.theta;

end