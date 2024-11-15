%% Load the trained network.
load('./data/hfopt_.mat')
net.layers(1).transfun = @(x) x;
net.layers(1).Doperator = @(y) ones(size(y));
net.layers(2).transfun = @ tanh;
net.layers(2).Doperator = @(y) 1.0-y.^2;
net.layers(3).transfun = @(x) x;
net.layers(3).Doperator = @(y) ones(size(y));
inv_trans_fun = net.layers(2).invfun;

[n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = unpackRNN(net, net.theta);

[N,V] = size(n_Wru_v);
M = size(m_Wzr_n,1);

%% Setup the input.
dt = net.dt;

T_transient   = 100; % Duration of the rule-dependent input in the transient-rule paradigm.
T_minprecue   = 200; % Minimal length of the precue period (before the switch of the phase-dependent inputs).
T_maxprecue   = 500; % Maximal length of the precue period.
T_delay       = 100; % This corresponds to the delay period.
T_sensory     = 200; % Duration of the sensory input/integration.
T_minpostintg = 0;   % Minimal length of the trial after sensory integration.
T = T_maxprecue + T_delay + T_sensory + T_minpostintg; % Total length of the trial.

ntransient    = floor(T_transient/dt);
nminprecue    = floor(T_minprecue/dt);
ndelay        = floor(T_delay/dt);
nsensory      = floor(T_sensory/dt);
nminpostintg  = floor(T_minpostintg/dt);

times  = dt:dt:T;
ntimes = length(times);

num_of_rules = 2;
rules = 1:num_of_rules;

num_of_precue_intervals = 5; % How many different precue intervals do we use to train the network.
all_nprecue   = floor(linspace(T_minprecue, T_maxprecue, num_of_precue_intervals)/dt);
all_npostintg = ntimes - all_nprecue - ndelay - nsensory;

num_of_stimuli = 2;
assert(V == 4 + num_of_stimuli, 'Error input dimension!');
M_rule    = 0.5; % Magnitude of the rule-dependent input.
M_phase   = 0.5; % Magnitude of the phase-dependent input.
M_sensory = 0.1; % Magnitude of the sensory input.
input_noise_sigma = 0.1;

ntrials = num_of_stimuli * num_of_precue_intervals * num_of_rules;

init_conditions = cell(1, ntrials); % Allows one to set different initial conditions.
inputs          = cell(1, ntrials);

for i = 1:ntrials
    init_conditions{i} = 1;
    inputs{i} = input_noise_sigma * sqrt(dt) * randn(V, ntimes);
    [idx_stimulus, idx_precue, idx_rule] = ind2sub([num_of_stimuli num_of_precue_intervals num_of_rules], i);
    % Define the rule-dependent inputs.
    inputs{i}(idx_rule, :) = inputs{i}(idx_rule, :) + M_rule * ones(1, ntimes); % Tonic rule paradigm.
    % inputs{i}(idx_rule, 1:ntransient) = inputs{i}(idx_rule, 1:ntransient) + M_rule * ones(1, ntransient); % Transient rule paradigm.
    % Define the phase-dependent inputs.
    nprecue   = all_nprecue(idx_precue);
    npostintg = all_npostintg(idx_precue);
    inputs{i}(3, :) = inputs{i}(3, :) + [M_phase * ones(1, nprecue) zeros(1, ndelay + nsensory + npostintg)];
    inputs{i}(4, :) = inputs{i}(4, :) + [zeros(1, nprecue) M_phase * ones(1, ndelay + nsensory + npostintg)];
    % Define the sensory inputs.
    inputs{i}(4 + idx_stimulus, :) = inputs{i}(4 + idx_stimulus, :) + [zeros(1, nprecue + ndelay) M_sensory * ones(1, nsensory) zeros(1, npostintg)];
end

%% Do parallel computing.
do_parallel = true;
nworkers = 7;
if do_parallel
    mycluster = parcluster('local');
    if mycluster.NumWorkers < nworkers
        mycluster.NumWorkers = nworkers;
        saveProfile(mycluster);
    end
    if matlabpool('size') == 0
        matlabpool('local', nworkers);
    elseif matlabpool('size') ~= nworkers
        matlabpool close;
        matlabpool('local', nworkers);
    end
end

%% Generate simulated data.
if ~do_parallel
    for i = 1:ntrials
        package = eval_network(net, init_conditions{i}, inputs{i}, []);
        forward_passes{i} = package{1};
    end
else
    parfor i = 1:ntrials
        package = eval_network(net, init_conditions{i}, inputs{i}, []);
        forward_passes{i} = package{1};
    end
end
clear package;

for i = 1:ntrials
    [idx_stimulus, idx_precue, idx_rule] = ind2sub([num_of_stimuli num_of_precue_intervals num_of_rules], i);
    nprecue   = all_nprecue(idx_precue);
    npostintg = all_npostintg(idx_precue);
    % Align data at stimulus onset.
    Rs{i} = forward_passes{i}{1}(:, nprecue-nminprecue+1:nprecue+ndelay+nsensory+nminpostintg);
    Xs{i} = forward_passes{i}{5}(:, nprecue-nminprecue+1:nprecue+ndelay+nsensory+nminpostintg);
    Zs{i} = forward_passes{i}{3}(:, nprecue-nminprecue+1:nprecue+ndelay+nsensory+nminpostintg);
end

% Plot sample neuron responses.
num_to_show = 10;
colors = jet(num_to_show);
lw = 2.5;  % Linewidth.
ms = 10.0; % Markersize.
fs = 20.0; % Fontsize.

figure;
for i = 1:num_to_show
    plot(Rs{1}(i, :), 'color', colors(i, :), 'linewidth', lw);
    hold on
end
axis([0 T -1.5 1.5])
set(gca, 'fontname', 'Times New Roman', 'linewidth', lw, 'fontsize', fs)
xlabel('\bfTime')
ylabel('\bfx')
% set(gca, 'xtick', [])
% set(gca, 'ytick', [])

% Plot network output.
colors = jet(ntrials);
lw = 2.5;  % Linewidth.
ms = 10.0; % Markersize.
fs = 20.0; % Fontsize.

figure;
for i = 1:ntrials
    plot(Zs{i}(1, :), 'color', colors(i, :), 'linewidth', lw);
    hold on
end
axis([0 T -1.2 1.2])
set(gca, 'fontname', 'Times New Roman', 'linewidth', lw, 'fontsize', fs)
xlabel('\bfTime')
ylabel('\bfz')
% set(gca, 'xtick', [])
% set(gca, 'ytick', [])

%% Principal component analysis.
sample_rate = 1; % Downsample if needed.
disp(['Downsampling... sample rate = ' num2str(sample_rate) '.']);
for i = 1:ntrials
    Rs{i} = downsample(Rs{i}', sample_rate)';
    Xs{i} = downsample(Xs{i}', sample_rate)';
    Zs{i} = downsample(Zs{i}', sample_rate)';
end

R = [Rs{:}];
X = [Xs{:}];
Z = [Zs{:}];

Rbar = mean(R, 2);
Rz   = bsxfun(@minus, R, Rbar);
CR   = (Rz * Rz')/size(Rz, 2);
[V, DR] = eig(CR);
dr   = diag(DR);
[dr, sidx_r] = sort(dr, 'descend');
dr(dr<eps) = eps;
Rpca.C = CR;
Rpca.V = V(:, sidx_r);
Rpca.d = dr;
Rpca.sortIdx = 1:N;
Rpca.mean = Rbar;

Xbar = mean(X, 2);
Xz   = bsxfun(@minus, X, Xbar);
CX   = (Xz * Xz')/size(Xz, 2);
[V, DX] = eig(CX);
dx   = diag(DX);
[dx, sidx_x] = sort(dx, 'descend');
dx(dx<eps) = eps;
Xpca.C = CX;
Xpca.V = V(:, sidx_x);
Xpca.d = dx;
Xpca.sortIdx = 1:N;
Xpca.mean = Xbar;

% % Plot explained variance.
% lw = 2.5;  % Linewidth.
% ms = 10.0; % Markersize.
% fs = 20.0; % Fontsize.
% 
% figure;
% eigs_to_show = 30;
% plot(1:eigs_to_show, log10(Xpca.d(1:eigs_to_show)), '-kx', 'linewidth', lw);
% hold on;
% plot(1:eigs_to_show, log10(Rpca.d(1:eigs_to_show)), '-rx', 'linewidth', lw);
% axis tight;
% xlabel('\lambda #', 'fontsize', fs);
% ylabel('log10(\lambda)', 'fontsize', fs);
% legend('X eigenvalues', 'R eigenvalues');
% set(gca(gcf), 'fontsize', fs);
% 
% frac_cumsum_x = cumsum(Xpca.d) / sum(Xpca.d);
% frac_cumsum_r = cumsum(Rpca.d) / sum(Rpca.d);
% var_explained = 0.9;
% disp([num2str(var_explained) ' variance explained by ']);
% thing = find(frac_cumsum_x > var_explained);
% pc_num_var_explained_x = thing(1);
% disp([num2str(pc_num_var_explained_x) ' PCs for x,']);
% thing = find(frac_cumsum_r > var_explained);
% pc_num_var_explained_r = thing(1);
% disp(['or ' num2str(pc_num_var_explained_r) ' PCs for r.']);

%% Plot model dynamics in the space determined by the first three PCs.
% Average across trials.
nconditions = ntrials; 
assert(mod(ntrials, nconditions) == 0, 'ntrials must be dividable by ncondition.')
ntrials_for_each_condition = floor(ntrials/nconditions);
for idx_group = 1:nconditions
    avrRs{idx_group} = zeros(size(Rs{1}));
    for idx_trial = (idx_group-1)*ntrials_for_each_condition + 1:idx_group*ntrials_for_each_condition
        avrRs{idx_group} = avrRs{idx_group} + Rs{idx_trial};
    end
    avrRs{idx_group} = avrRs{idx_group} / ntrials_for_each_condition;
end
for idx_group = 1:nconditions
    avrXs{idx_group} = zeros(size(Xs{1}));
    for idx_trial = (idx_group-1)*ntrials_for_each_condition + 1:idx_group*ntrials_for_each_condition
        avrXs{idx_group} = avrXs{idx_group} + Xs{idx_trial};
    end
    avrXs{idx_group} = avrXs{idx_group} / ntrials_for_each_condition;
end
for idx_group = 1:nconditions
    avrZs{idx_group} = zeros(size(Zs{1}));
    for idx_trial = (idx_group-1)*ntrials_for_each_condition + 1:idx_group*ntrials_for_each_condition
        avrZs{idx_group} = avrZs{idx_group} + Zs{idx_trial};
    end
    avrZs{idx_group} = avrZs{idx_group} / ntrials_for_each_condition;
end

% Plot dynamic trajectories.
Vshow = Xpca.V(:, 1:3);
% Make sure that the axes are orthogonal.
Vorth = orth(Vshow);

colors = jet(nconditions);
lw = 2.5;  % Linewidth.
ms = 10.0; % Markersize.
fs = 20.0; % Fontsize.

figure;
for i = 1:nconditions
    proj = Vorth' * avrXs{i};
    plot3(proj(1, :), proj(2, :), proj(3, :), 'color', colors(i, :), 'linewidth', lw)
    hold on
    % Mark stimulus onset.
    idx_p = nminprecue;
    plot3(proj(1, idx_p), proj(2, idx_p), proj(3, idx_p), 'o', 'color', colors(i, :), 'markersize', ms, 'markerfacecolor', colors(i, :), 'linewidth', lw)
    hold on
    % Mark the beginning of sensory input/integration.
    idx_p = nminprecue + ndelay;
    plot3(proj(1, idx_p), proj(2, idx_p), proj(3, idx_p), '^', 'color', colors(i, :), 'markersize', ms, 'markerfacecolor', colors(i, :), 'linewidth', lw)
    hold on
    % Mark the final state.
    plot3(proj(1, end), proj(2, end), proj(3, end), 's', 'color', colors(i, :), 'markersize', ms, 'markerfacecolor', colors(i, :), 'linewidth', lw)
    hold on
end
axis equal;

% set(gca, 'fontname', 'Times New Roman', 'linewidth', lw, 'fontsize', fs)
% set(gca, 'xtick', [])
% set(gca, 'ytick', [])
% set(gca, 'ztick', [])

%% Find equilibrium/fixed points.
nconst_input = 2; % Equilibrium/fixed points may depend on constant (contextual) input.
nfps = 2*(nconditions/2); % Number of expected equilibrium/fixed points (per constant input).
init_radius = 0.0001; % Radius of the neighborhood of the starting_point from which the search starts.
max_radius  = 0.01;
tolq   = 1.0e-6; % Tolerance on q(x) = 1/2 |F(x)|^2, where dx/dt = F(x) describes the dynamics of the RNN.
tolfun = 1.0e-6; % Tolerance on the function value for fminunc.
tolx   = 1.0e-6; % Tolerance on x for fminunc.

for i = 1:nconst_input
    if i == 1
        const_input = [1 0 0 1 zeros(1,2)]'; % Define the constant (contextual) input for the equilibrium/fixed points.
    else
        const_input = [0 1 0 1 zeros(1,2)]';
    end
    starting_points = [];
    for j = 1:nconditions
        [idx_stimulus, idx_precue, idx_rule] = ind2sub([num_of_stimuli num_of_precue_intervals num_of_rules], j);
        if idx_rule == i
            % starting_points = [starting_points avrXs{j}(:, [nminprecue+ndelay end])];
            starting_points = [starting_points .5*(avrXs{j}(:, end) + avrXs{4*(j-floor(j/2))-1-j}(:, end)) avrXs{j}(:, end)]; % Use the average of the two final states (each corresponds to one outcome contingency) as a starting point.
        end
    end
    [fp_structs{i}, D_fps{i}] = find_all_fps(net, nfps, starting_points, init_radius, max_radius, tolq, ...
        'const_input', const_input, 'do_topo_map', false, 'display', 'off', 'tolfun', tolfun, 'tolx', tolx);
end

tolfp = 1e-1; % Two fps that have a distance larger than this value will be classified as different fps.
for i = 1:nconst_input
    idx_goodfps = find([fp_structs{i}.fpnorm] < inf);
    ngoodfps = length(idx_goodfps);
    if ngoodfps > 0
        idx_fps = [idx_goodfps(1)];
        for j = 2:ngoodfps
            diff_fp = [fp_structs{i}(idx_fps).fp] - repmat(fp_structs{i}(idx_goodfps(j)).fp, 1, length(idx_fps));
            if min(sqrt(diag(diff_fp'*diff_fp))) >= tolfp % If the fp is different from all previous ones, count it as a new fp.
                idx_fps = [idx_fps idx_goodfps(j)];
            end
        end
    end
    fp_structs{i} = fp_structs{i}(idx_fps);
end

%% Plot equilibrium/fixed points.
do_plot_eigvecs = true;

lw = 2.5;  % Linewidth.
ms = 20.0; % Markersize.
fs = 20.0; % Fontsize.

figure;
for i = 1:nconst_input
    fps = fp_structs{i};
    for j = 1:length(fps)
        fp_proj = Vorth'*(fps(j).fp);

        plot3(fp_proj(1), fp_proj(2), fp_proj(3), 'p', 'linewidth', lw, 'color', 'k', 'markerfacecolor', 'k', 'markersize', ms)
        hold on;
        
        if ~do_plot_eigvecs
            continue;
        end
        
        % Find the largest real eigval and plot the corresponding eigenvectors.
        idx_eig = 0;
        while true
            idx_eig = idx_eig + 1;
            if isreal(fps(j).eigvals(idx_eig)) || idx_eig > length(fps(j).eigvals)
                break;
            end
        end
        if idx_eig <= length(fps(j).eigvals)
            ev_proj = Vorth'*(fps(j).eigvecs(:,idx_eig) + fps(j).fp);
            plot3([fp_proj(1) ev_proj(1)], [fp_proj(2) ev_proj(2)], [fp_proj(3) ev_proj(3)], 'color', 'r', 'linewidth', lw)
            hold on
            ev_proj = Vorth'*(-fps(j).eigvecs(:,idx_eig) + fps(j).fp);
            plot3([fp_proj(1) ev_proj(1)], [fp_proj(2) ev_proj(2)], [fp_proj(3) ev_proj(3)], 'color', 'r', 'linewidth', lw)
            hold on
        end
    end
end

%% Analyze the vector field before and after the switch of the phase-dependent inputs.
% Calculate the corresponding equilibrium/fixed points and intrinsic dynamics using the methods above.
fp_rule         = ; % This has to be calculated separately (by setting the constant input).
fp_contingency1 = fp_structs{1}(1);
fp_saddle       = fp_structs{1}(2);
fp_contingency2 = fp_structs{1}(3);

Xs_intrinsic_before_switch = NaN; % This has to be calculated separately (by turning off sensory input and noise).
Xs_intrinsic_after_switch  = NaN; % This has to be calculated separately (by turning off sensory input and noise).

% The three equilibrium/fixed points determine a plane.
center = fp_rule.fp;
Vorth = orth([fp_contingency1.fp - fp_rule.fp, fp_contingency2.fp - fp_rule.fp]);

% Set the region we are looking at.
grid_x = -17:1:3;
grid_y = -11:1:9;

rec_transfun = net.layers(2).transfun;
F = @(x, u) bsxfun(@plus, -x + n_Wrr_n * rec_transfun(x), n_Wru_v * u + n_bx_1) / net.tau;

points = [];
for i = grid_x
    for j = grid_y
        points = [points i * Vorth(:, 1) + j * Vorth(:, 2)];
    end
end
points = bsxfun(@plus, points, center);

figure;
% % Before the switch of the phase-dependent inputs
% field = F(points, [M_rule 0 M_phase 0 zeros(1, 2)]');
% quiver(Vorth(:, 1)'*bsxfun(@minus, points, center), Vorth(:, 2)'*bsxfun(@minus, points, center), ...
%     Vorth(:, 1)'*field, Vorth(:, 2)'*field, 5.0, 'color', 'k')
% hold on
% After the switch.
field = F(points, [M_rule 0 0 M_phase zeros(1, 2)]');
% quiver(Vorth(:, 1)'*bsxfun(@minus, points, center), Vorth(:, 2)'*bsxfun(@minus, points, center), ...
%     Vorth(:, 1)'*field, Vorth(:, 2)'*field, 5.0, 'color', 'k')
% hold on

% Normalize the vectors for visualization purpose.
field_x = Vorth(:, 1)'*field;
field_y = Vorth(:, 2)'*field;
field_norm = sqrt(field_x.^2 + field_y.^2);
field_x = field_x ./ field_norm;
field_y = field_y ./ field_norm;
quiver(Vorth(:, 1)'*bsxfun(@minus, points, center), Vorth(:, 2)'*bsxfun(@minus, points, center), ...
    field_x, field_y, .75, 'color', 'k')
hold on

% Plot dynamic trajectories.
color_transform = @(x) flipud([0:num_of_precue_intervals - 1]'/num_of_precue_intervals * ([.8 .8 .8] - x) + repmat(x, num_of_precue_intervals, 1)); % Gradually shift to a darker color.
colors = [color_transform([0 0 1]); color_transform([0 1 1]); color_transform([1 0 0]); color_transform([1 0 1])];

lw = 2.5;  % Linewidth.
ms = 7.5;  % Markersize.
fs = 20.0; % Fontsize.

for i = 1:floor(ntrials/2)
    proj = Vorth'*(bsxfun(@minus, Xs{i}, center));
    % plot(proj(1, 1:nminprecue), proj(2, 1:nminprecue), 'color', colors(i, :), 'linewidth', lw)
    plot(proj(1, nminprecue+1:end), proj(2, nminprecue+1:end), 'color', colors(i, :), 'linewidth', lw)
    hold on
end

proj_intrinsic_before_switch = Vorth'*(bsxfun(@minus, Xs_intrinsic_before_switch{1}, center));
proj_intrinsic_after_switch  = Vorth'*(bsxfun(@minus, Xs_intrinsic_after_switch{1},  center));

plot(proj_intrinsic_before_switch(1, :), proj_intrinsic_before_switch(2, :), 'color', [.5 .5 .5], 'linewidth', lw)
hold on
plot(proj_intrinsic_after_switch(1, :),  proj_intrinsic_after_switch(2, :),  'color', [.5 .5 .5], 'linewidth', lw)
hold on

for i = 1:floor(ntrials/2)
    % Mark the beginning of sensory input/integration.
    idx_p = nminprecue + ndelay;
    plot(proj(1, idx_p), proj(2, idx_p), 'k^', 'markersize', ms, 'linewidth', lw)
    hold on
end
axis equal

% Plot equilibrium/fixed points.
linelen = 2.5; % Length of the linearized 1D unstable manifold.

% % Before the switch.
% fp_proj = Vorth'*(fp_rule.fp - center); % The stable equilibrium/fixed point.
% plot(fp_proj(1), fp_proj(2), 'kx', 'markersize', 12.5, 'linewidth', lw)
% hold on

% After the switch.
fp_proj = Vorth'*(fp_contingency1.fp - center); % The 1st stable equilibrium/fixed point.
plot(fp_proj(1), fp_proj(2), 'kx', 'markersize', 12.5, 'linewidth', lw)
hold on

fp_proj = Vorth'*(fp_sadlle.fp - center); % The saddle point.
plot(fp_proj(1), fp_proj(2), 'ko', 'markersize', 10, 'linewidth', lw)
hold on
ev_proj = Vorth'*(linelen * fp_saddle.eigvecs(:, 1) + fp_saddle.fp - center);
plot([fp_proj(1) ev_proj(1)], [fp_proj(2) ev_proj(2)], 'color', 'g', 'linewidth', lw)
hold on
ev_proj = Vorth'*(-linelen * fp_saddle.eigvecs(:, 1) + fp_saddle.fp - center);
plot([fp_proj(1) ev_proj(1)], [fp_proj(2) ev_proj(2)], 'color', 'g', 'linewidth', lw)
hold on

fp_proj = Vorth'*(fp_contingency2.fp - center); % The 2nd stable equilibrium/fixed point.
plot(fp_proj(1), fp_proj(2), 'kx', 'markersize', 12.5, 'linewidth', lw)
hold on

axis([-18 4 -12 10])
set(gca, 'fontname', 'Times New Roman', 'linewidth', lw, 'fontsize', fs)
set(gca, 'xtick', [])
set(gca, 'ytick', [])

% Mark the converging and diverging vectors, respectively.
plot([-3.0  2.0], [-2.5 -2.5], 'm', 'linewidth', lw)
hold on
plot([2.0   2.0], [-2.5  2.5], 'm', 'linewidth', lw)
hold on
plot([-3.0  2.0], [2.5   2.5], 'm', 'linewidth', lw)
hold on
plot([-3.0 -3.0], [-2.5  2.5], 'm', 'linewidth', lw)
hold on

plot([-11.5  -6.5], [-2.0 -2.0], 'r', 'linewidth', lw)
hold on
plot([-6.5   -6.5], [-2.0  3.0], 'r', 'linewidth', lw)
hold on
plot([-11.5  -6.5], [3.0   3.0], 'r', 'linewidth', lw)
hold on
plot([-11.5 -11.5], [-2.0  3.0], 'r', 'linewidth', lw)

% axis([-3 1 -2 2])

% axis([-10 -6 -1 3])

%% Calculate the overlap between the unstable manifold and input dimensions.
eigvecs = fp_saddle.eigvecs;
vec_m    = eigvecs(:, 1);
vec_in_1 = n_Wru_v(:, 5);
vec_in_2 = n_Wru_v(:, 6);
vec_m    = vec_m / norm(vec_m);
vec_in_1 = vec_in_1 / norm(vec_in_1);
vec_in_2 = vec_in_2 / norm(vec_in_2);

angle0 = acosd(dot(vec_in_1, vec_in_2));
angle0 = min([angle0 180-angle0]);
angle1 = acosd(dot(vec_m, vec_in_1));
angle1 = min([angle1 180-angle1]);
angle2 = acosd(dot(vec_m, vec_in_2));
angle2 = min([angle2 180-angle2]);

disp(['Angle between two input dimensions: ' num2str(angle0) '.'])
disp(['Angle between the unstable manifold and input dimension 1: ' num2str(angle1) '.'])
disp(['Angle between the unstable manifold and input dimension 2: ' num2str(angle2) '.'])

% Randomized permutation test.
N = 100; % Dimensionality.
nshuffle = 1e5;

% Assuming that we have two orthogonal N-dimensional vectors v1 = (1, 1, 0,
% 0,...) and v2 = (1, -1, 0, 0,...). The internal bisector of the angle
% between v1 and v2 is (1, 0, 0, 0,...).
vec1 = [1; 1; zeros(N-2, 1)];
vec2 = [1; -1; zeros(N-2, 1)];
vec1 = vec1 / norm(vec1);

angles = NaN(nshuffle, 1);

% Randomly choose a third vector, v3, from the orthogonal complementary
% space of the internal bisector, and calculate the angle between v1 and
% v3.
for i = 1:nshuffle
    vec3 = [0; randn(N-1, 1)]; % Use normal distribution because it is rotationally symmetric.
    vec3 = vec3 / norm(vec3);
    angle = acosd(dot(vec1, vec3)); % Due to symmetry, the angle between v2 and v3 is identical to the angle between v1 and v3. 
    angles(i, 1) = min([angle 180-angle]);
end

figure;
hist(angles, 200)
disp(['The null distribution: ' num2str(mean(angles)) ' ' char(177) ' ' num2str(std(angles)) '.'])
disp(['The p-value: ' num2str(sum(angles<.5*(angle1+angle2))/nshuffle) '.'])