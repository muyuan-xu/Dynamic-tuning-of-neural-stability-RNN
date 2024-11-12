function [fp_structs, D_fps] = find_all_fps(net, nfps, starting_points, init_radius, max_radius, tolq, varargin)
% Front end to finding fixed points for the Hessian Free optimization code.
%
% net - the matlab structure of the network being analyzed.
%
% nfps - the number of fixed points to find.
%
% starting_points - an array of points to start with, typically taken from
% network trajectories.
%
% init_radius - pick a random point within this range to a given
% starting_point, and use it as the initial condition of optimization.
%
% max_radius - the maximal range we try before exiting or starting over.
%
% tolq - the maximal value of fixed point we'll accept. This is useful for
% fine control of finding iso-speed contours or extremely accurate fixed
% points.
const_input = [];
do_topo_map = true; % Don't worry about whether the speed is a true local minima, just find low speed areas.
do_bail = 1; % Bail after we hit the max radius and still fail.
radius_factor = 2.0; % Factor by which we increase the radius.
stability_analysis = 'full';

display = 'off'; % Level of display for fminunc. 'off or 'iter', for example.
maxiter = 1e4; % Maximum number of iterations for fminunc.
tolfun = 1e-16; % Termination tolerance on the function value for fminunc. Perhaps overly small, but it's cautious.
tolx = 1e-16; % Termination tolerance on x for fminunc. Perhaps overly small, but it's cautious.

optargin = size(varargin,2);
for i = 1:2:optargin
    switch varargin{i}
        case 'const_input' % Allows input-dependent fixed points.
            const_input = varargin{i+1};
        case 'do_topo_map' % Help to find areas of slow speed. This will return the first x whose value is less than tolq.
            do_topo_map = varargin{i+1};
        case 'do_bail' % If the optimization continues to fail the tolerances, should we exit or start over?
            do_bail = varargin{i+1};
        case 'radius_factor' % How much do we increase the radius after a failure.
            radius_factor = varargin{i+1};
        case 'stability_analysis' % 'compact' returns only eigenvalues, 'full' returns eigenvectors as well.
            stability_analysis = varargin{i+1};
        case 'display' % Level of display for fminunc. 'off or 'iter', for example.
            display = varargin{i+1};
        case 'tolfun' % Termination tolerance on the function value for fminunc.
            tolfun = varargin{i+1};
        case 'tolx' % Termination tolerance on x for fminunc.
            tolx = varargin{i+1};
        case 'maxiters' % Maximum number of iterations for fminunc.
            maxiter = varargin{i+1};
        otherwise
            assert(false, ['Variable argument ' varargin{i} ' not recognized.']);
    end
end

N = net.layers(2).npost;

fps = cell(1,nfps);
qs = zeros(1,nfps);
startps = cell(1,nfps);
eigvals = cell(1,nfps);
eigvecs_right = cell(1,nfps);
eigvecs_left = cell(1,nfps);
neigs_uns = zeros(1,nfps); % Number of unstable eigenvalues.

% Create optimization options structure for fminunc.
options = optimset('display', display, 'maxiter', maxiter, 'tolfun', tolfun, 'tolx', tolx, ...
    'gradobj','on', 'hessian','on', 'largescale', 'on'); % Use specified grad and hessian.

tic
nstartps = size(starting_points,2);
if nfps < nstartps
    x_idxs = randi(nstartps, [1 nfps]); % Cover a random set of options.
else
    x_idxs = [[1:nstartps] randi(nstartps, [1 (nfps-nstartps)])]; % Cover all options, then random.
end

% Main.
for i = 1:nfps
    radius = init_radius;
    fps{i} = zeros(N,1);
    qs(i) = inf;
    while true
        % Set the initial condition of optimization.
        x0 = starting_points(:,x_idxs(i)) + radius * randn(N,1);
        disp(['Starting for the ' num2str(i) '-th fp from a random point with norm: ' num2str(norm(x0)) '.']);
        startps{i} = x0;
        
        % Fixed points are found by minimizing the function q(x) = 1/2
        % |F(x)|^2, where dx/dt = F(x) describes the (continuous) dynamics
        % of RNN. See Mante et al., 2013.
        [fp, q, exitflag] = fminunc(@(x) find_one_fp(net, x, const_input, do_topo_map, tolq), x0, options);
        
        do_quit = false;
        switch exitflag
            case 1
                disp('Finished. Magnitude (infinity norm) of the gradient is smaller than the tolerance.');
                do_quit = true;
            case 2
                disp(['Finished. Change in x is smaller than ' num2str(tolx) '.']);
                do_quit = true;
            case 3
                disp(['Finished. Change in objfun is smaller than ' num2str(tolfun) '.']);
                do_quit = true;
            case 5
                disp(['Finished. Predicted decrease in objfun is smaller than ' num2str(tolfun) '.']);
                do_quit = true;
            case 0
                disp('Finished. Number of iterations or function evaluations exceeded maximum.');
                if do_topo_map
                    do_quit = true;
                end
            case -1
                disp('Error! Algorithm was terminated by the output function.');
            otherwise
                assert(false, 'Error! Algorithm was terminated due to unknown reason.');
        end
        assert(isreal(fp), 'Error! Fixed points must be real.');
        fps{i} = fp;
        qs(i) = q;
        disp(['q = 1/2 |F(x)|^2 = ' num2str(q) '.']);
        
        % Accept the fixed point if it fell below the tolerance. If one
        % wants all local minima, then set do_topo_map false and tolq high.
        if do_quit && q < tolq
            if strcmp(stability_analysis, 'full')
                do_return_eigvecs = true;
                [eigvals{i}, neigs_uns(i), eigvecs_right{i}, eigvecs_left{i}] = get_linear_stability(net, fps{i}, do_return_eigvecs);
            elseif strcmp(stability_analysis, 'compact')
                do_return_eigvecs = false;
                [eigvals{i}, neigs_uns(i), eigvecs_right{i}, eigvecs_left{i}] = get_linear_stability(net, fps{i}, do_return_eigvecs);
            else
                eigvals{i} = [];
                neigs_uns(i) = 0;
                eigvecs_right{i} = [];
                eigvecs_left{i} = [];
            end
            disp(['Finished ' num2str(i) '-th fp, norm: ' num2str(norm(fps{i})) ' with ' num2str(neigs_uns(i)) ' unstable eigenvalues.']);
            break;
        else
            radius = radius * radius_factor; % Start over with a larger searching radius.
            if radius > max_radius
                if ~do_bail
                    radius = init_radius;
                else
                    qs(i) = inf;
                    fps{i} = NaN(N,1);
                    neigs_uns(i) = inf;
                    disp(['Couldn''t find fixed point with tolerance ' num2str(tolq) '. Bailing.']);
                    break;
                end
            end
            if ~do_quit
                disp(['Try again for ' num2str(i) '-th fp, because the point with norm ' num2str(norm(fp)) ' didn''t meet the criteria.']);
            else
                disp(['Try again for ' num2str(i) '-th fp, because the point with norm ' num2str(norm(fp)) ' didn''t meet the tolerance of q.']);
            end
            disp(['Radius = ' num2str(radius) '.']);
        end
    end
end
toc

% Collect the results.
D_fps = zeros(nfps,nfps);
for i = 1:nfps
    for j = 1:nfps
        D_fps(i,j) = norm(fps{i} - fps{j});
    end
end

for i = 1:nfps
    fp_structs(i).fp = fps{i};
    fp_structs(i).fpnorm = norm(fps{i});
    fp_structs(i).q = qs(i);
    fp_structs(i).neigs_uns = neigs_uns(i);
    fp_structs(i).eigvals = eigvals{i};
    fp_structs(i).eigvecs = eigvecs_right{i};
    fp_structs(i).eigvecs_left = eigvecs_left{i};
    fp_structs(i).startp = startps{i};
end

disp('Fixed point search complete.');

end
