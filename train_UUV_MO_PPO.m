clear; clc; rng(0);

%% Environment setup
env = UUVEnvMO();
stateDim = env.getObservationDim();
patchSize = 2 * env.patchRadius + 1;
patchArea = patchSize * patchSize;
globalDim = stateDim - patchArea;
if globalDim <= 0
    error('Global feature dimension must be positive. Check UUVEnvMO observation split.');
end
nActions = 4;      % 1:forward 2:turn-left 3:turn-right 4:sonar
kObj = 4;          % reward dimensions (time, coverage, sonar use, collision)

%% Hyperparameters
numEpisodes   = 3000;
gamma         = 0.99;
actorLR       = 1e-3;
epsilonClip   = 0.2;
valueLossCoef = 0.5;
sharedDim     = 64;
gradDecay     = 0.9;
sqGradDecay   = 0.999;

% Adaptive weight parameters (Pearson correlation method)
w_obj = ones(kObj,1) / kObj;
histMax = 128;  % Increased to match paper's window size
G_hist = zeros(kObj, histMax);
U_hist = zeros(1, histMax);
histCount = 0;
weightWindowSize = 128;  % Sliding window for weight calculation
updateWeightEvery = 10;
warmupEpisodes = 100;

%% Episode visualization configuration
vizStartEp = 2773;                              % First episode to record
vizEndEp = 2775;                                % Last episode to record
videoFilename = 'UUV_episodes_2771_2775.mp4';   % Output video filename
videoFrameRate = 10;                            % Frames per second
videoQuality = 95;                              % Compression quality (0-100)
showVortexField = true;                         % Enable/disable vortex visualization
videoWriter = [];                               % VideoWriter object (initialized when needed)
trajectoryBuffer = [];                          % Stores [x, y] positions for current episode
episodeVizFigure = [];                          % Figure handle for rendering frames (separate from training monitor)

%% Initialize visualization data structures
history = struct();
history.episodes = [];           % episode编号数组
history.coverage = [];           % 覆盖率数组
history.utility = [];            % 效用值数组
history.weights = zeros(4, 0);   % 4xN权重矩阵
history.collisions = [];         % 累积碰撞数组
collisionCount = 0;              % 碰撞计数器

% 创建可视化窗口
vizFig = figure('Name', 'UUV Training Monitor', 'NumberTitle', 'off', ...
                'Position', [100, 100, 1200, 800]);

% 子图1: 覆盖率
subplot(2, 2, 1);
hCoverage = plot(nan, nan, 'b-', 'LineWidth', 1.5);
xlabel('Episode'); ylabel('Coverage (%)'); 
title('Coverage Progress');
grid on; ylim([0, 100]);

% 子图2: 效用值
subplot(2, 2, 2);
hUtility = plot(nan, nan, 'r-', 'LineWidth', 1.5);
hold on; yline(0, 'k--', 'LineWidth', 1);
xlabel('Episode'); ylabel('Utility Value');
title('Utility Value Trend');
grid on;

% 子图3: 权重
subplot(2, 2, 3);
hWeights = plot(nan, nan, 'LineWidth', 1.5);  % 将返回4个句柄
xlabel('Episode'); ylabel('Weight Value');
title('Objective Weights');
legend('Time', 'Coverage', 'Sonar', 'Collision', 'Location', 'best');
grid on; ylim([0, 1]);

% 子图4: 碰撞率
subplot(2, 2, 4);
hCollision = plot(nan, nan, 'm-', 'LineWidth', 1.5);
xlabel('Episode'); ylabel('Collision Rate (%)');
title('Collision Rate');
grid on; ylim([0, 100]);

%% Initialize shared policy/value network (CNN + MLP trunk)
policyValueNet = createUUVPolicyValueNet(patchSize, globalDim, nActions, kObj, sharedDim);
avgGrad = initializeAdamState(policyValueNet);
avgGradSq = initializeAdamState(policyValueNet);
adamIter = 0;

%% Training loop
for ep = 1:numEpisodes

    state = env.reset();
    maxSteps = env.maxSteps;
    
    % Initialize video recording at start of visualization range
    if ep == vizStartEp
        try
            videoWriter = initializeVideoRecording(videoFilename, videoFrameRate);
            if ~isempty(videoWriter)
                fprintf('=== Starting video recording: Episodes %d-%d ===\n', vizStartEp, vizEndEp);
            else
                warning('Video recording initialization failed. Training will continue without visualization.');
            end
        catch ME
            warning('Error initializing video recording: %s. Training will continue without visualization.', getReport(ME, 'basic'));
            videoWriter = [];
        end
    end
    
    % Reset trajectory buffer for new episode
    if ep >= vizStartEp && ep <= vizEndEp
        try
            trajectoryBuffer = [];
        catch ME
            warning('Error resetting trajectory buffer at ep %d: %s', ep, getReport(ME, 'basic'));
            trajectoryBuffer = [];
        end
    end

    patchesEp    = zeros(patchSize, patchSize, 1, maxSteps, 'single');
    globalFeatEp = zeros(globalDim, maxSteps, 'single');
    actionsEp    = zeros(1, maxSteps);
    oldLogProbsEp = zeros(1, maxSteps);
    rewardsEp    = zeros(kObj, maxSteps);

    done = false;
    t = 0;
    info = struct('coverage',0,'collision',0,'stepCount',0,'sonarSteps',0);

    while ~done && t < maxSteps
        t = t + 1;
        [patchTensor, globalVec] = splitObservation(state, patchSize, globalDim);
        patchesEp(:,:,1,t) = patchTensor;
        globalFeatEp(:, t) = globalVec;

        singlePatch = reshape(patchTensor, patchSize, patchSize, 1, 1);
        dlPatchSingle = dlarray(singlePatch, 'SSCB');
        dlGlobalSingle = dlarray(globalVec, 'CB');
        [actorLogits, ~] = forwardMultiHeadCritic(policyValueNet, dlPatchSingle, dlGlobalSingle);
        logits = extractdata(actorLogits);
        pi = softmaxLocal(logits);

        a = sampleAction(pi);
        actionsEp(t) = a;
        oldLogProbsEp(t) = log(max(pi(a), 1e-8));

        [nextState, rVec, done, info] = env.step(a);
        rewardsEp(:, t) = rVec;
        state = nextState;
        
        % Capture frame for visualization
        if ep >= vizStartEp && ep <= vizEndEp
            % Add null check for videoWriter before frame operations
            if ~isempty(videoWriter) && isvalid(videoWriter)
                try
                    % Append current UUV position to trajectory buffer
                    trajectoryBuffer(end+1, :) = env.pos';
                    
                    % Create metrics structure with episode, timestep, coverage, rewards, action
                    metrics = struct('episode', ep, ...
                                     'timestep', t, ...
                                     'coverage', info.coverage, ...
                                     'rewards', rewardsEp(:, 1:t), ...
                                     'action', a);
                    
                    % Call captureFrame() with current state
                    captureFrame(videoWriter, env, trajectoryBuffer, metrics, showVortexField);
                catch ME
                    warning('Frame capture failed at ep %d, step %d: %s. Training continues.', ep, t, getReport(ME, 'basic'));
                    % Training continues even if frame capture fails
                end
            end
        end
    end

    T = t;
    patchesEp     = patchesEp(:,:,:,1:T);
    globalFeatEp  = globalFeatEp(:, 1:T);
    actionsEp     = actionsEp(1:T);
    oldLogProbsEp = oldLogProbsEp(1:T);
    rewardsEp     = rewardsEp(:, 1:T);

    %% Return per objective
    G = zeros(kObj, T);
    for i = 1:kObj
        G(i, T) = rewardsEp(i, T);
        for tt = T-1:-1:1
            G(i, tt) = rewardsEp(i, tt) + gamma * G(i, tt+1);
        end
    end

    %% Critic estimates and advantages
    dlPatchBatch = dlarray(patchesEp, 'SSCB');
    dlGlobalBatch = dlarray(globalFeatEp, 'CB');
    [~, dlValuePred] = forwardMultiHeadCritic(policyValueNet, dlPatchBatch, dlGlobalBatch);
    V = double(extractdata(dlValuePred));
    advantages = G - V;
    for i = 1:kObj
        m = mean(advantages(i,:));
        sdev = std(advantages(i,:)) + 1e-8;
        advantages(i,:) = (advantages(i,:) - m) / sdev;
    end
    A_scalar = w_obj' * advantages;

    %% PPO update via Adam
    adamIter = adamIter + 1;
    [~, gradients] = dlfeval(@ppoLoss, policyValueNet, dlPatchBatch, dlGlobalBatch, actionsEp, oldLogProbsEp, A_scalar, G, epsilonClip, valueLossCoef);
    [policyValueNet, avgGrad, avgGradSq] = adamupdate(policyValueNet, gradients, avgGrad, avgGradSq, adamIter, actorLR, gradDecay, sqGradDecay);

    %% Adaptive weight bookkeeping
    G_ep = sum(rewardsEp, 2);
    coverage   = info.coverage;
    collision  = info.collision;
    steps      = info.stepCount;
    sonarSteps = info.sonarSteps;
    U = computeUtility(coverage, collision, steps, sonarSteps, env.maxSteps);

    histCount = histCount + 1;
    idx = mod(histCount-1, histMax) + 1;
    G_hist(:, idx) = G_ep;
    U_hist(1, idx) = U;

    % Update adaptive weights using Pearson correlation (Paper Eq. 28)
    if ep >= warmupEpisodes && mod(ep, updateWeightEvery) == 0 && histCount >= kObj
        w_obj = updateAdaptiveWeights(G_hist, U_hist, weightWindowSize);
    end
    
    %% Render transition frame between episodes
    if ep >= vizStartEp && ep <= vizEndEp
        % Add null check for videoWriter before frame operations
        if ~isempty(videoWriter) && isvalid(videoWriter)
            try
                % Create episodeSummary structure with final metrics
                episodeSummary = struct('episode', ep, ...
                                        'coverage', coverage, ...
                                        'steps', steps, ...
                                        'collision', collision, ...
                                        'utility', U);
                
                % Call renderTransitionFrame() to add transition
                renderTransitionFrame(videoWriter, episodeSummary);
                
                % Reset trajectory buffer for next episode
                trajectoryBuffer = [];
            catch ME
                warning('Transition frame rendering failed at ep %d: %s. Training continues.', ep, getReport(ME, 'basic'));
                % Training continues even if transition frame fails
            end
        end
    end
    
    %% Finalize video recording at end of visualization range
    if ep == vizEndEp
        % Add null check for videoWriter before finalization
        if ~isempty(videoWriter)
            try
                % Call finalizeVideoRecording() to close video
                finalizeVideoRecording(videoWriter, videoFilename);
            catch ME
                warning('Error finalizing video recording: %s. Training continues.', getReport(ME, 'basic'));
            end
        end
        
        % Clear videoWriter variable
        videoWriter = [];
        
        % Print completion message
        fprintf('=== Visualization complete. Training continues... ===\n');
    end

    %% Collect metrics data for visualization
    history.episodes(end+1) = ep;
    history.coverage(end+1) = coverage * 100;  % 转换为百分比
    history.utility(end+1) = U;
    history.weights(:, end+1) = w_obj;
    
    % 更新碰撞计数和碰撞率
    if collision
        collisionCount = collisionCount + 1;
    end
    collisionRate = (collisionCount / ep) * 100;  % 碰撞率百分比
    history.collisions(end+1) = collisionRate;
    
    if mod(ep, 10) == 0
        fprintf('Ep %3d: steps=%3d, cov=%.2f, coll=%d, U=%.3f | w=[%.2f %.2f %.2f %.2f]\n', ...
            ep, steps, coverage, collision, U, w_obj(1), w_obj(2), w_obj(3), w_obj(4));
        
        % Update visualization graphs
        if isvalid(vizFig)
            % 应用强平滑滤波 - 使用更大的窗口和指数加权移动平均
            n = length(history.episodes);
            
            % 使用自适应窗口大小（最多50个episode的移动平均）
            windowSize = min(50, max(10, floor(n * 0.1)));
            
            % 先应用移动平均
            if n >= windowSize
                smoothCoverage = movmean(history.coverage, windowSize);
                smoothUtility = movmean(history.utility, windowSize);
                smoothWeights = movmean(history.weights, windowSize, 2);
                smoothCollisions = movmean(history.collisions, windowSize);
            else
                smoothCoverage = history.coverage;
                smoothUtility = history.utility;
                smoothWeights = history.weights;
                smoothCollisions = history.collisions;
            end
            
            % 再应用指数加权移动平均（EWMA）进一步平滑
            alpha = 0.15;  % 平滑系数，越小越平滑
            if n > 1
                smoothCoverage = filter(alpha, [1 alpha-1], smoothCoverage);
                smoothUtility = filter(alpha, [1 alpha-1], smoothUtility);
                for i = 1:4
                    smoothWeights(i,:) = filter(alpha, [1 alpha-1], smoothWeights(i,:));
                end
                smoothCollisions = filter(alpha, [1 alpha-1], smoothCollisions);
            end
            
            % Update coverage curve (smoothed)
            set(hCoverage, 'XData', history.episodes, 'YData', smoothCoverage);
            
            % Update utility value curve (smoothed)
            set(hUtility, 'XData', history.episodes, 'YData', smoothUtility);
            
            % Update weight curves (redraw with 4 separate smoothed lines)
            subplot(2, 2, 3); cla; hold on;
            plot(history.episodes, smoothWeights(1,:), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Time');
            plot(history.episodes, smoothWeights(2,:), 'g-', 'LineWidth', 1.5, 'DisplayName', 'Coverage');
            plot(history.episodes, smoothWeights(3,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Sonar');
            plot(history.episodes, smoothWeights(4,:), 'k-', 'LineWidth', 1.5, 'DisplayName', 'Collision');
            xlabel('Episode'); ylabel('Weight Value'); title('Objective Weights');
            legend('Location', 'best'); grid on; ylim([0, 1]);
            
            % Update collision rate curve (smoothed)
            set(hCollision, 'XData', history.episodes, 'YData', smoothCollisions);
            
            drawnow;  % Force refresh display
        end
    end
end

%% --------- Helper functions --------- %%

function p = softmaxLocal(z)
    z = z - max(z);
    ez = exp(z);
    p = ez / sum(ez);
end

function a = sampleAction(p)
    r = rand();
    cum = cumsum(p);
    a = find(r <= cum, 1, 'first');
    if isempty(a)
        a = numel(p);
    end
end

function U = computeUtility(coverage, collision, steps, sonarSteps, maxSteps)
    covTerm   = coverage;
    collTerm  = double(collision ~= 0);
    pathTerm  = steps / maxSteps;
    sonarTerm = sonarSteps / maxSteps;
    U = 1.5 * covTerm - 0.4 * collTerm - 0.05 * pathTerm - 0.05 * sonarTerm;
end

function [patchTensor, globalVec] = splitObservation(stateVec, patchSize, globalDim)
    stateVec = stateVec(:);
    patchArea = patchSize * patchSize;
    patchTensor = reshape(single(stateVec(1:patchArea)), patchSize, patchSize, 1);
    globalVec = reshape(single(stateVec(patchArea+1:patchArea+globalDim)), globalDim, 1);
end

function net = createUUVPolicyValueNet(patchSize, globalDim, nActions, kObj, sharedDim)
    % Multi-Head Critic Architecture
    % Creates a network with shared feature extraction (CNN + MLP) and multiple heads:
    % - 1 Actor head for action logits
    % - 4 independent Critic heads (one per objective)
    
    patchBranch = [
        imageInputLayer([patchSize patchSize 1], 'Normalization', 'none', 'Name', 'patchInput')
        convolution2dLayer(3, 8, 'Padding', 0, 'WeightsInitializer', 'he', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
        convolution2dLayer(3, 16, 'Padding', 0, 'WeightsInitializer', 'he', 'Name', 'conv2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool')
        flattenLayer('Name', 'flatten')
    ];

    stateBranch = [
        featureInputLayer(globalDim, 'Normalization', 'none', 'Name', 'stateInput')
        fullyConnectedLayer(16, 'Name', 'state_fc', 'WeightsInitializer', 'he')
        reluLayer('Name', 'state_relu')
    ];

    sharedLayers = [
        concatenationLayer(1, 2, 'Name', 'concat')
        fullyConnectedLayer(sharedDim, 'Name', 'shared_fc', 'WeightsInitializer', 'he')
        reluLayer('Name', 'shared_relu')
    ];

    % Actor head (unchanged)
    actorHead = fullyConnectedLayer(nActions, 'Name', 'actor_fc');
    
    % Multi-head Critic: 4 independent heads, one per objective
    criticHead1 = fullyConnectedLayer(1, 'Name', 'critic_fc_1');
    criticHead2 = fullyConnectedLayer(1, 'Name', 'critic_fc_2');
    criticHead3 = fullyConnectedLayer(1, 'Name', 'critic_fc_3');
    criticHead4 = fullyConnectedLayer(1, 'Name', 'critic_fc_4');

    lgraph = layerGraph(patchBranch);
    lgraph = addLayers(lgraph, stateBranch);
    lgraph = addLayers(lgraph, sharedLayers);
    lgraph = addLayers(lgraph, actorHead);
    lgraph = addLayers(lgraph, criticHead1);
    lgraph = addLayers(lgraph, criticHead2);
    lgraph = addLayers(lgraph, criticHead3);
    lgraph = addLayers(lgraph, criticHead4);
    
    lgraph = connectLayers(lgraph, 'state_relu', 'concat/in2');
    lgraph = connectLayers(lgraph, 'flatten', 'concat/in1');
    lgraph = connectLayers(lgraph, 'shared_relu', 'actor_fc');
    lgraph = connectLayers(lgraph, 'shared_relu', 'critic_fc_1');
    lgraph = connectLayers(lgraph, 'shared_relu', 'critic_fc_2');
    lgraph = connectLayers(lgraph, 'shared_relu', 'critic_fc_3');
    lgraph = connectLayers(lgraph, 'shared_relu', 'critic_fc_4');

    net = dlnetwork(lgraph);
end

function adamState = initializeAdamState(dlnet)
    learnables = dlnet.Learnables;
    adamState = learnables;
    for i = 1:size(learnables, 1)
        adamState.Value{i} = zeros(size(learnables.Value{i}), 'like', learnables.Value{i});
    end
end

function [dlActor, dlCriticMulti] = forwardMultiHeadCritic(dlnet, dlPatch, dlGlobal)
    % Forward pass for multi-head critic network
    % Returns:
    %   dlActor: [nActions, batchSize] - Actor logits
    %   dlCriticMulti: [kObj, batchSize] - Concatenated critic values from 4 heads
    
    % Perform forward pass - predict returns outputs for specified layers
    [dlActor, dlCritic1, dlCritic2, dlCritic3, dlCritic4] = predict(dlnet, dlPatch, dlGlobal, ...
        'Outputs', {'actor_fc', 'critic_fc_1', 'critic_fc_2', 'critic_fc_3', 'critic_fc_4'});
    
    % Concatenate critic heads: [4, batchSize]
    dlCriticMulti = cat(1, dlCritic1, dlCritic2, dlCritic3, dlCritic4);
end

function w_new = updateAdaptiveWeights(G_hist, U_hist, windowSize)
    % Adaptive weight update using Pearson correlation coefficient (Paper Eq. 28)
    % Inputs:
    %   G_hist: [kObj, N] - Historical cumulative returns for each objective
    %   U_hist: [1, N] - Historical utility values
    %   windowSize: Sliding window size (e.g., 128)
    % Returns:
    %   w_new: [kObj, 1] - New weight vector (normalized via Softmax)
    
    [kObj, N] = size(G_hist);
    N_use = min(N, windowSize);
    
    % Extract recent window
    G_window = G_hist(:, end-N_use+1:end);  % [kObj, N_use]
    U_window = U_hist(:, end-N_use+1:end);  % [1, N_use]
    
    % Initialize correlation coefficients
    r = zeros(kObj, 1);
    
    % Compute Pearson correlation coefficient for each objective
    for i = 1:kObj
        G_i = G_window(i, :);  % [1, N_use]
        
        % Compute means
        G_i_mean = mean(G_i);
        U_mean = mean(U_window);
        
        % Pearson correlation: r_i = sum((G_i - mean(G_i))(U - mean(U))) / (std(G_i) * std(U))
        numerator = sum((G_i - G_i_mean) .* (U_window - U_mean));
        denominator = sqrt(sum((G_i - G_i_mean).^2)) * sqrt(sum((U_window - U_mean).^2));
        
        % Avoid division by zero
        if denominator < 1e-8
            r(i) = 0;
        else
            r(i) = numerator / denominator;
        end
    end
    
    % Apply Softmax to get normalized weights: c_i = exp(r_i) / sum(exp(r_k))
    r_exp = exp(r - max(r));  % Numerical stability: subtract max
    w_new = r_exp / sum(r_exp);
end

function [loss, gradients] = ppoLoss(dlnet, dlPatch, dlGlobal, actions, oldLogProbs, advantages, returns, epsilonClip, valueLossCoef)
    % PPO Loss with Multi-Head Critic
    % Key: Critic loss is NOT weighted by adaptive weights
    % Weights are only used for Actor's advantage function (A_scalar)
    
    [dlActor, dlCriticMulti] = forwardMultiHeadCritic(dlnet, dlPatch, dlGlobal);
    % dlCriticMulti: [kObj, batchSize]

    % ========== Actor Loss (unchanged) ==========
    maxLogits = max(dlActor, [], 1);
    stabilized = dlActor - maxLogits;
    logSumExp = log(sum(exp(stabilized), 1)) + maxLogits;
    logSoftmax = dlActor - logSumExp;

    actions = reshape(double(actions), 1, []);
    cols = 1:numel(actions);
    idx = sub2ind(size(logSoftmax), actions, cols);
    logPiA = logSoftmax(idx);

    oldLog = dlarray(single(reshape(oldLogProbs, size(logPiA))));
    adv = dlarray(single(reshape(advantages, size(logPiA))));

    ratio = exp(logPiA - oldLog);
    ratioClipped = min(max(ratio, 1 - epsilonClip), 1 + epsilonClip);
    surrogate1 = ratio .* adv;
    surrogate2 = ratioClipped .* adv;
    actorLoss = -mean(min(surrogate1, surrogate2));

    % ========== Critic Loss (multi-head, NOT weighted) ==========
    % Each critic head learns its objective's value function independently
    dlReturns = dlarray(single(returns));  % [kObj, batchSize]
    criticLoss = mean((dlCriticMulti - dlReturns).^2, 'all');

    % Total loss
    loss = actorLoss + valueLossCoef * criticLoss;
    gradients = dlgradient(loss, dlnet.Learnables);
end

function vWriter = initializeVideoRecording(filename, frameRate)
    % Initialize video recording with MPEG-4 format
    % Inputs:
    %   filename: Output video filename (e.g., 'UUV_episodes_4690_4700.mp4')
    %   frameRate: Frames per second (e.g., 10)
    % Returns:
    %   vWriter: VideoWriter object or empty array on failure
    
    % Initialize to empty
    vWriter = [];
    
    % Validate inputs
    if isempty(filename) || ~ischar(filename)
        warning('Invalid filename provided for video recording. Expected non-empty string.');
        return;
    end
    
    if isempty(frameRate) || ~isnumeric(frameRate) || frameRate <= 0
        warning('Invalid frame rate provided for video recording. Expected positive number.');
        return;
    end
    
    try
        vWriter = VideoWriter(filename, 'MPEG-4');
        vWriter.FrameRate = frameRate;
        vWriter.Quality = 95;
        open(vWriter);
        
        % Verify the video writer is valid after opening
        if ~isvalid(vWriter)
            warning('VideoWriter object is invalid after initialization.');
            vWriter = [];
        end
    catch ME
        warning('Failed to initialize video recording: %s', getReport(ME, 'basic'));
        vWriter = [];
    end
end

function captureFrame(vWriter, env, trajectory, metrics, showVortex)
    % Capture and render current environment state as video frame
    % Inputs:
    %   vWriter: VideoWriter object
    %   env: UUVEnvMO environment instance
    %   trajectory: Nx2 matrix of [x, y] positions
    %   metrics: Structure with episode, timestep, coverage, rewards, action
    %   showVortex: Boolean flag to enable vortex visualization
    
    % Validate videoWriter before proceeding
    if isempty(vWriter)
        warning('VideoWriter is empty. Skipping frame capture.');
        return;
    end
    
    if ~isvalid(vWriter)
        warning('VideoWriter is not valid. Skipping frame capture.');
        return;
    end
    
    % Validate environment object
    if isempty(env) || ~isobject(env)
        warning('Invalid environment object. Skipping frame capture.');
        return;
    end
    
    % Validate metrics structure
    if isempty(metrics) || ~isstruct(metrics)
        warning('Invalid metrics structure. Skipping frame capture.');
        return;
    end
    
    try
        % Create invisible figure with specified dimensions
        fig = figure('Visible', 'off', 'Position', [100, 100, 800, 900]);
        
        % Create main axes for map visualization
        ax = axes('Parent', fig, 'Position', [0.1, 0.25, 0.8, 0.65]);
        hold(ax, 'on');
        
        % Render 50x50 grid map with color coding
        mapSize = env.mapSize;
        mapDisplay = zeros(mapSize(1), mapSize(2), 3); % RGB image
        
        for i = 1:mapSize(1)
            for j = 1:mapSize(2)
                if env.grid(i, j) == 1
                    % Obstacle: black
                    mapDisplay(i, j, :) = [0, 0, 0];
                elseif env.grid(i, j) == 0
                    % Explored free space: white
                    mapDisplay(i, j, :) = [1, 1, 1];
                else
                    % Unexplored: gray
                    mapDisplay(i, j, :) = [0.5, 0.5, 0.5];
                end
            end
        end
        
        % Display map (no flip - grid(i,j) where i=row=Y, j=col=X)
        % Use 'reverse' YDir to match MATLAB image convention
        image(ax, 'XData', [0, env.width], 'YData', [0, env.height], ...
              'CData', mapDisplay);
        axis(ax, 'equal');
        xlim(ax, [0, env.width]);
        ylim(ax, [0, env.height]);
        set(ax, 'YDir', 'reverse');
        
        % Render vortex visualization (before trajectory and UUV)
        renderVortexVisualization(ax, env, showVortex);
        
        % Draw trajectory path as connected blue line
        if ~isempty(trajectory) && size(trajectory, 1) > 1
            plot(ax, trajectory(:, 1), trajectory(:, 2), 'b-', 'LineWidth', 2);
        end
        
        % Draw UUV position as red triangle with orientation arrow
        uuvX = env.pos(1);
        uuvY = env.pos(2);
        uuvYaw = env.yaw;
        
        % Triangle vertices (pointing in direction of yaw)
        triangleSize = 15;
        vertices = [
            triangleSize, 0;
            -triangleSize/2, triangleSize/2;
            -triangleSize/2, -triangleSize/2
        ];
        
        % Rotate vertices by yaw angle
        R = [cos(uuvYaw), -sin(uuvYaw); sin(uuvYaw), cos(uuvYaw)];
        rotatedVertices = (R * vertices')';
        triangleX = rotatedVertices(:, 1) + uuvX;
        triangleY = rotatedVertices(:, 2) + uuvY;
        
        % Draw filled triangle
        fill(ax, triangleX, triangleY, 'r', 'EdgeColor', 'k', 'LineWidth', 1.5);
        
        % Draw orientation arrow
        arrowLength = 30;
        arrowX = [uuvX, uuvX + arrowLength * cos(uuvYaw)];
        arrowY = [uuvY, uuvY + arrowLength * sin(uuvYaw)];
        plot(ax, arrowX, arrowY, 'r-', 'LineWidth', 2);
        
        % Draw sonar FOV cone when action is 4 (sonar)
        if metrics.action == 4
            sonarRange = 8 * env.cellSize; % 8 cells
            sonarFOV = pi/3; % 60 degrees
            
            % Create arc for sonar cone
            angles = linspace(uuvYaw - sonarFOV, uuvYaw + sonarFOV, 30);
            sonarX = [uuvX, uuvX + sonarRange * cos(angles), uuvX];
            sonarY = [uuvY, uuvY + sonarRange * sin(angles), uuvY];
            
            % Draw semi-transparent green cone
            fill(ax, sonarX, sonarY, 'g', 'FaceAlpha', 0.3, 'EdgeColor', 'g', 'LineWidth', 1.5);
        end
        
        % Add title with episode and timestep
        title(ax, sprintf('Episode %d | Step %d | Coverage: %.1f%%', ...
              metrics.episode, metrics.timestep, metrics.coverage * 100), ...
              'FontSize', 14, 'FontWeight', 'bold');
        
        % Add text overlay with cumulative rewards for each objective
        rewardText = sprintf(['Time: %.3f  |  Explore: %.3f\n' ...
                             'Sonar: %.3f  |  Collision: %.3f'], ...
                             sum(metrics.rewards(1, :)), ...
                             sum(metrics.rewards(2, :)), ...
                             sum(metrics.rewards(3, :)), ...
                             sum(metrics.rewards(4, :)));
        
        % Create text box at bottom of figure
        annotation(fig, 'textbox', [0.1, 0.05, 0.8, 0.15], ...
                   'String', rewardText, ...
                   'FontSize', 12, ...
                   'HorizontalAlignment', 'center', ...
                   'VerticalAlignment', 'middle', ...
                   'EdgeColor', 'black', ...
                   'LineWidth', 1.5, ...
                   'BackgroundColor', 'white');
        
        % Capture frame using getframe()
        frame = getframe(fig);
        
        % Validate frame before writing
        if isempty(frame) || ~isfield(frame, 'cdata')
            warning('Invalid frame captured. Skipping frame write.');
            close(fig);
            return;
        end
        
        % Write frame to video using writeVideo()
        writeVideo(vWriter, frame);
        
        % Close figure to free memory
        close(fig);
        
    catch ME
        warning('Frame capture failed: %s', getReport(ME, 'basic'));
        % Close figure if it exists
        if exist('fig', 'var') && ishandle(fig)
            try
                close(fig);
            catch
                % Silently fail if figure cannot be closed
            end
        end
    end
end

function renderTransitionFrame(vWriter, episodeSummary)
    % Render episode transition frame with summary statistics
    % Inputs:
    %   vWriter: VideoWriter object
    %   episodeSummary: Structure with episode, coverage, steps, collision, utility
    
    % Validate videoWriter before proceeding
    if isempty(vWriter)
        warning('VideoWriter is empty. Skipping transition frame.');
        return;
    end
    
    if ~isvalid(vWriter)
        warning('VideoWriter is not valid. Skipping transition frame.');
        return;
    end
    
    % Validate episodeSummary structure
    if isempty(episodeSummary) || ~isstruct(episodeSummary)
        warning('Invalid episode summary structure. Skipping transition frame.');
        return;
    end
    
    % Validate required fields in episodeSummary
    requiredFields = {'episode', 'coverage', 'steps', 'collision', 'utility'};
    for i = 1:length(requiredFields)
        if ~isfield(episodeSummary, requiredFields{i})
            warning('Missing required field "%s" in episode summary. Skipping transition frame.', requiredFields{i});
            return;
        end
    end
    
    try
        % Create text-only figure with episode summary
        fig = figure('Visible', 'off', 'Position', [100, 100, 800, 900]);
        
        % Create axes for text display
        ax = axes('Parent', fig, 'Position', [0.1, 0.1, 0.8, 0.8]);
        axis(ax, 'off');
        
        % Display episode number
        text(ax, 0.5, 0.75, sprintf('EPISODE %d COMPLETE', episodeSummary.episode), ...
             'FontSize', 24, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        
        % Display coverage
        text(ax, 0.5, 0.60, sprintf('Coverage: %.1f%%', episodeSummary.coverage * 100), ...
             'FontSize', 18, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        
        % Display steps
        text(ax, 0.5, 0.50, sprintf('Steps: %d', episodeSummary.steps), ...
             'FontSize', 18, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        
        % Display collision status
        if episodeSummary.collision
            statusText = 'Status: Collision';
            statusColor = [0.8, 0, 0]; % Red
        else
            statusText = 'Status: Goal Reached';
            statusColor = [0, 0.6, 0]; % Green
        end
        text(ax, 0.5, 0.40, statusText, ...
             'FontSize', 18, 'Color', statusColor, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        
        % Display utility
        text(ax, 0.5, 0.30, sprintf('Utility: %.3f', episodeSummary.utility), ...
             'FontSize', 18, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        
        % Display next episode message
        text(ax, 0.5, 0.15, '(Next episode starting...)', ...
             'FontSize', 14, 'FontStyle', 'italic', 'Color', [0.5, 0.5, 0.5], ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        
        % Capture frame as image
        frame = getframe(fig);
        
        % Validate frame before writing
        if isempty(frame) || ~isfield(frame, 'cdata')
            warning('Invalid transition frame captured. Skipping transition.');
            close(fig);
            return;
        end
        
        % Write 10 identical frames to video (1 second at 10 fps)
        for i = 1:10
            try
                writeVideo(vWriter, frame);
            catch ME
                warning('Failed to write transition frame %d/10: %s', i, getReport(ME, 'basic'));
                break; % Stop writing frames if one fails
            end
        end
        
        % Close figure
        close(fig);
        
    catch ME
        warning('Transition frame rendering failed: %s', getReport(ME, 'basic'));
        % Close figure if it exists
        if exist('fig', 'var') && ishandle(fig)
            try
                close(fig);
            catch
                % Silently fail if figure cannot be closed
            end
        end
    end
end

function finalizeVideoRecording(vWriter, filename)
    % Finalize video recording and close video file
    % Inputs:
    %   vWriter: VideoWriter object
    %   filename: Video filename for confirmation message
    
    % Validate videoWriter before proceeding
    if isempty(vWriter)
        warning('VideoWriter is empty. Nothing to finalize.');
        return;
    end
    
    if ~isvalid(vWriter)
        warning('VideoWriter is not valid. Cannot finalize video.');
        return;
    end
    
    try
        % Close VideoWriter object
        close(vWriter);
        
        % Verify file was created
        videoPath = fullfile(pwd, filename);
        if exist(videoPath, 'file')
            % Print confirmation message with file location
            fprintf('=== Video recording complete: %s ===\n', filename);
            fprintf('Video saved to: %s\n', videoPath);
            
            % Display file size for verification
            fileInfo = dir(videoPath);
            fileSizeMB = fileInfo.bytes / (1024 * 1024);
            fprintf('Video file size: %.2f MB\n', fileSizeMB);
        else
            warning('Video file was not created at expected location: %s', videoPath);
        end
    catch ME
        % Handle errors gracefully with try-catch
        warning('Failed to finalize video: %s', getReport(ME, 'basic'));
    end
end

function [X, Y, U, V] = sampleCurrentField(env, numSamplesX, numSamplesY)
    % Sample the ocean current field at a grid of points
    % Inputs:
    %   env: UUVEnvMO environment instance
    %   numSamplesX: Number of sample points in X direction
    %   numSamplesY: Number of sample points in Y direction
    % Returns:
    %   X, Y: Meshgrid arrays of sample point coordinates
    %   U, V: Velocity components at each sample point
    
    % Generate uniform grid of sample points
    x = linspace(0, env.width, numSamplesX);
    y = linspace(0, env.height, numSamplesY);
    [X, Y] = meshgrid(x, y);
    
    % Evaluate current field at each point
    U = zeros(size(X));
    V = zeros(size(Y));
    for i = 1:numel(X)
        [U(i), V(i)] = env.currentField(X(i), Y(i));
    end
end

function renderVortexVisualization(ax, env, showVortex)
    % Render ocean current vortex visualization
    % Inputs:
    %   ax: Axes handle for plotting
    %   env: UUVEnvMO environment instance
    %   showVortex: Boolean flag to enable/disable visualization
    
    if ~showVortex
        return;
    end
    
    % Check if environment has current field parameters
    if isempty(env.currentCenter) || isempty(env.currentSigma) || isempty(env.currentStrength)
        warning('Ocean current parameters not initialized. Skipping vortex visualization.');
        return;
    end
    
    % Sample current field
    numSamplesX = 12;
    numSamplesY = 12;
    [X, Y, U, V] = sampleCurrentField(env, numSamplesX, numSamplesY);
    
    % Calculate arrow scaling
    maxVelocity = max(sqrt(U(:).^2 + V(:).^2));
    if maxVelocity > 0
        scaleFactor = 50 / maxVelocity;  % Scale so max arrow is ~50m
    else
        scaleFactor = 1;
    end
    
    % Render velocity vectors as arrows
    quiver(ax, X, Y, U * scaleFactor, V * scaleFactor, ...
           'Color', [0.2, 0.6, 0.8], ...  % Light blue
           'LineWidth', 1.2, ...
           'MaxHeadSize', 0.5, ...
           'AutoScale', 'off');
    
    % Mark vortex center
    plot(ax, env.currentCenter(1), env.currentCenter(2), ...
         'o', 'MarkerSize', 10, ...
         'MarkerFaceColor', [0.2, 0.6, 0.8], ...
         'MarkerEdgeColor', 'k', ...
         'LineWidth', 2);
end
