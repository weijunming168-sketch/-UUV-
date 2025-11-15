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
numEpisodes   = 5000;
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
