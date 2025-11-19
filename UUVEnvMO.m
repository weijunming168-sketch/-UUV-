classdef UUVEnvMO < handle
    % 多目标 UUV 环境（简化版）
    % - 2D 连续空间 + 栅格地图
    % - 矩形障碍 / 扇形声呐 / 简单洋流
    % - 奖励向量：[时间, 探索, 声呐, 碰撞/完成]

    properties
        % 地图与网格参数
        mapSize            % [Ny, Nx]
        cellSize           % 单个网格边长 (m)
        width              % 地图宽度 (m)
        height             % 地图高度 (m)

        trueMap            % 真实地图：0=可行 1=障碍
        grid               % 已感知地图：-1=未知 0=可行 1=障碍
        freeCellsTotal     % 自由网格总数
        knownFreeCount     % 已知自由网格数

        % 运动状态
        pos                % [x;y] 连续坐标 (m)
        yaw                % 航向角 (rad)
        speed              % 前向速度 (m/s)
        dt                 % 时间步长 (s)
        deltaYaw           % 单步转向角 (rad)

        maxSteps           % Episode 最大步数
        stepCount          % 当前步数
        sonarSteps         % 声呐开启步数

        targetCoverage         % 终止覆盖率阈值
        quickCoverageThreshold % 较低覆盖率也可提前结束

        % 洋流参数（简单涡旋）
        currentCenter
        currentSigma
        currentStrength

        % 观测
        patchRadius
        doneFlag

        % 障碍生成配置
        obstacleCountRange
        obstacleSizeRange
        startSafeRadius
        obstacleBuffer
        
        % 安全区域属性
        safeZoneRows       % [minRow, maxRow] 安全区域行索引
        safeZoneCols       % [minCol, maxCol] 安全区域列索引
        safeZoneCenter     % [centerRow, centerCol] 安全区域中心
    end

    methods
        function obj = UUVEnvMO()
            % 构造函数：初始化参数
            obj.mapSize = [50, 50];
            obj.cellSize = 20;
            obj.height = obj.mapSize(1) * obj.cellSize;
            obj.width  = obj.mapSize(2) * obj.cellSize;

            obj.speed = 2;
            obj.dt = 1.0;
            obj.deltaYaw = pi/12;

            obj.maxSteps = 2048;
            obj.targetCoverage = 1;
            obj.quickCoverageThreshold = 0.95;

            obj.currentCenter = [obj.width/2; obj.height/2];
            obj.currentSigma = obj.width/2;
            obj.currentStrength = 5e-4;

            obj.patchRadius = 7;   % 15x15 patch

            obj.trueMap = zeros(obj.mapSize);
            obj.grid = -1 * ones(obj.mapSize);
            obj.freeCellsTotal = 0;
            obj.knownFreeCount = 0;

            obj.obstacleCountRange = [1, 2];
            obj.obstacleSizeRange = [3, 7];
            obj.startSafeRadius = 5;
            obj.obstacleBuffer = 3;

            % 使用新的初始化系统
            obj.initializeEnvironment();
            
            obj.stepCount = 0;
            obj.sonarSteps = 0;
            obj.doneFlag = false;
        end

        function d = getObservationDim(obj)
            patchSize = 2 * obj.patchRadius + 1;
            d = patchSize * patchSize + 6; % patch 展平 + 六维全局特征
        end

        function state = reset(obj)
            % 使用新的初始化系统重置环境
            obj.initializeEnvironment();
            
            % 重置计数器
            obj.freeCellsTotal = nnz(obj.trueMap == 0);
            obj.knownFreeCount = 0;
            obj.stepCount = 0;
            obj.sonarSteps = 0;
            obj.doneFlag = false;

            % 初始化已知地图
            [i, j] = obj.posToIdx(obj.pos(1), obj.pos(2));
            obj.grid(i, j) = obj.trueMap(i, j);
            if obj.trueMap(i, j) == 0
                obj.knownFreeCount = 1;
            end

            state = obj.getObservation();
        end

        function [nextState, rewardVec, done, info] = step(obj, action)
            if obj.doneFlag
                error('Environment is done. Call reset() before step().');
            end

            obj.stepCount = obj.stepCount + 1;
            sonarActive = false;
            switch action
                case 1
                    dpsi = -obj.deltaYaw;
                case 2
                    dpsi = 0;
                case 3
                    dpsi = obj.deltaYaw;
                case 4
                    dpsi = 0;
                    sonarActive = true;
                otherwise
                    error('Invalid action %d', action);
            end

            obj.yaw = UUVEnvMO.wrapToPiLocal(obj.yaw + dpsi);
            [vxC, vyC] = obj.currentField(obj.pos(1), obj.pos(2));
            vx = obj.speed * cos(obj.yaw) + vxC;
            vy = obj.speed * sin(obj.yaw) + vyC;
            obj.pos(1) = obj.pos(1) + vx * obj.dt;
            obj.pos(2) = obj.pos(2) + vy * obj.dt;

            done = false;
            collision = false;
            reachedGoal = false;

            if obj.pos(1) < 0 || obj.pos(1) >= obj.width || obj.pos(2) < 0 || obj.pos(2) >= obj.height
                done = true;
                collision = true;
            else
                [i, j] = obj.posToIdx(obj.pos(1), obj.pos(2));
                if obj.trueMap(i, j) == 1
                    done = true;
                    collision = true;
                end
            end

            prevKnownFree = obj.knownFreeCount;
            newFreeCells = 0;
            if sonarActive && ~done
                newly = obj.applySonar();
                obj.sonarSteps = obj.sonarSteps + 1;
                newFreeCells = newly;
            end

            obj.knownFreeCount = nnz(obj.grid == 0);
            coverage = 0;
            if obj.freeCellsTotal > 0
                coverage = obj.knownFreeCount / obj.freeCellsTotal;
            end
            coverageGain = max(obj.knownFreeCount - prevKnownFree, 0);

            if ~done
                if coverage >= obj.targetCoverage
                    done = true;
                    reachedGoal = true;
                elseif coverage >= obj.quickCoverageThreshold
                    done = true;
                elseif obj.stepCount >= obj.maxSteps
                    done = true;
                    reachedGoal = coverage >= obj.targetCoverage;
                end
                collision = false;
            end

            r_time = -0.001;
            if obj.freeCellsTotal > 0
                r_explore = 10 * coverageGain / obj.freeCellsTotal;
            else
                r_explore = 0;
            end

            if sonarActive
                sonarGain = newFreeCells / max(1, obj.freeCellsTotal);
                r_sonar = 0.02 * sonarGain + 0.001;
            else
                r_sonar = 0;
            end

            if collision
                r_collision = -0.5;
            elseif done && reachedGoal
                r_collision = 1.0;
            elseif done && coverage >= obj.quickCoverageThreshold
                r_collision = 0.3;
            else
                r_collision = 0;
            end

            rewardVec = [r_time; r_explore; r_sonar; r_collision];
            nextState = obj.getObservation();

            info.coverage   = coverage;
            info.collision  = collision;
            info.reachedGoal = reachedGoal;
            info.stepCount  = obj.stepCount;
            info.sonarSteps = obj.sonarSteps;

            if done
                obj.doneFlag = true;
            end
        end

        function obs = getObservation(obj)
            % 观测 = 局部 15x15 patch + 6 维全局特征
            r = obj.patchRadius;
            [ci, cj] = obj.posToIdx(obj.pos(1), obj.pos(2));
            Ny = obj.mapSize(1);
            Nx = obj.mapSize(2);
            patchSize = 2 * r + 1;
            patch = 0.5 * ones(patchSize, patchSize);

            for di = -r:r
                for dj = -r:r
                    ii = ci + di;
                    jj = cj + dj;
                    if ii >= 1 && ii <= Ny && jj >= 1 && jj <= Nx
                        val = obj.grid(ii, jj);
                        if val == -1
                            patch(di + r + 1, dj + r + 1) = 0.5;
                        else
                            patch(di + r + 1, dj + r + 1) = val;
                        end
                    end
                end
            end

            patchVec = reshape(patch, [], 1);
            posNormX = obj.pos(1) / obj.width;
            posNormY = obj.pos(2) / obj.height;
            yawSin = sin(obj.yaw);
            yawCos = cos(obj.yaw);
            coverage = 0;
            if obj.freeCellsTotal > 0
                coverage = obj.knownFreeCount / obj.freeCellsTotal;
            end
            sonarFlag = 0;

            globalFeat = [posNormX; posNormY; yawSin; yawCos; coverage; sonarFlag];
            obs = [patchVec; globalFeat];
        end

        function [vx, vy] = currentField(obj, x, y)
            dx = x - obj.currentCenter(1);
            dy = y - obj.currentCenter(2);
            r2 = dx * dx + dy * dy;
            sigma2 = obj.currentSigma^2;
            factor = obj.currentStrength * exp(-r2 / (2 * sigma2));
            vx = -dy * factor;
            vy =  dx * factor;
        end

        function [i, j] = posToIdx(obj, x, y)
            j = floor(x / obj.cellSize) + 1;
            i = floor(y / obj.cellSize) + 1;
            j = max(1, min(obj.mapSize(2), j));
            i = max(1, min(obj.mapSize(1), i));
        end

        function [x, y] = idxToPos(obj, i, j)
            x = (j - 0.5) * obj.cellSize;
            y = (i - 0.5) * obj.cellSize;
        end

        function newCells = applySonar(obj)
            maxRangeCells = 8;
            thetaFOV = pi/3;
            newCells = 0;

            [Ny, Nx] = size(obj.grid);
            x0 = obj.pos(1);
            y0 = obj.pos(2);
            yaw = obj.yaw;

            for i = 1:Ny
                for j = 1:Nx
                    [xc, yc] = obj.idxToPos(i, j);
                    dx = xc - x0;
                    dy = yc - y0;
                    dist = sqrt(dx*dx + dy*dy);
                    if dist <= maxRangeCells * obj.cellSize && dist > 0
                        angle = UUVEnvMO.wrapToPiLocal(atan2(dy, dx) - yaw);
                        if abs(angle) <= thetaFOV && obj.grid(i, j) == -1
                            obj.grid(i, j) = obj.trueMap(i, j);
                            if obj.trueMap(i, j) == 0
                                newCells = newCells + 1;
                            end
                        end
                    end
                end
            end
        end
        
        function generateObstacles(obj)
            % 生成障碍物，避开安全区域
            Ny = obj.mapSize(1);
            Nx = obj.mapSize(2);
            
            countRange = obj.obstacleCountRange;
            sizeRange = obj.obstacleSizeRange;
            numObs = randi([countRange(1), countRange(2)]);
            
            ic = floor(Ny/2);
            jc = floor(Nx/2);
            safeRows = [max(1, ic - obj.startSafeRadius), min(Ny, ic + obj.startSafeRadius)];
            safeCols = [max(1, jc - obj.startSafeRadius), min(Nx, jc + obj.startSafeRadius)];
            
            for k = 1:numObs
                placed = false;
                attempts = 0;
                while ~placed && attempts < 50
                    attempts = attempts + 1;
                    w = randi([sizeRange(1), sizeRange(2)]);
                    h = randi([sizeRange(1), sizeRange(2)]);
                    w = min(w, Nx-1);
                    h = min(h, Ny-1);
                    i0 = randi([1, Ny - h]);
                    j0 = randi([1, Nx - w]);
                    i1 = i0;
                    i2 = i0 + h - 1;
                    j1 = j0;
                    j2 = j0 + w - 1;
                    
                    rowSafe = (i2 < safeRows(1) - obj.obstacleBuffer) || (i1 > safeRows(2) + obj.obstacleBuffer);
                    colSafe = (j2 < safeCols(1) - obj.obstacleBuffer) || (j1 > safeCols(2) + obj.obstacleBuffer);
                    if ~(rowSafe || colSafe)
                        continue;
                    end
                    
                    obj.trueMap(i1:i2, j1:j2) = 1;
                    placed = true;
                end
            end
        end
        
        function initializeOceanCurrent(obj)
            % 初始化洋流参数
            obj.currentCenter = [obj.width/2; obj.height/2];
            obj.currentSigma = obj.width/2;
            obj.currentStrength = 5e-4;
        end
        
        function clearSafeZone(obj)
            % 清空安全区域并存储边界
            Ny = obj.mapSize(1);
            Nx = obj.mapSize(2);
            
            ic = floor(Ny / 2);
            jc = floor(Nx / 2);
            obj.safeZoneRows = [max(1, ic - obj.startSafeRadius), min(Ny, ic + obj.startSafeRadius)];
            obj.safeZoneCols = [max(1, jc - obj.startSafeRadius), min(Nx, jc + obj.startSafeRadius)];
            obj.safeZoneCenter = [ic, jc];
            
            % 显式清空安全区域
            obj.trueMap(obj.safeZoneRows(1):obj.safeZoneRows(2), ...
                        obj.safeZoneCols(1):obj.safeZoneCols(2)) = 0;
            
            % 验证至少有一个自由网格
            freeCellsInSafeZone = nnz(obj.trueMap(obj.safeZoneRows(1):obj.safeZoneRows(2), ...
                                                  obj.safeZoneCols(1):obj.safeZoneCols(2)) == 0);
            if freeCellsInSafeZone == 0
                error('Safe zone contains no free cells after clearing');
            end
        end
        
        function generateSafeStartPosition(obj)
            % 在安全区域内生成随机起始位置
            validStart = false;
            maxAttempts = 100;
            attempt = 0;
            
            % 计算安全区域的连续坐标边界
            [minX, minY] = obj.idxToPos(obj.safeZoneRows(1), obj.safeZoneCols(1));
            [maxX, maxY] = obj.idxToPos(obj.safeZoneRows(2), obj.safeZoneCols(2));
            
            % 添加边界边距
            margin = 2 * obj.cellSize;
            minX = max(minX, margin);
            minY = max(minY, margin);
            maxX = min(maxX, obj.width - margin);
            maxY = min(maxY, obj.height - margin);
            
            % 尝试随机采样
            while ~validStart && attempt < maxAttempts
                attempt = attempt + 1;
                
                % 在安全区域内均匀随机采样
                randX = minX + rand() * (maxX - minX);
                randY = minY + rand() * (maxY - minY);
                
                % 转换为网格索引并验证
                [i, j] = obj.posToIdx(randX, randY);
                
                % 检查：在安全区域内 且 是自由网格
                if i >= obj.safeZoneRows(1) && i <= obj.safeZoneRows(2) && ...
                   j >= obj.safeZoneCols(1) && j <= obj.safeZoneCols(2) && ...
                   obj.trueMap(i, j) == 0
                    obj.pos = [randX; randY];
                    validStart = true;
                end
            end
            
            % 回退到安全区域中心
            if ~validStart
                [centerX, centerY] = obj.idxToPos(obj.safeZoneCenter(1), obj.safeZoneCenter(2));
                obj.pos = [centerX; centerY];
                
                % 验证回退位置是安全的
                [i, j] = obj.posToIdx(obj.pos(1), obj.pos(2));
                if obj.trueMap(i, j) ~= 0
                    error('Fallback position at safe zone center is not a free cell');
                end
                
                warning('Failed to find random safe start position after %d attempts. Using safe zone center.', maxAttempts);
            end
        end
        
        function validateStartState(obj)
            % 验证起始状态
            [i, j] = obj.posToIdx(obj.pos(1), obj.pos(2));
            
            % 验证UUV位置在自由网格
            if obj.trueMap(i, j) ~= 0
                error('UUV starting position (%d, %d) is not in a free cell', i, j);
            end
            
            % 验证位置在安全区域内
            if i < obj.safeZoneRows(1) || i > obj.safeZoneRows(2) || ...
               j < obj.safeZoneCols(1) || j > obj.safeZoneCols(2)
                error('UUV starting position (%d, %d) is outside safe zone', i, j);
            end
            
            % 验证边界边距
            margin = 2 * obj.cellSize;
            if obj.pos(1) < margin || obj.pos(1) > obj.width - margin || ...
               obj.pos(2) < margin || obj.pos(2) > obj.height - margin
                warning('UUV starting position violates boundary margin');
            end
        end
        
        function initializeEnvironment(obj)
            % 协调完整的初始化序列
            % 步骤1: 初始化地图结构
            obj.trueMap = zeros(obj.mapSize);
            obj.grid = -1 * ones(obj.mapSize);
            
            % 步骤2: 生成障碍物（尊重安全区域）
            obj.generateObstacles();
            
            % 步骤3: 清空并验证安全区域
            obj.clearSafeZone();
            
            % 步骤4: 初始化洋流参数
            obj.initializeOceanCurrent();
            
            % 步骤5: 生成随机安全起始位置
            obj.generateSafeStartPosition();
            
            % 步骤6: 设置随机方向
            obj.yaw = -pi + 2 * pi * rand();
            
            % 步骤7: 最终验证
            obj.validateStartState();
        end
    end
    
    methods (Static)
        function ang = wrapToPiLocal(ang)
            ang = mod(ang + pi, 2*pi) - pi;
        end
    end
end
