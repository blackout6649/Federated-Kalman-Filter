% Monte Carlo Simulation: Fault Detection vs Traditional Filtering
% Tests effectiveness of chi-square fault detection in federated KF
clear; clc; close all;
%% Simulation Parameters
numRuns = 10;              % Number of Monte Carlo runs
T = 100;                    % Simulation time
dt = 0.1;                   % Time step
N = round(T/dt);            % Number of time steps

% Fault scenarios to test
faultScenarios = [
    0.00, 0.00;    % No faults (baseline)
    0.02, 0.00;    % Light faults in sensor 1
    0.05, 0.02;    % Moderate faults in both sensors
    0.10, 0.05;    % Heavy faults
    0.15, 0.10;    % Very heavy faults
];
faultMagnitudes = 25:50:300; % Fault magnitudes to test

% Storage for results
results = struct();

%% Fixed Parameters (same for all runs)
sensorIntervals = [1, 1];
fusionInterval = 10;
weight = length(sensorIntervals);

% Motion model
F = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];
G = [0.5*dt^2*eye(2); dt*eye(2)];
q = 0.05;
Q = q*eye(2);
model = MotionModel(F,G,Q);

% Measurement model
H = [1 0 0 0; 0 1 0 0];
R1 = (5^2)*eye(2);
R2 = (8^2)*eye(2);

% Initial conditions
x0 = [0; 0; 1; 0.6]; 
P0 = diag([25 25 4 4]);

%% Monte Carlo Simulation
fprintf('Starting Monte Carlo simulation with %d runs...\n', numRuns);
for magIdx = 1:length(faultMagnitudes)
    currentFaultMagnitude = faultMagnitudes(magIdx);
    
    fprintf('\nTesting Fault Magnitude: %d\n', currentFaultMagnitude);
    
    for scenarioIdx = 1:size(faultScenarios, 1)
        faultProb1 = faultScenarios(scenarioIdx, 1);
        faultProb2 = faultScenarios(scenarioIdx, 2);
        
        fprintf('  Scenario %d: Fault probabilities [%.2f, %.2f]\n', ...
            scenarioIdx, faultProb1, faultProb2);
        
        % Storage for this scenario
        rmse_fkf_fd = zeros(numRuns, 2);      % FKF with fault detection
        rmse_fkf_nfd = zeros(numRuns, 2);     % FKF without fault detection
        rmse_ckf_nfd = zeros(numRuns, 2);     % CKF without fault detection
        
        detectionStats = struct();
        detectionStats.truePositives = zeros(numRuns, 2);
        detectionStats.falsePositives = zeros(numRuns, 2);
        detectionStats.trueNegatives = zeros(numRuns, 2);
        detectionStats.falseNegatives = zeros(numRuns, 2);
        
        for run = 1:numRuns
            if mod(run, 10) == 0
                fprintf('    Run %d/%d\n', run, numRuns);
            end
            
            % Set random seed for reproducibility
            rng(run + scenarioIdx * 1000 + magIdx * 10000);
            
            %% Create sensors with current fault parameters
            measDimension = 2;
            s1_fd = LinearSensor(H, R1, sensorIntervals(1), measDimension, "PosSensor-5m", faultProb1, currentFaultMagnitude);
            s2_fd = LinearSensor(H, R2, sensorIntervals(2), measDimension, "PosSensor-8m", faultProb2, currentFaultMagnitude);
            
            s1_nfd = LinearSensor(H, R1, sensorIntervals(1), measDimension, "PosSensor-5m", faultProb1, currentFaultMagnitude);
            s2_nfd = LinearSensor(H, R2, sensorIntervals(2), measDimension, "PosSensor-8m", faultProb2, currentFaultMagnitude);
            
            %% Create filter instances
            % FKF with fault detection
            lkf1_fd = LocalKalmanFilter(model, s1_fd, x0, P0, weight, "LKF1_FD");
            lkf2_fd = LocalKalmanFilter(model, s2_fd, x0, P0, weight, "LKF2_FD");
            fkf_fd = FederatedKF([lkf1_fd, lkf2_fd], weight, "FKF_FD", 0.05, 10);
            
            % FKF without fault detection (traditional)
            lkf1_nfd = LocalKalmanFilter(model, s1_nfd, x0, P0, weight, "LKF1_NFD");
            lkf2_nfd = LocalKalmanFilter(model, s2_nfd, x0, P0, weight, "LKF2_NFD");
            fkf_nfd = FederatedKFTraditional([lkf1_nfd, lkf2_nfd], weight, "FKF_NFD");
            
            % CKF without fault detection
            ckf_nfd = CentralizedKF(model, x0, P0, {s1_nfd, s2_nfd}, "CKF_NFD");
            
            %% Generate truth trajectory
            Xtrue = zeros(4, N);
            x_true = x0;
            for k = 1:N
                a = sqrt(q) * randn(2, 1);
                x_true = F * x_true + G * a;
                Xtrue(:, k) = x_true;
            end
            
            %% Storage for this run
            Xhat_fd = zeros(4, N);
            Xhat_nfd = zeros(4, N);
            Xhat_ckf = zeros(4, N);
            
            % Fault tracking
            actualFaults1 = zeros(1, N);
            actualFaults2 = zeros(1, N);
            detectedFaults1 = zeros(1, N);
            detectedFaults2 = zeros(1, N);
            
            %% Run simulation
            for k = 1:N
                % Generate measurements (same for all filters)
                z1_base = H * Xtrue(:, k) + mvnrnd(zeros(2, 1), R1)';
                z2_base = H * Xtrue(:, k) + mvnrnd(zeros(2, 1), R2)';
                
                % Add faults with specified probabilities
                z1 = z1_base; z2 = z2_base;
                if mod(k, sensorIntervals(1)) == 0 && rand < faultProb1
                    fault1 = currentFaultMagnitude * (2 * rand(2, 1) - 1);
                    z1 = z1 + fault1;
                    actualFaults1(k) = 1;
                end
                if mod(k, sensorIntervals(2)) == 0 && rand < faultProb2
                    fault2 = currentFaultMagnitude * (2 * rand(2, 1) - 1);
                    z2 = z2 + fault2;
                    actualFaults2(k) = 1;
                end
                
                % Handle NaN for non-measurement times
                if mod(k, sensorIntervals(1)) ~= 0, z1 = [NaN; NaN]; end
                if mod(k, sensorIntervals(2)) ~= 0, z2 = [NaN; NaN]; end
                
                fuseFlag = mod(k, fusionInterval) == 0;
                
                % FKF with fault detection
                fkf_fd.step({z1, z2}, fuseFlag);
                Xhat_fd(:, k) = fkf_fd.x;
                
                % Track fault detection
                if ~any(isnan(z1))
                    detectedFaults1(k) = 1 - fkf_fd.fault_flags(1);
                end
                if ~any(isnan(z2))
                    detectedFaults2(k) = 1 - fkf_fd.fault_flags(2);
                end
                
                % FKF without fault detection
                fkf_nfd.step({z1, z2}, fuseFlag);
                Xhat_nfd(:, k) = fkf_nfd.x;
                
                % CKF without fault detection
                ckf_nfd.step({z1, z2});
                Xhat_ckf(:, k) = ckf_nfd.x;
            end
            
            %% Calculate RMSEs
            rmse_fkf_fd(run, :) = sqrt(mean((Xhat_fd(1:2, :) - Xtrue(1:2, :)).^2, 2));
            rmse_fkf_nfd(run, :) = sqrt(mean((Xhat_nfd(1:2, :) - Xtrue(1:2, :)).^2, 2));
            rmse_ckf_nfd(run, :) = sqrt(mean((Xhat_ckf(1:2, :) - Xtrue(1:2, :)).^2, 2));
            
            %% Calculate detection statistics
            for sensorIdx = 1:2
                if sensorIdx == 1
                    actual = actualFaults1;
                    detected = detectedFaults1;
                else
                    actual = actualFaults2;
                    detected = detectedFaults2;
                end
                
                % Remove NaN measurement times
                validTimes = ~isnan(actual) & ~isnan(detected);
                actual = actual(validTimes);
                detected = detected(validTimes);
                
                if ~isempty(actual)
                    detectionStats.truePositives(run, sensorIdx) = sum(actual == 1 & detected == 1);
                    detectionStats.falsePositives(run, sensorIdx) = sum(actual == 0 & detected == 1);
                    detectionStats.trueNegatives(run, sensorIdx) = sum(actual == 0 & detected == 0);
                    detectionStats.falseNegatives(run, sensorIdx) = sum(actual == 1 & detected == 0);
                end
            end
        end
        
        %% Store results for this scenario
        scenarioKey = sprintf('mag%d_prob%d', currentFaultMagnitude, scenarioIdx);
        results.(scenarioKey).faultProbs = [faultProb1, faultProb2];
        results.(scenarioKey).faultMagnitude = currentFaultMagnitude;
        results.(scenarioKey).rmse_fkf_fd = rmse_fkf_fd;
        results.(scenarioKey).rmse_fkf_nfd = rmse_fkf_nfd;
        results.(scenarioKey).rmse_ckf_nfd = rmse_ckf_nfd;
        results.(scenarioKey).detectionStats = detectionStats;
        
        %% Calculate summary statistics
        results.(scenarioKey).mean_rmse_fkf_fd = mean(rmse_fkf_fd, 1);
        results.(scenarioKey).mean_rmse_fkf_nfd = mean(rmse_fkf_nfd, 1);
        results.(scenarioKey).mean_rmse_ckf_nfd = mean(rmse_ckf_nfd, 1);
        
        results.(scenarioKey).std_rmse_fkf_fd = std(rmse_fkf_fd, 1);
        results.(scenarioKey).std_rmse_fkf_nfd = std(rmse_fkf_nfd, 1);
        results.(scenarioKey).std_rmse_ckf_nfd = std(rmse_ckf_nfd, 1);
        
        % Detection performance metrics
        for sensorIdx = 1:2
            tp = detectionStats.truePositives(:, sensorIdx);
            fp = detectionStats.falsePositives(:, sensorIdx);
            tn = detectionStats.trueNegatives(:, sensorIdx);
            fn = detectionStats.falseNegatives(:, sensorIdx);
            
            precision = tp ./ (tp + fp + eps);
            recall = tp ./ (tp + fn + eps);
            specificity = tn ./ (tn + fp + eps);
            
            results.(scenarioKey).precision(sensorIdx) = mean(precision);
            results.(scenarioKey).recall(sensorIdx) = mean(recall);
            results.(scenarioKey).specificity(sensorIdx) = mean(specificity);
            results.(scenarioKey).f1_score(sensorIdx) = mean(2 * precision .* recall ./ (precision + recall + eps));
        end
    end
end

%% Display Results
fprintf('\n=== MONTE CARLO SIMULATION RESULTS ===\n');
fprintf('Number of runs per scenario: %d\n', numRuns);
fprintf('Simulation time: %.1f s, Time step: %.2f s\n', T, dt);
fieldNames = fieldnames(results);
for i = 1:length(fieldNames)
    scenarioKey = fieldNames{i};
    currentResults = results.(scenarioKey);
    
    fprintf('\n--- Fault Magnitude: %d, Fault Probabilities [%.2f, %.2f] ---\n', ...
        currentResults.faultMagnitude, currentResults.faultProbs(1), currentResults.faultProbs(2));
    
    % RMSE comparison
    fprintf('RMSE Results (mean ± std) [x, y]:\n');
    fprintf('  FKF w/ Fault Detection: [%.2f±%.2f, %.2f±%.2f] m\n', ...
        currentResults.mean_rmse_fkf_fd(1), currentResults.std_rmse_fkf_fd(1), ...
        currentResults.mean_rmse_fkf_fd(2), currentResults.std_rmse_fkf_fd(2));
    fprintf('  FKF Traditional:        [%.2f±%.2f, %.2f±%.2f] m\n', ...
        currentResults.mean_rmse_fkf_nfd(1), currentResults.std_rmse_fkf_nfd(1), ...
        currentResults.mean_rmse_fkf_nfd(2), currentResults.std_rmse_fkf_nfd(2));
    fprintf('  CKF Traditional:        [%.2f±%.2f, %.2f±%.2f] m\n', ...
        currentResults.mean_rmse_ckf_nfd(1), currentResults.std_rmse_ckf_nfd(1), ...
        currentResults.mean_rmse_ckf_nfd(2), currentResults.std_rmse_ckf_nfd(2));
    
    % Improvement percentages
    improvement_vs_fkf = (currentResults.mean_rmse_fkf_nfd - currentResults.mean_rmse_fkf_fd) ./ ...
                         currentResults.mean_rmse_fkf_nfd * 100;
    improvement_vs_ckf = (currentResults.mean_rmse_ckf_nfd - currentResults.mean_rmse_fkf_fd) ./ ...
                         currentResults.mean_rmse_ckf_nfd * 100;
    
    fprintf('  Improvement vs FKF Traditional: [%.1f%%, %.1f%%]\n', improvement_vs_fkf(1), improvement_vs_fkf(2));
    fprintf('  Improvement vs CKF Traditional: [%.1f%%, %.1f%%]\n', improvement_vs_ckf(1), improvement_vs_ckf(2));
    
    % Detection performance
    if currentResults.faultProbs(1) > 0 || currentResults.faultProbs(2) > 0
        fprintf('Fault Detection Performance:\n');
        for sensorIdx = 1:2
            fprintf('  Sensor %d: Precision=%.2f, Recall=%.2f, Specificity=%.2f, F1=%.2f\n', ...
                sensorIdx, currentResults.precision(sensorIdx), ...
                currentResults.recall(sensorIdx), currentResults.specificity(sensorIdx), ...
                currentResults.f1_score(sensorIdx));
        end
    end
end

%% Visualization (updated to handle multiple magnitudes)
createResultsPlots(results, faultMagnitudes);

%% Helper function for traditional FKF (without fault detection)
function createResultsPlots(results, faultMagnitudes)
    % Create comprehensive results visualization
    
    numMags = length(faultMagnitudes);
    
    for sensorIdx = 1:2
        figure('Position', [100, 100, 1200, 800]);
        
        for scenarioIdx = 1:5 % Assuming 5 scenarios as in the original code
            
            % Collect data for the current scenario across all magnitudes
            mean_rmse_fd = zeros(1, numMags);
            mean_rmse_nfd = zeros(1, numMags);
            mean_rmse_ckf = zeros(1, numMags);
            
            for i = 1:numMags
                mag = faultMagnitudes(i);
                scenarioKey = sprintf('mag%d_prob%d', mag, scenarioIdx);
                if isfield(results, scenarioKey)
                    mean_rmse_fd(i) = results.(scenarioKey).mean_rmse_fkf_fd(sensorIdx);
                    mean_rmse_nfd(i) = results.(scenarioKey).mean_rmse_fkf_nfd(sensorIdx);
                    mean_rmse_ckf(i) = results.(scenarioKey).mean_rmse_ckf_nfd(sensorIdx);
                end
            end
            
            subplot(2, 3, scenarioIdx);
            plot(faultMagnitudes, mean_rmse_fd, 'g-o', 'LineWidth', 2, 'MarkerSize', 8);
            hold on;
            plot(faultMagnitudes, mean_rmse_nfd, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
            plot(faultMagnitudes, mean_rmse_ckf, 'b-^', 'LineWidth', 2, 'MarkerSize', 8);
            xlabel('Fault Magnitude'); ylabel(sprintf('RMSE [m] - Sensor %d', sensorIdx));
            title(sprintf('Scenario %d: Fault Probs [%.2f, %.2f]', scenarioIdx, results.(sprintf('mag%d_prob%d', faultMagnitudes(1), scenarioIdx)).faultProbs(1), results.(sprintf('mag%d_prob%d', faultMagnitudes(1), scenarioIdx)).faultProbs(2)));
            legend('FKF w/ Fault Detection', 'FKF Traditional', 'CKF Traditional', 'Location', 'best');
            grid on;
            
        end
        sgtitle(sprintf('RMSE vs. Fault Magnitude for Sensor %d', sensorIdx));
    end
end