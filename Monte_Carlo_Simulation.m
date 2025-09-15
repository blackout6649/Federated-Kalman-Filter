% Monte Carlo Simulation: Fault Detection vs Traditional Filtering
% Tests effectiveness of chi-square fault detection in three federated KF scenarios
clear; clc; close all;

%% Simulation Parameters
numRuns = 10;           % Number of Monte Carlo runs
T = 600;                % Simulation time
dt = 0.1;               % Time step
N = round(T/dt);        % Number of time steps

% Fault scenarios to test
faultScenarios = [
    0.00, 0.00;     % No faults (baseline)
    0.02, 0.00;     % Light faults in sensor 1
    0.05, 0.02;     % Moderate faults in both sensors
    0.10, 0.05;     % Heavy faults
    0.15, 0.10;     % Very heavy faults
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
        rmse_fkf_fd_1 = zeros(numRuns, 2);    % FKF with fault detection (Scenario 1)
        rmse_fkf_fd_2 = zeros(numRuns, 2);    % FKF with fault detection (Scenario 2)
        rmse_fkf_fd_3 = zeros(numRuns, 2);    % FKF with fault detection (Scenario 3)
        rmse_fkf_nfd = zeros(numRuns, 2);     % FKF without fault detection
        rmse_ckf_nfd = zeros(numRuns, 2);     % CKF without fault detection
        
        % For simplicity, we assume detection stats are similar across scenarios 1-3
        % as they use the same local filters. We track stats from scenario 1.
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
            s1 = LinearSensor(H, R1, sensorIntervals(1), measDimension, "PosSensor-5m", faultProb1, currentFaultMagnitude);
            s2 = LinearSensor(H, R2, sensorIntervals(2), measDimension, "PosSensor-8m", faultProb2, currentFaultMagnitude);
            
            %% Create filter instances
            % FKF with fault detection (Scenario 1)
            lkf1_fd1 = LocalKalmanFilter(model, s1, x0, P0, weight, "LKF1_FD1");
            lkf2_fd1 = LocalKalmanFilter(model, s2, x0, P0, weight, "LKF2_FD1");
            fkf_fd_1 = FederatedKF([lkf1_fd1, lkf2_fd1], weight, "FKF_FD_1", 0.05, 10, 1);
            
            % FKF with fault detection (Scenario 2)
            lkf1_fd2 = LocalKalmanFilter(model, s1, x0, P0, weight, "LKF1_FD2");
            lkf2_fd2 = LocalKalmanFilter(model, s2, x0, P0, weight, "LKF2_FD2");
            fkf_fd_2 = FederatedKF([lkf1_fd2, lkf2_fd2], weight, "FKF_FD_2", 0.05, 10, 2);

            % FKF with fault detection (Scenario 3)
            lkf1_fd3 = LocalKalmanFilter(model, s1, x0, P0, weight, "LKF1_FD3");
            lkf2_fd3 = LocalKalmanFilter(model, s2, x0, P0, weight, "LKF2_FD3");
            fkf_fd_3 = FederatedKF([lkf1_fd3, lkf2_fd3], weight, "FKF_FD_3", 0.05, 10, 3);
            
            % FKF without fault detection (traditional)
            lkf1_nfd = LocalKalmanFilter(model, s1, x0, P0, weight, "LKF1_NFD");
            lkf2_nfd = LocalKalmanFilter(model, s2, x0, P0, weight, "LKF2_NFD");
            fkf_nfd = FederatedKFTraditional([lkf1_nfd, lkf2_nfd], weight, "FKF_NFD");
            
            % CKF without fault detection
            ckf_nfd = CentralizedKF(model, x0, P0, {s1, s2}, "CKF_NFD");
            
            %% Generate truth trajectory
            Xtrue = zeros(4, N);
            x_true = x0;
            for k = 1:N
                a = sqrt(q) * randn(2, 1);
                x_true = F * x_true + G * a;
                Xtrue(:, k) = x_true;
            end
            
            %% Storage for this run
            Xhat_fd_1 = zeros(4, N);
            Xhat_fd_2 = zeros(4, N);
            Xhat_fd_3 = zeros(4, N);
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
                
                % Step all filters
                fkf_fd_1.step({z1, z2}, fuseFlag);
                Xhat_fd_1(:, k) = fkf_fd_1.x;
                
                fkf_fd_2.step({z1, z2}, fuseFlag);
                Xhat_fd_2(:, k) = fkf_fd_2.x;

                fkf_fd_3.step({z1, z2}, fuseFlag);
                Xhat_fd_3(:, k) = fkf_fd_3.x;

                fkf_nfd.step({z1, z2}, fuseFlag);
                Xhat_nfd(:, k) = fkf_nfd.x;
                
                ckf_nfd.step({z1, z2});
                Xhat_ckf(:, k) = ckf_nfd.x;

                % Track fault detection (from FKF scenario 1 instance)
                if ~any(isnan(z1))
                    detectedFaults1(k) = 1 - fkf_fd_1.fault_flags(1);
                end
                if ~any(isnan(z2))
                    detectedFaults2(k) = 1 - fkf_fd_1.fault_flags(2);
                end
            end
            
            %% Calculate RMSEs
            rmse_fkf_fd_1(run, :) = sqrt(mean((Xhat_fd_1(1:2, :) - Xtrue(1:2, :)).^2, 2));
            rmse_fkf_fd_2(run, :) = sqrt(mean((Xhat_fd_2(1:2, :) - Xtrue(1:2, :)).^2, 2));
            rmse_fkf_fd_3(run, :) = sqrt(mean((Xhat_fd_3(1:2, :) - Xtrue(1:2, :)).^2, 2));
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
        
        results.(scenarioKey).rmse_fkf_fd_1 = rmse_fkf_fd_1;
        results.(scenarioKey).rmse_fkf_fd_2 = rmse_fkf_fd_2;
        results.(scenarioKey).rmse_fkf_fd_3 = rmse_fkf_fd_3;
        results.(scenarioKey).rmse_fkf_nfd = rmse_fkf_nfd;
        results.(scenarioKey).rmse_ckf_nfd = rmse_ckf_nfd;
        results.(scenarioKey).detectionStats = detectionStats;
        
        %% Calculate summary statistics
        results.(scenarioKey).mean_rmse_fkf_fd_1 = mean(rmse_fkf_fd_1, 1);
        results.(scenarioKey).mean_rmse_fkf_fd_2 = mean(rmse_fkf_fd_2, 1);
        results.(scenarioKey).mean_rmse_fkf_fd_3 = mean(rmse_fkf_fd_3, 1);
        results.(scenarioKey).mean_rmse_fkf_nfd = mean(rmse_fkf_nfd, 1);
        results.(scenarioKey).mean_rmse_ckf_nfd = mean(rmse_ckf_nfd, 1);
        
        results.(scenarioKey).std_rmse_fkf_fd_1 = std(rmse_fkf_fd_1, 1);
        results.(scenarioKey).std_rmse_fkf_fd_2 = std(rmse_fkf_fd_2, 1);
        results.(scenarioKey).std_rmse_fkf_fd_3 = std(rmse_fkf_fd_3, 1);
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

%% Display Results (Abbreviated for clarity, full data is in 'results' struct)
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
    fprintf('RMSE Results (mean of x, y): \n');
    fprintf('  FKF w/ FD (Scenario 1):  %.2f m\n', mean(currentResults.mean_rmse_fkf_fd_1));
    fprintf('  FKF w/ FD (Scenario 2):  %.2f m\n', mean(currentResults.mean_rmse_fkf_fd_2));
    fprintf('  FKF w/ FD (Scenario 3):  %.2f m\n', mean(currentResults.mean_rmse_fkf_fd_3));
    fprintf('  FKF Traditional:         %.2f m\n', mean(currentResults.mean_rmse_fkf_nfd));
    fprintf('  CKF Traditional:         %.2f m\n', mean(currentResults.mean_rmse_ckf_nfd));

    % Detection performance
    if currentResults.faultProbs(1) > 0 || currentResults.faultProbs(2) > 0
        fprintf('Fault Detection Performance (Sensor 1/Sensor 2):\n');
        fprintf('  Recall:    [%.2f, %.2f]\n', currentResults.recall(1), currentResults.recall(2));
        fprintf('  Precision: [%.2f, %.2f]\n', currentResults.precision(1), currentResults.precision(2));
    end
end

%% Visualization
createResultsPlots(results, faultMagnitudes);

%% Helper function for plotting
function createResultsPlots(results, faultMagnitudes)
    % Create comprehensive results visualization comparing all scenarios
    
    numMags = length(faultMagnitudes);
    numFaultScenarios = 5; % As defined in the main script
    
    for sensorIdx = 1:2
        figure('Position', [100, 100, 1600, 900]);
        
        for scenarioIdx = 1:numFaultScenarios
            
            % Collect data for the current fault scenario across all magnitudes
            mean_rmse_fd_1 = zeros(1, numMags);
            mean_rmse_fd_2 = zeros(1, numMags);
            mean_rmse_fd_3 = zeros(1, numMags);
            mean_rmse_nfd = zeros(1, numMags);
            mean_rmse_ckf = zeros(1, numMags);
            
            for i = 1:numMags
                mag = faultMagnitudes(i);
                scenarioKey = sprintf('mag%d_prob%d', mag, scenarioIdx);
                if isfield(results, scenarioKey)
                    mean_rmse_fd_1(i) = results.(scenarioKey).mean_rmse_fkf_fd_1(sensorIdx);
                    mean_rmse_fd_2(i) = results.(scenarioKey).mean_rmse_fkf_fd_2(sensorIdx);
                    mean_rmse_fd_3(i) = results.(scenarioKey).mean_rmse_fkf_fd_3(sensorIdx);
                    mean_rmse_nfd(i) = results.(scenarioKey).mean_rmse_fkf_nfd(sensorIdx);
                    mean_rmse_ckf(i) = results.(scenarioKey).mean_rmse_ckf_nfd(sensorIdx);
                end
            end
            
            subplot(2, 3, scenarioIdx);
            plot(faultMagnitudes, mean_rmse_fd_1, 'g-o', 'LineWidth', 2, 'MarkerSize', 8);
            hold on;
            plot(faultMagnitudes, mean_rmse_fd_2, 'c-d', 'LineWidth', 2, 'MarkerSize', 8);
            plot(faultMagnitudes, mean_rmse_fd_3, 'm-p', 'LineWidth', 2, 'MarkerSize', 8);
            plot(faultMagnitudes, mean_rmse_nfd, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
            plot(faultMagnitudes, mean_rmse_ckf, 'b-^', 'LineWidth', 2, 'MarkerSize', 8);
            
            xlabel('Fault Magnitude');
            ylabel(sprintf('RMSE [m] - Axis %d', sensorIdx));
            
            % Get fault probabilities for the title
            firstMagKey = sprintf('mag%d_prob%d', faultMagnitudes(1), scenarioIdx);
            titleStr = sprintf('Fault Probs [%.2f, %.2f]', ...
                results.(firstMagKey).faultProbs(1), results.(firstMagKey).faultProbs(2));
            title(titleStr);
            
            if scenarioIdx == 1
                legend('FKF - Assign last global estimate', 'FKF - Only skip update', 'FKF - Assign last global estimate and increase cov', ...
                       'FKF Traditional', 'CKF Traditional', 'Location', [0.72, 0.18, 0.2, 0.2]);
            end
            grid on;
            set(gca, 'FontSize', 12);
        end
        sgtitle(sprintf('RMSE vs. Fault Magnitude for Sensor %d', sensorIdx), 'FontSize', 16, 'FontWeight', 'bold');
    end
end