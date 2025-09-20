% Monte Carlo Simulation: Calibrated FKF vs Traditional Filtering
% Tests effectiveness of calibrated FKF with learned ISFs against traditional methods
clear; clc; close all;

%% Simulation Parameters
numRuns = 10;           % Number of Monte Carlo runs
T = 600;                % Simulation time
dt = 0.1;               % Time step
N = round(T/dt);        % Number of time steps

% Calibration parameters
calibrationDuration = 1000;  % Number of steps for calibration phase
referenceWeight = 0.3;      % Weight of reference filter during calibration

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
R_ref = (3^2)*eye(2);  % Reference sensor (better accuracy)

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
        rmse_calibrated_fkf = zeros(numRuns, 2);  % Calibrated FKF
        rmse_fkf_nfd = zeros(numRuns, 2);         % Traditional FKF without fault detection
        rmse_ckf_nfd = zeros(numRuns, 2);         % CKF without fault detection
        
        % Detection statistics for calibrated FKF
        detectionStats = struct();
        detectionStats.truePositives = zeros(numRuns, 2);
        detectionStats.falsePositives = zeros(numRuns, 2);
        detectionStats.trueNegatives = zeros(numRuns, 2);
        detectionStats.falseNegatives = zeros(numRuns, 2);
        
        % ISF tracking
        learnedISFs = zeros(numRuns, 2);
        
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
            
            % Reference sensor (high accuracy, no faults for calibration)
            s_ref = LinearSensor(H, R_ref, 1, measDimension, "RefSensor-3m", 0.0, 0);
            
            %% Create filter instances
            % Calibrated FKF with fault detection (method 2: skip update)
            lkf1_cal = LocalKalmanFilter(model, s1, x0, P0, weight, "LKF1_Calibrated");
            lkf2_cal = LocalKalmanFilter(model, s2, x0, P0, weight, "LKF2_Calibrated");
            ref_filter = LocalKalmanFilter(model, s_ref, x0, P0, weight, "RefFilter");
            calibrated_fkf = FederatedKF([lkf1_cal, lkf2_cal], ref_filter, calibrationDuration, ...
                                       "CalibratedFKF", 0.05, 10, 2, referenceWeight);
            
            % Traditional FKF without fault detection
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
            Xhat_calibrated = zeros(4, N);
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
                z_ref_base = H * Xtrue(:, k) + mvnrnd(zeros(2, 1), R_ref)';
                
                % Add faults with specified probabilities to local sensors only
                z1 = z1_base; z2 = z2_base; z_ref = z_ref_base;
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
                % Reference sensor measures every time step (interval = 1)
                
                fuseFlag = mod(k, fusionInterval) == 0;
                
                % Step all filters
                calibrated_fkf.step({z1, z2}, z_ref, fuseFlag);
                Xhat_calibrated(:, k) = calibrated_fkf.x;
                
                fkf_nfd.step({z1, z2}, fuseFlag);
                Xhat_nfd(:, k) = fkf_nfd.x;
                
                ckf_nfd.step({z1, z2});
                Xhat_ckf(:, k) = ckf_nfd.x;

                % Track fault detection (only in operational mode)
                if strcmp(calibrated_fkf.mode, 'operational')
                    if ~any(isnan(z1))
                        detectedFaults1(k) = 1 - calibrated_fkf.fault_flags(1);
                    end
                    if ~any(isnan(z2))
                        detectedFaults2(k) = 1 - calibrated_fkf.fault_flags(2);
                    end
                end
            end
            
            %% Store learned ISFs
            learnedISFs(run, :) = calibrated_fkf.ISF;
            
            %% Calculate RMSEs
            rmse_calibrated_fkf(run, :) = sqrt(mean((Xhat_calibrated(1:2, :) - Xtrue(1:2, :)).^2, 2));
            rmse_fkf_nfd(run, :) = sqrt(mean((Xhat_nfd(1:2, :) - Xtrue(1:2, :)).^2, 2));
            rmse_ckf_nfd(run, :) = sqrt(mean((Xhat_ckf(1:2, :) - Xtrue(1:2, :)).^2, 2));
            
            %% Calculate detection statistics (only for operational phase)
            % Only consider time steps after calibration
            operationalStart = calibrationDuration + 1;
            for sensorIdx = 1:2
                if sensorIdx == 1
                    actual = actualFaults1(operationalStart:end);
                    detected = detectedFaults1(operationalStart:end);
                else
                    actual = actualFaults2(operationalStart:end);
                    detected = detectedFaults2(operationalStart:end);
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
        
        results.(scenarioKey).rmse_calibrated_fkf = rmse_calibrated_fkf;
        results.(scenarioKey).rmse_fkf_nfd = rmse_fkf_nfd;
        results.(scenarioKey).rmse_ckf_nfd = rmse_ckf_nfd;
        results.(scenarioKey).detectionStats = detectionStats;
        results.(scenarioKey).learnedISFs = learnedISFs;
        
        %% Calculate summary statistics
        results.(scenarioKey).mean_rmse_calibrated_fkf = mean(rmse_calibrated_fkf, 1);
        results.(scenarioKey).mean_rmse_fkf_nfd = mean(rmse_fkf_nfd, 1);
        results.(scenarioKey).mean_rmse_ckf_nfd = mean(rmse_ckf_nfd, 1);
        
        results.(scenarioKey).std_rmse_calibrated_fkf = std(rmse_calibrated_fkf, 1);
        results.(scenarioKey).std_rmse_fkf_nfd = std(rmse_fkf_nfd, 1);
        results.(scenarioKey).std_rmse_ckf_nfd = std(rmse_ckf_nfd, 1);
        
        % ISF statistics
        results.(scenarioKey).mean_ISFs = mean(learnedISFs, 1);
        results.(scenarioKey).std_ISFs = std(learnedISFs, 1);
        
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
fprintf('Calibration duration: %d steps (%.1f s)\n', calibrationDuration, calibrationDuration * dt);
fieldNames = fieldnames(results);
for i = 1:length(fieldNames)
    scenarioKey = fieldNames{i};
    currentResults = results.(scenarioKey);
    
    fprintf('\n--- Fault Magnitude: %d, Fault Probabilities [%.2f, %.2f] ---\n', ...
        currentResults.faultMagnitude, currentResults.faultProbs(1), currentResults.faultProbs(2));
    
    % RMSE comparison
    fprintf('RMSE Results (mean of x, y): \n');
    fprintf('  Calibrated FKF:     %.2f m ± %.2f\n', ...
        mean(currentResults.mean_rmse_calibrated_fkf), mean(currentResults.std_rmse_calibrated_fkf));
    fprintf('  Traditional FKF:    %.2f m ± %.2f\n', ...
        mean(currentResults.mean_rmse_fkf_nfd), mean(currentResults.std_rmse_fkf_nfd));
    fprintf('  Traditional CKF:    %.2f m ± %.2f\n', ...
        mean(currentResults.mean_rmse_ckf_nfd), mean(currentResults.std_rmse_ckf_nfd));

    % Learned ISFs
    fprintf('Learned ISFs (mean ± std): [%.2f ± %.2f, %.2f ± %.2f]\n', ...
        currentResults.mean_ISFs(1), currentResults.std_ISFs(1), ...
        currentResults.mean_ISFs(2), currentResults.std_ISFs(2));
    fprintf('ISF Sum: %.2f (theoretical: %.0f)\n', sum(currentResults.mean_ISFs), length(currentResults.mean_ISFs));

    % Performance improvement
    if mean(currentResults.mean_rmse_fkf_nfd) > 0
        improvement_vs_fkf = (mean(currentResults.mean_rmse_fkf_nfd) - mean(currentResults.mean_rmse_calibrated_fkf)) / ...
                            mean(currentResults.mean_rmse_fkf_nfd) * 100;
        fprintf('Improvement vs Traditional FKF: %.1f%%\n', improvement_vs_fkf);
    end
    
    if mean(currentResults.mean_rmse_ckf_nfd) > 0
        improvement_vs_ckf = (mean(currentResults.mean_rmse_ckf_nfd) - mean(currentResults.mean_rmse_calibrated_fkf)) / ...
                            mean(currentResults.mean_rmse_ckf_nfd) * 100;
        fprintf('Improvement vs Traditional CKF: %.1f%%\n', improvement_vs_ckf);
    end

    % Detection performance (only if faults are present)
    if currentResults.faultProbs(1) > 0 || currentResults.faultProbs(2) > 0
        fprintf('Fault Detection Performance (Sensor 1/Sensor 2):\n');
        fprintf('  Recall:    [%.2f, %.2f]\n', currentResults.recall(1), currentResults.recall(2));
        fprintf('  Precision: [%.2f, %.2f]\n', currentResults.precision(1), currentResults.precision(2));
        fprintf('  F1 Score:  [%.2f, %.2f]\n', currentResults.f1_score(1), currentResults.f1_score(2));
    end
end

%% Visualization
createResultsPlots(results, faultMagnitudes);

%% Helper function for plotting
function createResultsPlots(results, faultMagnitudes)
    % Create comprehensive results visualization for calibrated FKF comparison
    
    numMags = length(faultMagnitudes);
    numFaultScenarios = 5; % As defined in the main script
    
    %% --------- RMSE Figure ---------
    figure('Position', [100, 100, 1800, 800]);
    
    for sensorIdx = 1:2
        for scenarioIdx = 1:numFaultScenarios
            
            % Collect data for the current fault scenario across all magnitudes
            mean_rmse_calibrated = zeros(1, numMags);
            mean_rmse_fkf = zeros(1, numMags);
            mean_rmse_ckf = zeros(1, numMags);
            std_rmse_calibrated = zeros(1, numMags);
            
            for i = 1:numMags
                mag = faultMagnitudes(i);
                scenarioKey = sprintf('mag%d_prob%d', mag, scenarioIdx);
                if isfield(results, scenarioKey)
                    mean_rmse_calibrated(i) = results.(scenarioKey).mean_rmse_calibrated_fkf(sensorIdx);
                    mean_rmse_fkf(i) = results.(scenarioKey).mean_rmse_fkf_nfd(sensorIdx);
                    mean_rmse_ckf(i) = results.(scenarioKey).mean_rmse_ckf_nfd(sensorIdx);
                    std_rmse_calibrated(i) = results.(scenarioKey).std_rmse_calibrated_fkf(sensorIdx);
                end
            end
            
            % RMSE comparison plot
            subplot(2, 5, (sensorIdx-1)*5 + scenarioIdx);
            errorbar(faultMagnitudes, mean_rmse_calibrated, std_rmse_calibrated, 'g-o', 'LineWidth', 2, 'MarkerSize', 6);
            hold on;
            plot(faultMagnitudes, mean_rmse_fkf, 'r-s', 'LineWidth', 2, 'MarkerSize', 6);
            plot(faultMagnitudes, mean_rmse_ckf, 'b-^', 'LineWidth', 2, 'MarkerSize', 6);
            
            xlabel('Fault Magnitude');
            ylabel(sprintf('RMSE [m] - Axis %d', sensorIdx));
            
            % Get fault probabilities for the title
            firstMagKey = sprintf('mag%d_prob%d', faultMagnitudes(1), scenarioIdx);
            if isfield(results, firstMagKey)
                titleStr = sprintf('Fault Probs [%.2f, %.2f]', ...
                    results.(firstMagKey).faultProbs(1), results.(firstMagKey).faultProbs(2));
                title(titleStr);
            end
            
            if scenarioIdx == 1 && sensorIdx == 1
                legend('Calibrated FKF', 'Traditional FKF', 'Traditional CKF', 'Location', 'best');
            end
            grid on;
            set(gca, 'FontSize', 10);
        end
    end
    sgtitle('Calibrated FKF RMSE Comparison', 'FontSize', 16, 'FontWeight', 'bold');
    
    %% --------- ISF Figure (standalone) ---------
    figure('Position', [100, 100, 1600, 400]);
    
    for scenarioIdx = 1:numFaultScenarios
        subplot(1, 5, scenarioIdx);
        
        % Collect ISF data
        mean_isf1 = zeros(1, numMags);
        mean_isf2 = zeros(1, numMags);
        
        for i = 1:numMags
            mag = faultMagnitudes(i);
            scenarioKey = sprintf('mag%d_prob%d', mag, scenarioIdx);
            if isfield(results, scenarioKey)
                mean_isf1(i) = results.(scenarioKey).mean_ISFs(1);
                mean_isf2(i) = results.(scenarioKey).mean_ISFs(2);
            end
        end
        
        plot(faultMagnitudes, mean_isf1, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
        hold on;
        plot(faultMagnitudes, mean_isf2, 'r-s', 'LineWidth', 2, 'MarkerSize', 6);
        plot(faultMagnitudes, ones(size(faultMagnitudes)), 'k--', 'LineWidth', 1);
        
        xlabel('Fault Magnitude');
        ylabel('Learned ISF');
        
        % Get fault probabilities for the title
        firstMagKey = sprintf('mag%d_prob%d', faultMagnitudes(1), scenarioIdx);
        if isfield(results, firstMagKey)
            titleStr = sprintf('ISFs - Fault Probs [%.2f, %.2f]', ...
                results.(firstMagKey).faultProbs(1), results.(firstMagKey).faultProbs(2));
            title(titleStr);
        end
        
        if scenarioIdx == 1
            legend('ISF Sensor 1', 'ISF Sensor 2', 'Equal Weight', 'Location', 'best');
        end
        grid on;
        set(gca, 'FontSize', 10);
    end
    sgtitle('Learned Sensor Weights (ISFs)', 'FontSize', 16, 'FontWeight', 'bold');
end
