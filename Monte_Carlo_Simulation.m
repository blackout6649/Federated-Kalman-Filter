% Monte Carlo Simulation: Calibrated FKF vs Standard FKF (Method 2) with Fault Handling
clear; clc; close all;

%% Simulation Parameters
numRuns = 10;           % Number of Monte Carlo runs
T = 600;                 % Simulation time (s)
dt = 0.1;               % Time step (s)
N = round(T/dt);        % Number of time steps

% Calibration parameters
calibrationDuration = 1000;  % Number of steps for calibration phase
referenceWeight = 0.3;       % Weight of reference filter during calibration

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

        % Storage for this scenario - separate full, calibration, and operational
        rmse_calibrated_fkf_full = zeros(numRuns, 2);
        rmse_calibrated_fkf_calib = zeros(numRuns, 2);
        rmse_calibrated_fkf_oper = zeros(numRuns, 2);
        
        rmse_standard_fkf_2_full = zeros(numRuns, 2);
        rmse_standard_fkf_2_calib = zeros(numRuns, 2);
        rmse_standard_fkf_2_oper = zeros(numRuns, 2);
        
        rmse_fkf_nfd_full = zeros(numRuns, 2);
        rmse_fkf_nfd_calib = zeros(numRuns, 2);
        rmse_fkf_nfd_oper = zeros(numRuns, 2);
        
        rmse_ckf_nfd_full = zeros(numRuns, 2);
        rmse_ckf_nfd_calib = zeros(numRuns, 2);
        rmse_ckf_nfd_oper = zeros(numRuns, 2);

        % Detection statistics
        detectionStats_calibrated = struct();
        detectionStats_standard = struct();
        for stat_type = {'truePositives', 'falsePositives', 'trueNegatives', 'falseNegatives'}
            detectionStats_calibrated.(stat_type{1}) = zeros(numRuns, 2);
            detectionStats_standard.(stat_type{1}) = zeros(numRuns, 2);
        end

        % ISF tracking for calibrated FKF
        learnedISFs = zeros(numRuns, 2);

        for run = 1:numRuns
            if mod(run, 5) == 0
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
            % 1. Calibrated FKF with fault detection (uses ref filter)
            lkf1_cal = LocalKalmanFilter(model, s1, x0, P0, weight, "LKF1_Calibrated");
            lkf2_cal = LocalKalmanFilter(model, s2, x0, P0, weight, "LKF2_Calibrated");
            ref_filter = LocalKalmanFilter(model, s_ref, x0, P0, weight, "RefFilter");
            calibrated_fkf = FederatedKF([lkf1_cal, lkf2_cal], ref_filter, calibrationDuration, ...
                                       "CalibratedFKF", 0.05, 10, 2, referenceWeight);

            % 2. Standard FKF with fault detection (Method 2: skip update)
            lkf1_std2 = LocalKalmanFilter(model, s1, x0, P0, weight, "LKF1_Std2");
            lkf2_std2 = LocalKalmanFilter(model, s2, x0, P0, weight, "LKF2_Std2");
            standard_fkf_2 = FederatedKF_ERR_only([lkf1_std2, lkf2_std2], weight, "StandardFKF_2", 0.05, 10, 2);

            % 3. Traditional FKF without fault detection
            lkf1_nfd = LocalKalmanFilter(model, s1, x0, P0, weight, "LKF1_NFD");
            lkf2_nfd = LocalKalmanFilter(model, s2, x0, P0, weight, "LKF2_NFD");
            fkf_nfd = FederatedKFTraditional([lkf1_nfd, lkf2_nfd], weight, "FKF_NFD");

            % 4. CKF without fault detection (centralized)
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
            Xhat_standard_2 = zeros(4, N);
            Xhat_nfd = zeros(4, N);
            Xhat_ckf = zeros(4, N);

            % Fault tracking
            actualFaults1 = zeros(1, N);
            actualFaults2 = zeros(1, N);
            detectedFaults1_cal = zeros(1, N);
            detectedFaults2_cal = zeros(1, N);
            detectedFaults1_std = zeros(1, N);
            detectedFaults2_std = zeros(1, N);

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

                fuseFlag = mod(k, fusionInterval) == 0;

                % Step all filters
                calibrated_fkf.step({z1, z2}, z_ref, fuseFlag);
                Xhat_calibrated(:, k) = calibrated_fkf.x;

                standard_fkf_2.step({z1, z2}, fuseFlag);
                Xhat_standard_2(:, k) = standard_fkf_2.x;

                fkf_nfd.step({z1, z2}, fuseFlag);
                Xhat_nfd(:, k) = fkf_nfd.x;

                ckf_nfd.step({z1, z2});
                Xhat_ckf(:, k) = ckf_nfd.x;

                % Track fault detection (only in operational mode for calibrated FKF)
                if strcmp(calibrated_fkf.mode, 'operational')
                    if ~any(isnan(z1))
                        detectedFaults1_cal(k) = 1 - calibrated_fkf.fault_flags(1);
                    end
                    if ~any(isnan(z2))
                        detectedFaults2_cal(k) = 1 - calibrated_fkf.fault_flags(2);
                    end
                end

                % Track fault detection for standard FKF (method 2)
                if ~any(isnan(z1))
                    detectedFaults1_std(k) = 1 - standard_fkf_2.fault_flags(1);
                end
                if ~any(isnan(z2))
                    detectedFaults2_std(k) = 1 - standard_fkf_2.fault_flags(2);
                end
            end

            %% Store learned ISFs
            learnedISFs(run, :) = calibrated_fkf.ISF;

            %% Calculate RMSEs for different phases
            % Full simulation
            rmse_calibrated_fkf_full(run, :) = sqrt(mean((Xhat_calibrated(1:2, :) - Xtrue(1:2, :)).^2, 2));
            rmse_standard_fkf_2_full(run, :) = sqrt(mean((Xhat_standard_2(1:2, :) - Xtrue(1:2, :)).^2, 2));
            rmse_fkf_nfd_full(run, :) = sqrt(mean((Xhat_nfd(1:2, :) - Xtrue(1:2, :)).^2, 2));
            rmse_ckf_nfd_full(run, :) = sqrt(mean((Xhat_ckf(1:2, :) - Xtrue(1:2, :)).^2, 2));
            
            % Calibration phase only (first calibrationDuration steps)
            calibPhase = 1:calibrationDuration;
            rmse_calibrated_fkf_calib(run, :) = sqrt(mean((Xhat_calibrated(1:2, calibPhase) - Xtrue(1:2, calibPhase)).^2, 2));
            rmse_standard_fkf_2_calib(run, :) = sqrt(mean((Xhat_standard_2(1:2, calibPhase) - Xtrue(1:2, calibPhase)).^2, 2));
            rmse_fkf_nfd_calib(run, :) = sqrt(mean((Xhat_nfd(1:2, calibPhase) - Xtrue(1:2, calibPhase)).^2, 2));
            rmse_ckf_nfd_calib(run, :) = sqrt(mean((Xhat_ckf(1:2, calibPhase) - Xtrue(1:2, calibPhase)).^2, 2));
            
            % Operational phase only (after calibrationDuration)
            operPhase = (calibrationDuration+1):N;
            rmse_calibrated_fkf_oper(run, :) = sqrt(mean((Xhat_calibrated(1:2, operPhase) - Xtrue(1:2, operPhase)).^2, 2));
            rmse_standard_fkf_2_oper(run, :) = sqrt(mean((Xhat_standard_2(1:2, operPhase) - Xtrue(1:2, operPhase)).^2, 2));
            rmse_fkf_nfd_oper(run, :) = sqrt(mean((Xhat_nfd(1:2, operPhase) - Xtrue(1:2, operPhase)).^2, 2));
            rmse_ckf_nfd_oper(run, :) = sqrt(mean((Xhat_ckf(1:2, operPhase) - Xtrue(1:2, operPhase)).^2, 2));

            %% Calculate detection statistics (operational phase only)
            operationalStart = calibrationDuration + 1;
            for sensorIdx = 1:2
                % Calibrated FKF detection stats (operational window only)
                if sensorIdx == 1
                    actual = actualFaults1(operationalStart:end);
                    detected = detectedFaults1_cal(operationalStart:end);
                else
                    actual = actualFaults2(operationalStart:end);
                    detected = detectedFaults2_cal(operationalStart:end);
                end

                validTimes = ~isnan(actual) & ~isnan(detected);
                if any(validTimes)
                    actual = actual(validTimes);
                    detected = detected(validTimes);

                    detectionStats_calibrated.truePositives(run, sensorIdx) = sum(actual == 1 & detected == 1);
                    detectionStats_calibrated.falsePositives(run, sensorIdx) = sum(actual == 0 & detected == 1);
                    detectionStats_calibrated.trueNegatives(run, sensorIdx) = sum(actual == 0 & detected == 0);
                    detectionStats_calibrated.falseNegatives(run, sensorIdx) = sum(actual == 1 & detected == 0);
                end

                % Standard FKF detection stats (full simulation)
                if sensorIdx == 1
                    actual = actualFaults1;
                    detected = detectedFaults1_std;
                else
                    actual = actualFaults2;
                    detected = detectedFaults2_std;
                end

                validTimes = ~isnan(actual) & ~isnan(detected);
                if any(validTimes)
                    actual = actual(validTimes);
                    detected = detected(validTimes);

                    detectionStats_standard.truePositives(run, sensorIdx) = sum(actual == 1 & detected == 1);
                    detectionStats_standard.falsePositives(run, sensorIdx) = sum(actual == 0 & detected == 1);
                    detectionStats_standard.trueNegatives(run, sensorIdx) = sum(actual == 0 & detected == 0);
                    detectionStats_standard.falseNegatives(run, sensorIdx) = sum(actual == 1 & detected == 0);
                end
            end
        end % runs

        %% Store results for this scenario
        scenarioKey = sprintf('mag%d_prob%d', currentFaultMagnitude, scenarioIdx);
        results.(scenarioKey).faultProbs = [faultProb1, faultProb2];
        results.(scenarioKey).faultMagnitude = currentFaultMagnitude;

        % Store RMSE results for all three phases
        results.(scenarioKey).rmse_calibrated_fkf_full = rmse_calibrated_fkf_full;
        results.(scenarioKey).rmse_calibrated_fkf_calib = rmse_calibrated_fkf_calib;
        results.(scenarioKey).rmse_calibrated_fkf_oper = rmse_calibrated_fkf_oper;
        
        results.(scenarioKey).rmse_standard_fkf_2_full = rmse_standard_fkf_2_full;
        results.(scenarioKey).rmse_standard_fkf_2_calib = rmse_standard_fkf_2_calib;
        results.(scenarioKey).rmse_standard_fkf_2_oper = rmse_standard_fkf_2_oper;
        
        results.(scenarioKey).rmse_fkf_nfd_full = rmse_fkf_nfd_full;
        results.(scenarioKey).rmse_fkf_nfd_calib = rmse_fkf_nfd_calib;
        results.(scenarioKey).rmse_fkf_nfd_oper = rmse_fkf_nfd_oper;
        
        results.(scenarioKey).rmse_ckf_nfd_full = rmse_ckf_nfd_full;
        results.(scenarioKey).rmse_ckf_nfd_calib = rmse_ckf_nfd_calib;
        results.(scenarioKey).rmse_ckf_nfd_oper = rmse_ckf_nfd_oper;

        % Store detection stats and ISFs
        results.(scenarioKey).detectionStats_calibrated = detectionStats_calibrated;
        results.(scenarioKey).detectionStats_standard = detectionStats_standard;
        results.(scenarioKey).learnedISFs = learnedISFs;

        %% Calculate summary statistics for each phase
        phases = {'full', 'calib', 'oper'};
        for phaseIdx = 1:length(phases)
            phase = phases{phaseIdx};
            
            results.(scenarioKey).(['mean_rmse_calibrated_fkf_' phase]) = mean(rmse_calibrated_fkf_full, 1);
            results.(scenarioKey).(['mean_rmse_standard_fkf_2_' phase]) = mean(rmse_standard_fkf_2_full, 1);
            results.(scenarioKey).(['mean_rmse_fkf_nfd_' phase]) = mean(rmse_fkf_nfd_full, 1);
            results.(scenarioKey).(['mean_rmse_ckf_nfd_' phase]) = mean(rmse_ckf_nfd_full, 1);
            
            results.(scenarioKey).(['std_rmse_calibrated_fkf_' phase]) = std(rmse_calibrated_fkf_full, 1);
            results.(scenarioKey).(['std_rmse_standard_fkf_2_' phase]) = std(rmse_standard_fkf_2_full, 1);
            results.(scenarioKey).(['std_rmse_fkf_nfd_' phase]) = std(rmse_fkf_nfd_full, 1);
            results.(scenarioKey).(['std_rmse_ckf_nfd_' phase]) = std(rmse_ckf_nfd_full, 1);
        end

        % ISF statistics
        results.(scenarioKey).mean_ISFs = mean(learnedISFs, 1);
        results.(scenarioKey).std_ISFs = std(learnedISFs, 1);

        % Detection performance metrics for calibrated
        stats = results.(scenarioKey).detectionStats_calibrated;
        for sensorIdx = 1:2
            tp = stats.truePositives(:, sensorIdx);
            fp = stats.falsePositives(:, sensorIdx);
            tn = stats.trueNegatives(:, sensorIdx);
            fn = stats.falseNegatives(:, sensorIdx);

            precision = tp ./ (tp + fp + eps);
            recall = tp ./ (tp + fn + eps);
            specificity = tn ./ (tn + fp + eps);

            results.(scenarioKey).precision_calibrated(sensorIdx)   = mean(precision);
            results.(scenarioKey).recall_calibrated(sensorIdx)      = mean(recall);
            results.(scenarioKey).specificity_calibrated(sensorIdx) = mean(specificity);
            results.(scenarioKey).f1_score_calibrated(sensorIdx)    = mean(2 * precision .* recall ./ (precision + recall + eps));
        end

        % Detection performance metrics for standard (Method 2)
        stats = results.(scenarioKey).detectionStats_standard;
        for sensorIdx = 1:2
            tp = stats.truePositives(:, sensorIdx);
            fp = stats.falsePositives(:, sensorIdx);
            tn = stats.trueNegatives(:, sensorIdx);
            fn = stats.falseNegatives(:, sensorIdx);

            precision = tp ./ (tp + fp + eps);
            recall = tp ./ (tp + fn + eps);
            specificity = tn ./ (tn + fp + eps);

            results.(scenarioKey).precision_standard(sensorIdx)   = mean(precision);
            results.(scenarioKey).recall_standard(sensorIdx)      = mean(recall);
            results.(scenarioKey).specificity_standard(sensorIdx) = mean(specificity);
            results.(scenarioKey).f1_score_standard(sensorIdx)    = mean(2 * precision .* recall ./ (precision + recall + eps));
        end

    end % scenarios
end % magnitudes

%% Display Results
fprintf('\n=== MONTE CARLO SIMULATION RESULTS ===\n');
fprintf('Number of runs per scenario: %d\n', numRuns);
fprintf('Simulation time: %.1f s, Time step: %.2f s\n', T, dt);
fprintf('Calibration duration: %d steps (%.1f s)\n', calibrationDuration, calibrationDuration * dt);
fprintf('Operational duration: %d steps (%.1f s)\n', N - calibrationDuration, (N - calibrationDuration) * dt);

fieldNames = fieldnames(results);
for i = 1:length(fieldNames)
    scenarioKey = fieldNames{i};
    currentResults = results.(scenarioKey);

    fprintf('\n--- Fault Magnitude: %d, Fault Probabilities [%.2f, %.2f] ---\n', ...
        currentResults.faultMagnitude, currentResults.faultProbs(1), currentResults.faultProbs(2));

    % RMSE comparison for all three phases
    fprintf('\n*** FULL SIMULATION (Calibration + Operation) ***\n');
    fprintf('  Calibrated FKF:        %.2f m ± %.2f\n', ...
        mean(currentResults.mean_rmse_calibrated_fkf_full), mean(currentResults.std_rmse_calibrated_fkf_full));
    fprintf('  Standard FKF (Skip):   %.2f m ± %.2f\n', ...
        mean(currentResults.mean_rmse_standard_fkf_2_full), mean(currentResults.std_rmse_standard_fkf_2_full));
    fprintf('  Traditional FKF:       %.2f m ± %.2f\n', ...
        mean(currentResults.mean_rmse_fkf_nfd_full), mean(currentResults.std_rmse_fkf_nfd_full));
    fprintf('  CKF:                   %.2f m ± %.2f\n', ...
        mean(currentResults.mean_rmse_ckf_nfd_full), mean(currentResults.std_rmse_ckf_nfd_full));
    
    fprintf('\n*** CALIBRATION PHASE ONLY ***\n');
    fprintf('  Calibrated FKF:        %.2f m ± %.2f\n', ...
        mean(currentResults.mean_rmse_calibrated_fkf_calib), mean(currentResults.std_rmse_calibrated_fkf_calib));
    fprintf('  Standard FKF (Skip):   %.2f m ± %.2f\n', ...
        mean(currentResults.mean_rmse_standard_fkf_2_calib), mean(currentResults.std_rmse_standard_fkf_2_calib));
    fprintf('  Traditional FKF:       %.2f m ± %.2f\n', ...
        mean(currentResults.mean_rmse_fkf_nfd_calib), mean(currentResults.std_rmse_fkf_nfd_calib));
    fprintf('  CKF:                   %.2f m ± %.2f\n', ...
        mean(currentResults.mean_rmse_ckf_nfd_calib), mean(currentResults.std_rmse_ckf_nfd_calib));
    
    fprintf('\n*** OPERATIONAL PHASE ONLY ***\n');
    fprintf('  Calibrated FKF:        %.2f m ± %.2f\n', ...
        mean(currentResults.mean_rmse_calibrated_fkf_oper), mean(currentResults.std_rmse_calibrated_fkf_oper));
    fprintf('  Standard FKF (Skip):   %.2f m ± %.2f\n', ...
        mean(currentResults.mean_rmse_standard_fkf_2_oper), mean(currentResults.std_rmse_standard_fkf_2_oper));
    fprintf('  Traditional FKF:       %.2f m ± %.2f\n', ...
        mean(currentResults.mean_rmse_fkf_nfd_oper), mean(currentResults.std_rmse_fkf_nfd_oper));
    fprintf('  CKF:                   %.2f m ± %.2f\n', ...
        mean(currentResults.mean_rmse_ckf_nfd_oper), mean(currentResults.std_rmse_ckf_nfd_oper));

    % Learned ISFs
    fprintf('\nLearned ISFs (mean ± std): [%.2f ± %.2f, %.2f ± %.2f]\n', ...
        currentResults.mean_ISFs(1), currentResults.std_ISFs(1), ...
        currentResults.mean_ISFs(2), currentResults.std_ISFs(2));

    % Detection performance (only if faults are present)
    if currentResults.faultProbs(1) > 0 || currentResults.faultProbs(2) > 0
        fprintf('\nFault Detection Performance (Operational Phase, Sensor 1 / Sensor 2):\n');
        fprintf('  Calibrated FKF - Recall: [%.2f, %.2f], Precision: [%.2f, %.2f]\n', ...
            currentResults.recall_calibrated(1), currentResults.recall_calibrated(2), ...
            currentResults.precision_calibrated(1), currentResults.precision_calibrated(2));
        fprintf('  Standard FKF   - Recall: [%.2f, %.2f], Precision: [%.2f, %.2f]\n', ...
            currentResults.recall_standard(1), currentResults.recall_standard(2), ...
            currentResults.precision_standard(1), currentResults.precision_standard(2));
    end
end

%% Create the three RMSE comparison plots
createPhaseComparisonPlots(results, faultMagnitudes, calibrationDuration, dt);

%% Plotting function for three phases
function createPhaseComparisonPlots(results, faultMagnitudes, calibrationDuration, dt)
    numMags = length(faultMagnitudes);
    numFaultScenarios = 5;
    
    phases = {'full', 'calib', 'oper'};
    phaseTitles = {'Full Simulation (Calibration + Operation)', ...
                   sprintf('Calibration Phase Only (%.1f s)', calibrationDuration * dt), ...
                   'Operational Phase Only'};

    % Create three separate figures for each phase
    for phaseIdx = 1:3
        phase = phases{phaseIdx};
        
        figure('Position', [100 + (phaseIdx-1)*50, 100 + (phaseIdx-1)*50, 1400, 900]);

        for sensorIdx = 1:2
            for scenarioIdx = 1:numFaultScenarios

                % Collect data for this phase
                mean_rmse_calibrated = zeros(1, numMags);
                mean_rmse_std_2 = zeros(1, numMags);
                mean_rmse_fkf = zeros(1, numMags);
                mean_rmse_ckf = zeros(1, numMags);

                for i = 1:numMags
                    mag = faultMagnitudes(i);
                    scenarioKey = sprintf('mag%d_prob%d', mag, scenarioIdx);
                    if isfield(results, scenarioKey)
                        mean_rmse_calibrated(i) = results.(scenarioKey).(['mean_rmse_calibrated_fkf_' phase])(sensorIdx);
                        mean_rmse_std_2(i) = results.(scenarioKey).(['mean_rmse_standard_fkf_2_' phase])(sensorIdx);
                        mean_rmse_fkf(i) = results.(scenarioKey).(['mean_rmse_fkf_nfd_' phase])(sensorIdx);
                        mean_rmse_ckf(i) = results.(scenarioKey).(['mean_rmse_ckf_nfd_' phase])(sensorIdx);
                    end
                end

                subplot(2, 5, (sensorIdx-1)*5 + scenarioIdx);

plot(faultMagnitudes, mean_rmse_calibrated, '-o', 'LineWidth', 2.5, 'MarkerSize', 7, 'DisplayName', 'Calibrated FKF');
                hold on;
                plot(faultMagnitudes, mean_rmse_std_2, '-s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'FKF (Skip Update)');
                plot(faultMagnitudes, mean_rmse_fkf, '--d', 'LineWidth', 1.5, 'MarkerSize', 5, 'DisplayName', 'Traditional FKF');
                plot(faultMagnitudes, mean_rmse_ckf, ':^', 'LineWidth', 1.5, 'MarkerSize', 5, 'DisplayName', 'CKF');

                xlabel('Fault Magnitude [m]');
                ylabel(sprintf('RMSE [m] - Axis %d', sensorIdx));

                firstMagKey = sprintf('mag%d_prob%d', faultMagnitudes(1), scenarioIdx);
                if isfield(results, firstMagKey)
                    titleStr = sprintf('Fault Probs [%.2f, %.2f]', ...
                        results.(firstMagKey).faultProbs(1), results.(firstMagKey).faultProbs(2));
                    title(titleStr);
                end

                if scenarioIdx == 1 && sensorIdx == 1
                    legend('Location', 'northwest');
                end
                grid on;
                hold off;
            end
        end
        sgtitle(phaseTitles{phaseIdx}, 'FontSize', 14, 'FontWeight', 'bold');
    end
end