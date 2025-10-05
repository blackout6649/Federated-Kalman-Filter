% DEMO: Federated KF with two linear position sensors of different noise
% Version 1.2 - Modified to stop reference sensor after calibration period
% Develop
clear; clc; rng(4); close all;

% ===== Frequencies =====
sensorIntervals = [1, 1, 1]; % Sensor 1 every 1 steps, sensor 2 every 1 steps, reference (last) every 1 steps
fusionInterval = 10;       % Fuse every 10 steps
weight = length(sensorIntervals); % Number of sensors
calibrationPeriod = 1000;  % Same as calibrated FKF - stop reference after this

% ===== Model =====
dt = 0.1;
F  = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];
G  = [0.5*dt^2*eye(2); dt*eye(2)];
q  = 0.05;
Q  = q*eye(2);
model = MotionModel(F,G,Q);

% ===== Sensors (both linear position, different accuracies, with faults) =====
H = [1 0 0 0; 0 1 0 0];
measDimension = 2;
R1 = (5^2)*eye(2);   
s1 = LinearSensor(H,R1, sensorIntervals(1), measDimension, "PosSensor-5m", 0.02, 50); % 2% chance, ±50m error
R2 = (8^2)*eye(2);   
s2 = LinearSensor(H,R2, sensorIntervals(2), measDimension, "PosSensor-8m", 0.05, 100);   % 5% chance, ±100m error
Rref = (0.1^2)*eye(2);   
sref = LinearSensor(H,Rref, sensorIntervals(3), measDimension, "PosSensor-0.1m", 0.001, 1);   % 0.1% chance, ±1m error

% ===== Locals =====
x0 = [0; 0; 1; 0.6]; 
P0 = diag([25 25 4 4]);

% The traditional FKF and the new calibrated FKF must have their own unique LocalKalmanFilter objects
% so that their states and covariances do not interfere.
lkf1_trad = LocalKalmanFilter(model, s1, x0, P0, weight, "LKF1_Trad");
lkf2_trad = LocalKalmanFilter(model, s2, x0, P0, weight, "LKF2_Trad");
lkfRef_trad = LocalKalmanFilter(model, sref, x0, P0, weight, "LKFref_Trad");

lkf1_calib = LocalKalmanFilter(model, s1, x0, P0, weight, "LKF1_Calib");
lkf2_calib = LocalKalmanFilter(model, s2, x0, P0, weight, "LKF2_Calib");
lkfRef_calib = LocalKalmanFilter(model, sref, x0, P0, weight, "LKFref_Calib");

% ===== Centralized KF (baseline) =====
ckf = CentralizedKF(model, x0, P0, {s1, s2, sref}, "CKF");

% ===== Traditional Federated KF =====
% This one will not have a calibration phase and will not use the reference sensor's fusion weight
fkf_Trad = FederatedKFTraditional([lkf1_trad, lkf2_trad, lkfRef_trad] ,weight, "FKF_Trad");

% ===== Calibrated Federated manager =====
% This one will use the reference sensor for calibration and fusion
% The reference filter is lkfRef_calib, and the local filters are lkf1_calib and lkf2_calib
fkf_Calib = FederatedKF([lkf1_calib, lkf2_calib], lkfRef_calib, calibrationPeriod, "Calib-FKF", 0.05, 10, 2, 0.6);

% ===== Truth sim =====
T = 600; 
N = round(T/dt);
Xtrue = zeros(4,N);
x = [0; 0; 1; 0.6];
for k=1:N
    a = sqrt(q)*randn(2,1);
    x = F*x + G*a;
    Xtrue(:,k) = x;
end

% ===== Run =====
% Data storage for all filters
Xhat1_calib = zeros(4,N); Xhat2_calib = zeros(4,N); 
Xhat1_trad = zeros(4,N); Xhat2_trad = zeros(4,N);
XhatRef = zeros(4,N);
Xf_calib = zeros(4,N);
Xf_trad = zeros(4,N);
Xc = zeros(4,N);

% Track when reference sensor is used
refUsageFlags = zeros(1,N); % 1 = reference used, 0 = reference not used

for k=1:N
    % Generate measurements
    z1 = s1.measure(Xtrue(:,k), k);
    z2 = s2.measure(Xtrue(:,k), k);
    zref = sref.measure(Xtrue(:,k), k);
    
    % Store measurements for plotting (optional, but good for diagnostics)
    XMeas1(:,k) = z1; 
    XMeas2(:,k) = z2; 
    XMeasref(:,k) = zref;
    
    fuseFlag  = mod(k, fusionInterval) == 0;
    
    % Determine if we should use reference sensor
    useReference = k <= calibrationPeriod;
    refUsageFlags(k) = useReference;
    
    % --- Update Calibrated FKF ---
    % z_cell for calibrated FKF only includes the two local filters
    % (The calibrated FKF handles its own reference sensor logic internally)
    fkf_Calib.step({z1, z2}, zref, fuseFlag);         
    Xhat1_calib(:,k) = lkf1_calib.x;
    Xhat2_calib(:,k) = lkf2_calib.x;
    XhatRef(:,k) = lkfRef_calib.x;
    Xf_calib(:,k) = fkf_Calib.x;

    % --- Update Traditional FKF ---
    if useReference
        % During calibration period: include reference sensor
        fkf_Trad.step({z1, z2, zref}, fuseFlag);
    else
        % After calibration period: exclude reference sensor
        fkf_Trad.step({z1, z2}, fuseFlag);
        if k == calibrationPeriod + 1
            fprintf('--- Traditional FKF: Stopped using reference sensor at step %d ---\n', k);
        end
    end
    Xhat1_trad(:,k) = lkf1_trad.x;
    Xhat2_trad(:,k) = lkf2_trad.x;
    Xf_trad(:,k) = fkf_Trad.x;

    % --- Centralized KF step ---
    if useReference
        % During calibration period: include reference sensor
        ckf.step({z1, z2, zref});
    else
        % After calibration period: exclude reference sensor  
        ckf.step({z1, z2});
        if k == calibrationPeriod + 1
            fprintf('--- Centralized KF: Stopped using reference sensor at step %d ---\n', k);
        end
    end
    Xc(:,k) = ckf.x;
end

%% Visual Aid
% ===== Evaluation =====
% Calculate RMSE for entire period
rmse1_calib = sqrt(mean((Xhat1_calib(1:2,:)-Xtrue(1:2,:)).^2,2));
rmse2_calib = sqrt(mean((Xhat2_calib(1:2,:)-Xtrue(1:2,:)).^2,2));
rmsef_calib = sqrt(mean((Xf_calib(1:2,:)-Xtrue(1:2,:)).^2,2));

rmse1_trad = sqrt(mean((Xhat1_trad(1:2,:)-Xtrue(1:2,:)).^2,2));
rmse2_trad = sqrt(mean((Xhat2_trad(1:2,:)-Xtrue(1:2,:)).^2,2));
rmsef_trad = sqrt(mean((Xf_trad(1:2,:)-Xtrue(1:2,:)).^2,2));

rmseRef = sqrt(mean((XhatRef(1:2,:)-Xtrue(1:2,:)).^2,2));
rmsec = sqrt(mean((Xc(1:2,:)-Xtrue(1:2,:)).^2,2));

% Calculate RMSE for operational period only (after calibration)
opPeriodIdx = (calibrationPeriod+1):N;
if ~isempty(opPeriodIdx)
    rmsef_calib_op = sqrt(mean((Xf_calib(1:2,opPeriodIdx)-Xtrue(1:2,opPeriodIdx)).^2,2));
    rmsef_trad_op = sqrt(mean((Xf_trad(1:2,opPeriodIdx)-Xtrue(1:2,opPeriodIdx)).^2,2));
    rmsec_op = sqrt(mean((Xc(1:2,opPeriodIdx)-Xtrue(1:2,opPeriodIdx)).^2,2));
else
    rmsef_calib_op = [NaN; NaN];
    rmsef_trad_op = [NaN; NaN];
    rmsec_op = [NaN; NaN];
end

fprintf('=== OVERALL PERFORMANCE (Full Simulation) ===\n');
fprintf('--- Calibrated FKF Results ---\n');
fprintf('Local 1 RMSE [x,y]: [%.2f, %.2f] m\n', rmse1_calib(1), rmse1_calib(2));
fprintf('Local 2 RMSE [x,y]: [%.2f, %.2f] m\n', rmse2_calib(1), rmse2_calib(2));
fprintf('Fused Calibrated FKF RMSE [x,y]: [%.2f, %.2f] m\n', rmsef_calib(1), rmsef_calib(2));
fprintf('\n--- Traditional FKF Results ---\n');
fprintf('Local 1 RMSE [x,y]: [%.2f, %.2f] m\n', rmse1_trad(1), rmse1_trad(2));
fprintf('Local 2 RMSE [x,y]: [%.2f, %.2f] m\n', rmse2_trad(1), rmse2_trad(2));
fprintf('Fused Traditional FKF RMSE [x,y]: [%.2f, %.2f] m\n', rmsef_trad(1), rmsef_trad(2));
fprintf('\n--- Baselines ---\n');
fprintf('Reference Sensor RMSE [x,y]: [%.2f, %.2f] m\n', rmseRef(1), rmseRef(2));
fprintf('Centralized KF RMSE [x,y]: [%.2f, %.2f] m\n', rmsec(1), rmsec(2));

fprintf('\n=== OPERATIONAL PERFORMANCE (After Step %d) ===\n', calibrationPeriod);
if ~isempty(opPeriodIdx)
    fprintf('Fused Calibrated FKF RMSE [x,y]: [%.2f, %.2f] m\n', rmsef_calib_op(1), rmsef_calib_op(2));
    fprintf('Fused Traditional FKF RMSE [x,y]: [%.2f, %.2f] m\n', rmsef_trad_op(1), rmsef_trad_op(2));
    fprintf('Centralized KF RMSE [x,y]: [%.2f, %.2f] m\n', rmsec_op(1), rmsec_op(2));
else
    fprintf('No operational period data (simulation too short)\n');
end

figure; hold on; grid on; axis equal;
% Truth 
plot(Xtrue(1,:), Xtrue(2,:), 'k--', 'LineWidth', 1.5);
% Measurements (dots)
plot(XMeas1(1,:), XMeas1(2,:), 'b.', 'MarkerSize', 0.5); % Sensor 1 dots
plot(XMeas2(1,:), XMeas2(2,:), 'r.', 'MarkerSize', 0.5); % Sensor 2 dots
plot(XMeasref(1,:), XMeasref(2,:), 'c.', 'MarkerSize', 0.5); % Reference sensor dots
% Filter outputs (bold lines)
plot(Xf_calib(1,:),   Xf_calib(2,:),   'g-',  'LineWidth', 3);
plot(Xf_trad(1,:),   Xf_trad(2,:),   'm-',  'LineWidth', 3);
plot(Xc(1,:),   Xc(2,:),   'y-',  'LineWidth', 3);

% Mark calibration end point
if calibrationPeriod <= N
    plot(Xtrue(1,calibrationPeriod), Xtrue(2,calibrationPeriod), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'black');
    text(Xtrue(1,calibrationPeriod), Xtrue(2,calibrationPeriod), sprintf('  Cal. End (Step %d)', calibrationPeriod), 'FontSize', 10);
end

legend('Truth','Sensor 1 meas','Sensor 2 meas','Ref. Sensor meas','FKF (Calibrated)','FKF (Traditional)','CKF', 'Calibration End');
title(sprintf('Federated vs Centralized KF (Ref. sensor stopped after step %d)', calibrationPeriod));
xlabel("X - Coordinate"); ylabel("Y - Coordinate")

%% ===== Improved Multi-Plot Visualization =====
t = (0:N-1) * dt; % time vector
fuseTimes = find(mod(1:N, fusionInterval) == 0); % fusion step indices
fuseT = t(fuseTimes); % fusion timestamps

%% Plot 1: Local Filter Performance Comparison
figure('Position', [100, 100, 1400, 600]);

% X-position errors for local filters
subplot(2,2,1); hold on; grid on;
plot(t, Xtrue(1,:) - Xhat1_calib(1,:), 'b-', 'LineWidth', 1.5);
plot(t, Xtrue(1,:) - Xhat2_calib(1,:), 'r-', 'LineWidth', 1.5);
% Mark calibration end
if calibrationPeriod <= N
    xline(t(calibrationPeriod), 'k--', 'LineWidth', 2, 'Alpha', 0.7);
    text(t(calibrationPeriod), max(ylim)*0.8, 'Cal. End', 'Rotation', 90, 'FontSize', 10);
end
xlabel('Time [s]'); ylabel('x-error [m]');
title('Local Filters: X-Position Error');
legend('Local Filter 1', 'Local Filter 2', 'Location', 'best');

% Y-position errors for local filters
subplot(2,2,2); hold on; grid on;
plot(t, Xtrue(2,:) - Xhat1_calib(2,:), 'b-', 'LineWidth', 1.5);
plot(t, Xtrue(2,:) - Xhat2_calib(2,:), 'r-', 'LineWidth', 1.5);
% Mark calibration end
if calibrationPeriod <= N
    xline(t(calibrationPeriod), 'k--', 'LineWidth', 2, 'Alpha', 0.7);
    text(t(calibrationPeriod), max(ylim)*0.8, 'Cal. End', 'Rotation', 90, 'FontSize', 10);
end
xlabel('Time [s]'); ylabel('y-error [m]');
title('Local Filters: Y-Position Error');
legend('Local Filter 1', 'Local Filter 2', 'Location', 'best');

% Combined position error magnitude
subplot(2,2,3:4); hold on; grid on;
error1_mag = sqrt((Xtrue(1,:) - Xhat1_calib(1,:)).^2 + (Xtrue(2,:) - Xhat2_calib(2,:)).^2);
error2_mag = sqrt((Xtrue(1,:) - Xhat2_calib(1,:)).^2 + (Xtrue(2,:) - Xhat2_calib(2,:)).^2);
plot(t, error1_mag, 'b-', 'LineWidth', 1.5);
plot(t, error2_mag, 'r-', 'LineWidth', 1.5);
% Mark calibration end
if calibrationPeriod <= N
    xline(t(calibrationPeriod), 'k--', 'LineWidth', 2, 'Alpha', 0.7);
    text(t(calibrationPeriod), max(ylim)*0.8, 'Calibration Ends', 'Rotation', 90, 'FontSize', 12);
end
xlabel('Time [s]'); ylabel('Position Error Magnitude [m]');
title('Local Filters: Combined Position Error Magnitude');
legend('Local Filter 1', 'Local Filter 2', 'Location', 'best');

sgtitle('Local Filter Performance Analysis', 'FontSize', 14, 'FontWeight', 'bold');

%% Plot 2: Fusion Methods Comparison
figure('Position', [200, 200, 1400, 600]);

% X-position fusion comparison
subplot(2,1,1); hold on; grid on;
plot(t, Xtrue(1,:) - Xf_calib(1,:), 'g-', 'LineWidth', 2);
plot(t, Xtrue(1,:) - Xf_trad(1,:), 'm-', 'LineWidth', 2);
plot(t, Xtrue(1,:) - Xc(1,:), 'y-', 'LineWidth', 1.5);
% Mark fusion points
plot(fuseT, Xtrue(1,fuseTimes) - Xf_calib(1,fuseTimes), 'go', 'MarkerFaceColor','g', 'MarkerSize', 6);
plot(fuseT, Xtrue(1,fuseTimes) - Xf_trad(1,fuseTimes), 'mo', 'MarkerFaceColor','m', 'MarkerSize', 6);
% Mark calibration end
if calibrationPeriod <= N
    xline(t(calibrationPeriod), 'k--', 'LineWidth', 2, 'Alpha', 0.7);
    text(t(calibrationPeriod), max(ylim)*0.8, 'Calibration Ends', 'Rotation', 90, 'FontSize', 12);
end
xlabel('Time [s]'); ylabel('x-error [m]');
title('Fusion Methods Comparison: X-Position Error');
legend('Calibrated FKF', 'Traditional FKF', 'Centralized KF', 'Cal. Fusion Pts', 'Trad. Fusion Pts', 'Location', 'best');

% Y-position fusion comparison
subplot(2,1,2); hold on; grid on;
plot(t, Xtrue(2,:) - Xf_calib(2,:), 'g-', 'LineWidth', 2);
plot(t, Xtrue(2,:) - Xf_trad(2,:), 'm-', 'LineWidth', 2);
plot(t, Xtrue(2,:) - Xc(2,:), 'y-', 'LineWidth', 1.5);
% Mark fusion points
plot(fuseT, Xtrue(2,fuseTimes) - Xf_calib(2,fuseTimes), 'go', 'MarkerFaceColor','g', 'MarkerSize', 6);
plot(fuseT, Xtrue(2,fuseTimes) - Xf_trad(2,fuseTimes), 'mo', 'MarkerFaceColor','m', 'MarkerSize', 6);
% Mark calibration end
if calibrationPeriod <= N
    xline(t(calibrationPeriod), 'k--', 'LineWidth', 2, 'Alpha', 0.7);
    text(t(calibrationPeriod), max(ylim)*0.8, 'Calibration Ends', 'Rotation', 90, 'FontSize', 12);
end
xlabel('Time [s]'); ylabel('y-error [m]');
title('Fusion Methods Comparison: Y-Position Error');
legend('Calibrated FKF', 'Traditional FKF', 'Centralized KF', 'Cal. Fusion Pts', 'Trad. Fusion Pts', 'Location', 'best');

sgtitle(sprintf('Fusion Performance (Every %d steps, Sensor intervals [%s])', ...
    fusionInterval, num2str(sensorIntervals)), 'FontSize', 14, 'FontWeight', 'bold');

%% Plot 3: Error Magnitude and Statistics
figure('Position', [300, 300, 1400, 800]);

% Combined error magnitudes
subplot(2,2,1); hold on; grid on;
error_calib = sqrt((Xtrue(1,:) - Xf_calib(1,:)).^2 + (Xtrue(2,:) - Xf_calib(2,:)).^2);
error_trad = sqrt((Xtrue(1,:) - Xf_trad(1,:)).^2 + (Xtrue(2,:) - Xf_trad(2,:)).^2);
error_central = sqrt((Xtrue(1,:) - Xc(1,:)).^2 + (Xtrue(2,:) - Xc(2,:)).^2);

plot(t, error_calib, 'g-', 'LineWidth', 2);
plot(t, error_trad, 'm-', 'LineWidth', 2);
plot(t, error_central, 'y-', 'LineWidth', 1.5);
% Mark calibration end
if calibrationPeriod <= N
    xline(t(calibrationPeriod), 'k--', 'LineWidth', 2, 'Alpha', 0.7);
    text(t(calibrationPeriod), max(ylim)*0.8, 'Cal. End', 'Rotation', 90, 'FontSize', 10);
end
xlabel('Time [s]'); ylabel('Position Error [m]');
title('Position Error Magnitude Comparison');
legend('Calibrated FKF', 'Traditional FKF', 'Centralized KF', 'Location', 'best');

% Running RMSE
subplot(2,2,2); hold on; grid on;
window_size = 100; % RMSE calculation window
rmse_calib = zeros(1, N);
rmse_trad = zeros(1, N);
rmse_central = zeros(1, N);

for i = window_size:N
    start_idx = max(1, i - window_size + 1);
    rmse_calib(i) = sqrt(mean(error_calib(start_idx:i).^2));
    rmse_trad(i) = sqrt(mean(error_trad(start_idx:i).^2));
    rmse_central(i) = sqrt(mean(error_central(start_idx:i).^2));
end

plot(t(window_size:end), rmse_calib(window_size:end), 'g-', 'LineWidth', 2);
plot(t(window_size:end), rmse_trad(window_size:end), 'm-', 'LineWidth', 2);
plot(t(window_size:end), rmse_central(window_size:end), 'y-', 'LineWidth', 1.5);
% Mark calibration end
if calibrationPeriod <= N
    xline(t(calibrationPeriod), 'k--', 'LineWidth', 2, 'Alpha', 0.7);
    text(t(calibrationPeriod), max(ylim)*0.8, 'Cal. End', 'Rotation', 90, 'FontSize', 10);
end
xlabel('Time [s]'); ylabel('Running RMSE [m]');
title(sprintf('Running RMSE (Window = %d steps)', window_size));
legend('Calibrated FKF', 'Traditional FKF', 'Centralized KF', 'Location', 'best');

% Error distribution histograms
subplot(2,2,3);
edges = 0:0.5:max([error_calib, error_trad, error_central]) + 1;
histogram(error_calib, edges, 'FaceColor', 'g', 'FaceAlpha', 0.7, 'Normalization', 'probability');
hold on;
histogram(error_trad, edges, 'FaceColor', 'm', 'FaceAlpha', 0.7, 'Normalization', 'probability');
histogram(error_central, edges, 'FaceColor', 'y', 'FaceAlpha', 0.7, 'Normalization', 'probability');
xlabel('Position Error [m]'); ylabel('Probability');
title('Error Distribution');
legend('Calibrated FKF', 'Traditional FKF', 'Centralized KF', 'Location', 'best');
grid on;

% Summary statistics
subplot(2,2,4);
methods = {'Calibrated FKF', 'Traditional FKF', 'Centralized KF'};
mean_errors = [mean(error_calib), mean(error_trad), mean(error_central)];
std_errors = [std(error_calib), std(error_trad), std(error_central)];
max_errors = [max(error_calib), max(error_trad), max(error_central)];

x_pos = 1:3;
bar_width = 0.25;

b1 = bar(x_pos - bar_width, mean_errors, bar_width, 'FaceColor', [0.2, 0.6, 0.8]);
hold on;
b2 = bar(x_pos, std_errors, bar_width, 'FaceColor', [0.8, 0.4, 0.2]);
b3 = bar(x_pos + bar_width, max_errors, bar_width, 'FaceColor', [0.6, 0.8, 0.2]);

set(gca, 'XTick', x_pos, 'XTickLabel', methods);
ylabel('Error [m]');
title('Error Statistics Summary');
legend('Mean', 'Std Dev', 'Max', 'Location', 'best');
grid on;
xtickangle(45);

% Add values on top of bars
for i = 1:3
    text(i - bar_width, mean_errors(i) + 0.1, sprintf('%.2f', mean_errors(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
    text(i, std_errors(i) + 0.1, sprintf('%.2f', std_errors(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
    text(i + bar_width, max_errors(i) + 0.1, sprintf('%.2f', max_errors(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end

sgtitle('Statistical Analysis of Filter Performance', 'FontSize', 14, 'FontWeight', 'bold');

%% Plot 4: Phase Analysis (Calibration vs Operational)
if calibrationPeriod <= N
    figure('Position', [400, 400, 1400, 600]);
    
    % Split data into calibration and operational phases
    calib_indices = 1:calibrationPeriod;
    oper_indices = (calibrationPeriod+1):N;
    
    % Calibration phase analysis
    subplot(2,2,1); hold on; grid on;
    plot(t(calib_indices), error_calib(calib_indices), 'g-', 'LineWidth', 2);
    plot(t(calib_indices), error_trad(calib_indices), 'm-', 'LineWidth', 2);
    plot(t(calib_indices), error_central(calib_indices), 'y-', 'LineWidth', 1.5);
    xlabel('Time [s]'); ylabel('Position Error [m]');
    title('Calibration Phase Performance');
    legend('Calibrated FKF', 'Traditional FKF', 'Centralized KF', 'Location', 'best');
    
    % Operational phase analysis
    subplot(2,2,2); hold on; grid on;
    if ~isempty(oper_indices)
        plot(t(oper_indices), error_calib(oper_indices), 'g-', 'LineWidth', 2);
        plot(t(oper_indices), error_trad(oper_indices), 'm-', 'LineWidth', 2);
        plot(t(oper_indices), error_central(oper_indices), 'y-', 'LineWidth', 1.5);
    end
    xlabel('Time [s]'); ylabel('Position Error [m]');
    title('Operational Phase Performance');
    legend('Calibrated FKF', 'Traditional FKF', 'Centralized KF', 'Location', 'best');
    
    % Phase comparison statistics
    subplot(2,2,3);
    if ~isempty(oper_indices)
        calib_stats = [mean(error_calib(calib_indices)), mean(error_trad(calib_indices)), mean(error_central(calib_indices))];
        oper_stats = [mean(error_calib(oper_indices)), mean(error_trad(oper_indices)), mean(error_central(oper_indices))];
        
        x_pos = 1:3;
        bar_width = 0.35;
        
        bar(x_pos - bar_width/2, calib_stats, bar_width, 'FaceColor', [0.7, 0.7, 0.9]);
        hold on;
        bar(x_pos + bar_width/2, oper_stats, bar_width, 'FaceColor', [0.9, 0.7, 0.7]);
        
        set(gca, 'XTick', x_pos, 'XTickLabel', methods);
        ylabel('Mean Error [m]');
        title('Phase Comparison: Mean Error');
        legend('Calibration Phase', 'Operational Phase', 'Location', 'best');
        grid on;
        xtickangle(45);
    end
    
    % Improvement analysis
    subplot(2,2,4);
    if ~isempty(oper_indices)
        improvement_vs_trad = (mean(error_trad(oper_indices)) - mean(error_calib(oper_indices))) / mean(error_trad(oper_indices)) * 100;
        improvement_vs_central = (mean(error_central(oper_indices)) - mean(error_calib(oper_indices))) / mean(error_central(oper_indices)) * 100;
        
        improvements = [improvement_vs_trad, improvement_vs_central];
        comparison_methods = {'vs Traditional FKF', 'vs Centralized KF'};
        
        bar(improvements, 'FaceColor', [0.4, 0.8, 0.4]);
        set(gca, 'XTickLabel', comparison_methods);
        ylabel('Improvement [%]');
        title('Calibrated FKF Performance Improvement');
        grid on;
        
        % Add percentage labels
        for i = 1:length(improvements)
            text(i, improvements(i) + sign(improvements(i))*1, sprintf('%.1f%%', improvements(i)), ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold');
        end
    end
    
    sgtitle('Calibration vs Operational Phase Analysis', 'FontSize', 14, 'FontWeight', 'bold');
end

%% Display numerical summary
fprintf('\n=== PERFORMANCE SUMMARY ===\n');
fprintf('Overall Mean Position Error:\n');
fprintf('  Calibrated FKF:  %.3f m\n', mean(error_calib));
fprintf('  Traditional FKF: %.3f m\n', mean(error_trad));
fprintf('  Centralized KF:  %.3f m\n', mean(error_central));

fprintf('\nOverall RMSE:\n');
fprintf('  Calibrated FKF:  %.3f m\n', sqrt(mean(error_calib.^2)));
fprintf('  Traditional FKF: %.3f m\n', sqrt(mean(error_trad.^2)));
fprintf('  Centralized KF:  %.3f m\n', sqrt(mean(error_central.^2)));

if calibrationPeriod <= N && ~isempty(oper_indices)
    fprintf('\nOperational Phase Performance:\n');
    fprintf('  Calibrated FKF:  %.3f m (RMSE: %.3f m)\n', ...
        mean(error_calib(oper_indices)), sqrt(mean(error_calib(oper_indices).^2)));
    fprintf('  Traditional FKF: %.3f m (RMSE: %.3f m)\n', ...
        mean(error_trad(oper_indices)), sqrt(mean(error_trad(oper_indices).^2)));
    fprintf('  Centralized KF:  %.3f m (RMSE: %.3f m)\n', ...
        mean(error_central(oper_indices)), sqrt(mean(error_central(oper_indices).^2)));
    
    improvement_vs_trad = (mean(error_trad(oper_indices)) - mean(error_calib(oper_indices))) / mean(error_trad(oper_indices)) * 100;
    improvement_vs_central = (mean(error_central(oper_indices)) - mean(error_calib(oper_indices))) / mean(error_central(oper_indices)) * 100;
    
    fprintf('\nImprovement in Operational Phase:\n');
    fprintf('  vs Traditional FKF: %.1f%%\n', improvement_vs_trad);
    fprintf('  vs Centralized KF:  %.1f%%\n', improvement_vs_central);
end