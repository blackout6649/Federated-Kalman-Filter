% DEMO: Federated KF with two linear position sensors of different noise
% Version 1.1
% Develop
clear; clc; rng(1); close all;

% ===== Frequencies =====
sensorIntervals = [1, 1, 1]; % Sensor 1 every 1 steps, sensor 2 every 1 steps, reference every 1 steps
fusionInterval = 10;       % Fuse every 10 steps
weight = length(sensorIntervals); % Number of sensors

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
fkf_Calib = FederatedKF([lkf1_calib, lkf2_calib], lkfRef_calib, 1000, "Calib-FKF", 0.05, 10, 2, 0.8);

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
    
    % --- Update Calibrated FKF ---
    % z_cell for calibrated FKF only includes the two local filters
    fkf_Calib.step({z1, z2}, zref, fuseFlag);         
    Xhat1_calib(:,k) = lkf1_calib.x;
    Xhat2_calib(:,k) = lkf2_calib.x;
    XhatRef(:,k) = lkfRef_calib.x;
    Xf_calib(:,k) = fkf_Calib.x;

    % --- Update Traditional FKF ---
    % z_cell for traditional FKF includes all three local filters
    fkf_Trad.step({z1, z2, zref}, fuseFlag);
    Xhat1_trad(:,k) = lkf1_trad.x;
    Xhat2_trad(:,k) = lkf2_trad.x;
    Xf_trad(:,k) = fkf_Trad.x;

    % --- Centralized KF step ---
    ckf.step({z1, z2, zref});
    Xc(:,k) = ckf.x;
end

%% Visual Aid
% ===== Evaluation =====
rmse1_calib = sqrt(mean((Xhat1_calib(1:2,:)-Xtrue(1:2,:)).^2,2));
rmse2_calib = sqrt(mean((Xhat2_calib(1:2,:)-Xtrue(1:2,:)).^2,2));
rmsef_calib = sqrt(mean((Xf_calib(1:2,:)-Xtrue(1:2,:)).^2,2));

rmse1_trad = sqrt(mean((Xhat1_trad(1:2,:)-Xtrue(1:2,:)).^2,2));
rmse2_trad = sqrt(mean((Xhat2_trad(1:2,:)-Xtrue(1:2,:)).^2,2));
rmsef_trad = sqrt(mean((Xf_trad(1:2,:)-Xtrue(1:2,:)).^2,2));

rmseRef = sqrt(mean((XhatRef(1:2,:)-Xtrue(1:2,:)).^2,2));
rmsec = sqrt(mean((Xc(1:2,:)-Xtrue(1:2,:)).^2,2));

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

legend('Truth','Sensor 1 meas','Sensor 2 meas','Ref. Sensor meas','FKF (Calibrated)','FKF (Traditional)','CKF');
title('Federated vs Centralized KF');
xlabel("X - Coordinate"); ylabel("Y - Coordinate")

%% ===== Plotting clarity =====
t = (0:N-1) * dt;   % time vector
fuseTimes = find(mod(1:N, fusionInterval) == 0); % fusion step indices
fuseT = t(fuseTimes);  % fusion timestamps

figure;
% Position error in x
subplot(2,1,1); hold on; grid on;
plot(t, Xtrue(1,:) - Xhat1_calib(1,:), 'b--');
plot(t, Xtrue(1,:) - Xhat2_calib(1,:), 'r-.');
plot(t, Xtrue(1,:) - Xf_calib(1,:),    'g-');
plot(t, Xtrue(1,:) - Xf_trad(1,:),    'm-');
% Mark fusion points
plot(fuseT, Xtrue(1,fuseTimes) - Xf_calib(1,fuseTimes), 'ko', 'MarkerFaceColor','g', 'MarkerSize',6);
plot(fuseT, Xtrue(1,fuseTimes) - Xf_trad(1,fuseTimes), 'ks', 'MarkerFaceColor','m', 'MarkerSize',6);
xlabel('Time [s]'); ylabel('x-error [m]');
legend('Local 1','Local 2','Fused (Calibrated)','Fused (Traditional)','Fusion points (Calib)','Fusion points (Trad)');
title('Estimation Error in x-position');

% Position error in y
subplot(2,1,2); hold on; grid on;
plot(t, Xtrue(2,:) - Xhat1_calib(2,:), 'b--');
plot(t, Xtrue(2,:) - Xhat2_calib(2,:), 'r-.');
plot(t, Xtrue(2,:) - Xf_calib(2,:),    'g-');
plot(t, Xtrue(2,:) - Xf_trad(2,:),    'm-');
% Mark fusion points
plot(fuseT, Xtrue(2,fuseTimes) - Xf_calib(2,fuseTimes), 'ko', 'MarkerFaceColor','g', 'MarkerSize',6);
plot(fuseT, Xtrue(2,fuseTimes) - Xf_trad(2,fuseTimes), 'ks', 'MarkerFaceColor','m', 'MarkerSize',6);
xlabel('Time [s]'); ylabel('y-error [m]');
legend('Local 1','Local 2','Fused (Calibrated)','Fused (Traditional)','Fusion points (Calib)','Fusion points (Trad)');
title('Estimation Error in y-position');
sgtitle(sprintf('Fusion every %d steps | Sensor intervals = [%s]', ...
    fusionInterval, num2str(sensorIntervals)));