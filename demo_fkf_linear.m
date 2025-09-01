% DEMO: Federated KF with two linear position sensors of different noise

% Version 1.0
% Develop Branch

clear; clc; rng(1);

% ===== Frequencies =====
sensorIntervals = [1, 5]; % Sensor 1 every 1 step, sensor 2 every 5 steps
fusionInterval = 10;       % Fuse every 5 steps
N = length(sensorIntervals); % Number of sensors
weight = 1 / N;

% ===== Model =====
dt = 0.1;
F  = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];
G  = [0.5*dt^2*eye(2); dt*eye(2)];
q  = 0.05;
Q  = q*eye(2);
model = MotionModel(F,G,Q);

% ===== Sensors (both linear position, different accuracies) =====
H = [1 0 0 0; 0 1 0 0];
R1 = (5^2)*eye(2);   s1 = LinearSensor(H,R1, sensorIntervals(1), "PosSensor-5m");
R2 = (8^2)*eye(2);   s2 = LinearSensor(H,R2, sensorIntervals(2),"PosSensor-8m");

% ===== Locals =====
x0 = [0; 0; 1; 0.6]; P0 = diag([25 25 4 4]);
lkf1 = LocalKalmanFilter(model, s1, x0, P0, weight, "LKF1");
lkf2 = LocalKalmanFilter(model, s2, x0, P0, weight, "LKF2");

% ===== Federated manager =====
fkf = FederatedKF([lkf1, lkf2], [], "FKF");

% ===== Truth sim =====
T = 600; N = round(T/dt);
Xtrue = zeros(4,N);
x = [0; 0; 1; 0.6];
for k=1:N
    a = sqrt(q)*randn(2,1);
    x = F*x + G*a;
    Xtrue(:,k) = x;
end

% ===== Run =====
Xhat1 = zeros(4,N); Xhat2 = zeros(4,N); Xf = zeros(4,N);
for k=1:N
    z1 = s1.measure(Xtrue(:,k), k);
    z2 = s2.measure(Xtrue(:,k), k);
    fuseFlag  = mod(k, fusionInterval) == 0;
    fkf.step({z1, z2}, fuseFlag);         % update both locals + fuse
    Xhat1(:,k) = lkf1.x;
    Xhat2(:,k) = lkf2.x;
    Xf(:,k)    = fkf.x;
end

% ===== Evaluation =====
rmse1 = sqrt(mean((Xhat1(1:2,:)-Xtrue(1:2,:)).^2,2));
rmse2 = sqrt(mean((Xhat2(1:2,:)-Xtrue(1:2,:)).^2,2));
rmsef = sqrt(mean((Xf(1:2,:)-Xtrue(1:2,:)).^2,2));

fprintf('Local 1 RMSE [x,y]: [%.2f, %.2f] m\n', rmse1(1), rmse1(2));
fprintf('Local 2 RMSE [x,y]: [%.2f, %.2f] m\n', rmse2(1), rmse2(2));
fprintf('Fused   RMSE [x,y]: [%.2f, %.2f] m\n', rmsef(1), rmsef(2));

figure; hold on; grid on; axis equal;
plot(Xtrue(1,:),Xtrue(2,:),'k-','LineWidth',1.5);
plot(Xhat1(1,:),Xhat1(2,:),'b--');
plot(Xhat2(1,:),Xhat2(2,:),'r-.');
plot(Xf(1,:),Xf(2,:),'g-');
legend('Truth','Local 1','Local 2','Fused');
title('Federated KF (linear sensors)');

%% ===== Plotting clarity =====
t = (0:N-1) * dt;   % time vector
fuseTimes = find(mod(1:N, fusionInterval) == 0); % fusion step indices
fuseT = t(fuseTimes);  % fusion timestamps

figure;

% Position error in x
subplot(2,1,1); hold on; grid on;
plot(t, Xtrue(1,:) - Xhat1(1,:), 'b--');
plot(t, Xtrue(1,:) - Xhat2(1,:), 'r-.');
plot(t, Xtrue(1,:) - Xf(1,:),    'g-');
% Mark fusion points
plot(fuseT, Xtrue(1,fuseTimes) - Xf(1,fuseTimes), 'ko', 'MarkerFaceColor','k', 'MarkerSize',6);
xlabel('Time [s]'); ylabel('x-error [m]');
legend('Local 1','Local 2','Fused','Fusion points');
title('Estimation Error in x-position');

% Position error in y
subplot(2,1,2); hold on; grid on;
plot(t, Xtrue(2,:) - Xhat1(2,:), 'b--');
plot(t, Xtrue(2,:) - Xhat2(2,:), 'r-.');
plot(t, Xtrue(2,:) - Xf(2,:),    'g-');
% Mark fusion points
plot(fuseT, Xtrue(2,fuseTimes) - Xf(2,fuseTimes), 'ko', 'MarkerFaceColor','k', 'MarkerSize',6);
xlabel('Time [s]'); ylabel('y-error [m]');
legend('Local 1','Local 2','Fused','Fusion points');
title('Estimation Error in y-position');

sgtitle(sprintf('Fusion every %d steps | Sensor intervals = [%s]', ...
    fusionInterval, num2str(sensorIntervals)));