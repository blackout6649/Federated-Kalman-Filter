% DEMO: single-sensor KF sanity check (position-only measurements)
clear; clc; rng(0);

% Model: 2D constant velocity
dt = 0.1;
F  = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];
G  = [0.5*dt^2*eye(2); dt*eye(2)];
q  = 0.05;                 % accel noise spectral density
Q  = q*eye(2);
model = MotionModel(F,G,Q);

% Sensor: position only
H  = [1 0 0 0; 0 1 0 0];
sigma = 5;
R  = sigma^2 * eye(2);
sensor = LinearSensor(H,R,"PosSensor");

% Initial estimate
x0 = [0; 0; 1; 0.7];
P0 = diag([25 25 4 4]);

% Local KF
lkf = LocalKalmanFilter(model, sensor, x0, P0, "LKF1");

% Simulate truth
T = 40; N = round(T/dt);
Xtrue = zeros(4,N);
x = [0; 0; 1; 0.7];
for k=1:N
    a = sqrt(q)*randn(2,1);
    x = F*x + G*a;
    Xtrue(:,k) = x;
end

% Run filter
Xhat = zeros(4,N); Z = zeros(2,N); nis = zeros(1,N);
for k=1:N
    % measurement
    z = sensor.measure(Xtrue(:,k));
    Z(:,k) = z;

    % filter step
    lkf.predict();
    lkf.update(z);
    Xhat(:,k) = lkf.x;
    % NIS
    v = lkf.lastInnov; S = lkf.lastS;
    nis(k) = v'*(S\v);
end

% RMSE
pos_err = Xhat(1:2,:) - Xtrue(1:2,:);
rmse = sqrt(mean(pos_err.^2, 2));

fprintf('Position RMSE: [%.2f, %.2f] m\n', rmse(1), rmse(2));
fprintf('Mean NIS (should be ~2): %.2f\n', mean(nis));

% Plots
figure; plot(Xtrue(1,:), Xtrue(2,:), 'k-', 'LineWidth',1.5); hold on;
plot(Xhat(1,:),  Xhat(2,:),  'b--');
legend('Truth','Estimate'); axis equal; grid on; title('Trajectory');

figure; plot(nis,'-'); yline(2,'--');
grid on; title('NIS'); xlabel('k'); ylabel('\nu^T S^{-1} \nu');
