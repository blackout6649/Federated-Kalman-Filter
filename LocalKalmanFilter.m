classdef LocalKalmanFilter < handle
    % A local KF tied to one motion model and one sensor
    properties
        model  % MotionModel
        sensor % LinearSensor
        x     % Current estimate
        P     % Current covariance
        lastInnov
        lastS
        name
    end 
    methods 
        function obj = LocalKalmanFilter(model, sensor, x0, P0, weight, name)
            obj.model = model.FKFaug(weight); obj.sensor = sensor;
            obj.x = x0; obj.P = P0 * weight;
            if nargin < 5, name = "LocalKF"; end
            obj.name = name;
        end 
        function predict(obj)
            [obj.x, obj.P] = obj.model.predict(obj.x, obj.P);
        end 
        function update(obj, z)
            H = obj.sensor.H; R = obj.sensor.R;
            S = H * obj.P * H' + R;   % Innovation Matrix
            K = obj.P * H' / S;       % Kalman Gain
            innov = z - H * obj.x;
            obj.x = obj.x + K * innov;
            % Joseph form for numerical symmetry
            I = eye(size(obj.P));
            obj.P = (I - K*H) * obj.P * (I - K*H)' + K * R * K';
            obj.lastInnov = innov;
            obj.lastS = S;
        end 
        function [x, P] = estimate(obj)
            x = obj.x; P = obj.P;
        end
    end 
end 