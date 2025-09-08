classdef MotionModel < handle
   % Linear, time invariant motion model xk1 = Fx + Gw, w ~ N(0, Q)
   properties
       F     % State Transition matrix
       G     % Process noise distribution matrix
       Q     % Process noise covariance matrix
       weight % Scalar weight value for FKF augmentation
   end 
    
   methods
       function obj = MotionModel(F, G, Q, weight)
           if nargin < 4, weight = 1; end
           obj.F = F; obj.G = G; obj.Q = Q; obj.weight = weight;
       end 
       function newObj = FKFaug(obj, weight)
           newObj = MotionModel(obj.F, obj.G, obj.Q);
           newObj.F = obj.F;
           newObj.G = obj.G;
           newObj.Q = obj.Q * weight;
       end 
       function [xk1, Pk1] = predict(obj, x, P)
           xk1 = obj.F * x; % Time propogation of state estimation
           Pk1 = obj.F * P * obj.F' + obj.G * obj.Q * obj.G'; % Time propogation of covariance 
       end 
   end 
end