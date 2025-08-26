classdef MotionModel
   % Linear, time invariant motion model xk1 = Fx + Gw, w ~ N(0, Q)
   properties
       F     % State Transition matrix
       G     % Process noise distribution matrix
       Q     % Process noise covariance matrix
   end 
    
   methods
       function obj = MotionModel(F, G, Q)
           obj.F = F; obj.G = G; obj.Q = Q;
       end 
       function [xk1, Pk1] = predict(obj, x, P)
           xk1 = obj.F * x; % Time propogation of state estimation
           Pk1 = obj.F * P * obj.F' + obj.G * obj.Q * obj.G'; % Time propogation of covariance 
       end 
   end 
end