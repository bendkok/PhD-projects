%
% Inputs for the grid
%
Rmax=200;        % Box size
lmax=2;         % Number of partial waves
N=2000;          % Radial grid points 

%
% Input for the absorber
%
eta=0;           % Strength
%Onset=0.6*Rmax;
Onset=30;   
AbsPower=2;         % Power of the monimial

% Input for the Yukawa potential
Alpha=0.0;

%
% Inputs for the interaction
%
Ncycle=10;              % # optical cycles
E0=1;                  % Maximum electric field strength
w=.2;                   % Central frequency
cep=0;                  % Carrier-envelope phase
Textra=1500;             % Propagation after pluse 
% Derivative:
Tpulse=Ncycle*2*pi/w;   % Length of pulse - in time units


%
% Logical inputs
%
% Wether to plott on the fly on not
PlotOrNot = logical(0);   

%
% Numerical inputs for the propagation:
%
% Numerical time step
%dt=0.05;               
% Dimension of Krylov space for the Lanczos propagator(fixed):
KrylovDim=15;