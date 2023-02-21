% Dette skriptet propagerar bølgefunksjonen med fleire
% verdiar for dt.
% Etterpå sjekkar den slutt-normen - til bølgefunksjonen og
% til differansen med slutt-tilstanden vi fekk med førre dt.

clear

Inputs                  % Provide input parameters (hard coded in separate script)

SetUpH                  % Set up the Hamiltonian

% Construct initial state by imaginary time 
% (The inputs are hard coded within the subroutine)
disp('Constructing initial state by propagation in imaginary time')
ImaginaryTime
disp('Done')

% Save initial state
Psi0 = Psi;
InitialNormDeficiency = 1-NormMatrixPsi(Psi0, h)

% Set final time and initial time step.
% Case 1
%Tmax = 10;
%Nstep0 = 25;
% Case 2
Tmax = 50;
Nstep0 = 1000;
dt0 = Tmax/Nstep0;

% Fix the number of iteratons - with Nt doubling each time
Nit = 4;

% Allocate vectors with data
NormVector = zeros(1, Nit);
NormDiff = zeros(1, Nit-1);

% Initiate time step
Nstep = Nstep0;
dt = dt0;
for counter = 1:Nit;
  NstepVec(counter) = Nstep;
  % Copy old wave function
  if counter > 1
    PsiPrev = Psi;
  end
  % Initiate new wave function
  Psi = Psi0;               % Copy initial state
  t = 0;                    % Initiate time
  % Propagate
  for n = 1:Nstep    
    Psi=LanczosProp(H0rad,Sr,Sl,rVector,Gamma,Pz,...
    AngularOp,AngularOp2,E0,w,Ncycle,cep,t+dt/2,Psi,KrylovDim,dt,h);
    t=t+dt;                 % Update time
  end
  % 
  % Plot final wave fuction
  figure(2)
  for ll = 0:2
    subplot(3, 1, ll+1)
    plot(rVector,abs(Psi(:,ll+1)).^2)
    V = axis; axis([0 50 V(3) V(4)])
    hold on
  end
  % Update vector with norms
  NormVector(counter) = NormMatrixPsi(Psi, h)
  %
  % Update vector with the difference with previous estimate
  if counter > 1
    NormDiff(counter-1) = NormMatrixPsi(Psi-PsiPrev, h)
  end
  % Update time step
  dt = dt/2;
  Nstep = 2*Nstep;
end
hold off

% Plot convergence
figure(3)
% Plot error in norm
subplot(2,1,1)
loglog(NstepVec, abs(1-NormVector), 'b+-')
xlabel('Steps'); ylabel('|1 - |\Psi(T)|');
subplot(2,1,2)
% Plot difference between consequitve estimates
loglog(NstepVec(2:end), NormDiff, 'kx-')
xlabel('Steps, N'); ylabel('| \Psi_N(T)-\Psi_{2N}(T)|');
hold on
% Plot \Delta t^2 - for comparison 
loglog(NstepVec(2:end), 1e-2*NstepVec(2:end).^(-2), '--')
legend('Norm of difference', '\Delta t^2')
hold off
