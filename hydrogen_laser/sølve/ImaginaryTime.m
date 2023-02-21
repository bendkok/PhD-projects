
% Initial l-quantum number
lInit=0;

% Construct initial wavefunction within one l-channel
PsiCol=rand(N,1); PsiCol=PsiCol/(PsiCol'*PsiCol)/h;

% Numerical time-step for propagation in imaginary time
dtImag=0.01;
t=0;
tMax=200;    % Duration of propagation

% Propagator - specific to one l-channel
SentrifugalTerm=lInit*(lInit+1)/2*diag(1./rVector.^2);
U=expm(-(H0rad+SentrifugalTerm)*dtImag);

while t<tMax
  PsiCol=U*PsiCol;              % One step
  Norm=PsiCol'*PsiCol*h;        % New norm 
  PsiCol=PsiCol/sqrt(Norm);     % Renormalize wave function
  t=t+dtImag;                   % Update time
end

% Plot final wave function 
% If we start out in the ground state, we 
% also plot the analytical solution for comparison
figure(1)
plot(rVector,abs(PsiCol).^2)    
if lInit==0
  hold on
  PsiAnalytical=rVector.*exp(-rVector);
  NormAnalytical=sum(abs(PsiAnalytical).^2)*h;
  PsiAnalytical=PsiAnalytical/sqrt(NormAnalytical);
  plot(rVector,abs(PsiAnalytical).^2,'r--')
  hold off
  Discrepancy=trapz(rVector,(PsiCol-PsiAnalytical).^2)
end

% Finalize
E=-1/2/dtImag*log(Norm)             % Final energy
Psi=zeros(N,lmax+1);
Psi(:,lInit+1)=PsiCol;              % Initial state (full)

