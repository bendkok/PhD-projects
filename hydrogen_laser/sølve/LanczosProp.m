function PsiNew=LanczosProp(H0rad,Sr,Sl,rVector,Gamma,...
Pz,AngularOp,AngularOp2,E0,w,Ncycle,cep,t,Psi,n,dt,h)

% This routine implements the Lanczos propagator
% and uses it to propagate the state Psi a time-step
% dt. n is the dimension of the Krylov subspace.
% The action of the Hamiltonian is provided by
% the subroutine HamMult.m.
% H0rad, Sr and Sl are related to the Hamiltonian.
% Psi is the state.
% dt is the time-step. (For imaginary time this should be
% set to -i times the actual dt in the call.)
% h is the radial increment in the grid.

% Allocation
[Rows Cols]=size(Psi);
Vbig=zeros(Rows,Cols,n);
Beta=zeros(1,n+1);
Alfa=zeros(1,n);
T=zeros(n,n);

% Construct Arnoldi basis and Arnoldi Hamiltonian
InitialNorm=NormMatrixPsi(Psi,h); % In case of non-Hermcity: Perserver norm

Vbig(:,:,1)=Psi/InitialNorm;                            % First state
for j=1:n;
  V=Vbig(:,:,j);
% Apply Hamiltonian
  U=HamMultUnpert(H0rad,Sr,Sl,Gamma,V);
% Length gauge
%  U=U+HamMultPertLG(rVector,AngularOp,V,...
%  E0,w,Ncycle,cep,t);
% Velocity gauge
  U=U+HamMultPertVG(rVector,Pz,AngularOp,AngularOp2,...
      V,E0,w,Ncycle,cep,t);  
%
% Remove component of V_{j-1}
  if j>1
    U=U-Beta(j)*Vbig(:,:,j-1);
  end
  Alfa(j)=InnerProduct(V,U,h);
  U=U-Alfa(j)*V;
  Beta(j+1)=NormMatrixPsi(U,h);
  Vbig(:,:,j+1)=U/Beta(j+1);  
end

T=diag(Alfa);
T=T+diag(Beta(2:n),1)+diag(Beta(2:n),-1);

% Exponentiate Hamiltonian to construct approximate propagator
Ulanczos=expm(-1i*dt*T);
% Apply to Psi - extract first column
Ulanczos=Ulanczos(:,1);

% Reconstruct propagated state
PsiNew=zeros(Rows,Cols);
for k=1:n;
  PsiNew=PsiNew+Ulanczos(k)*Vbig(:,:,k);
end
% Renormalize
PsiNew=PsiNew*InitialNorm;