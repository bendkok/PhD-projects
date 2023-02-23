function N=NormMatrixPsi(Psi,h)

% Numerical evaluation of the norm \sqrt{<Psi | Psi>}.
% We use the subroutine InnerProduct.m in order to
% calculate <Psi|Psi>.

N = sqrt(InnerProduct(Psi,Psi,h));
