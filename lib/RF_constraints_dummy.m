function [F,J,H] = RF_constraints_dummy(b_coeff,VOP,Nch,power_factor_s,output)
%[F,J,H] = RF_constraints_dummy(b_coeff,VOP,Nch,power_factor_s,output)
%   Dummy function for the hardware and SAR constraints. This mimicks what
%   happens in the vendor supplied safety model which cannot be shared
%   without permission. This function only handles RF safety, with gradient
%   limits being handled by fmincon linear constraints.
% 
%   INPUTS:
%       b_coeff        -> column array with RF coefficients (units: V)
%       VOP            -> 3D array with all "VOPs" concatenated
%       Nch            -> no. of transmit channels
%       power_factor_s -> integral of squared normalized RF waveform (units: s)
%       output         -> string that determines the output; should contain
%                        3 digits, either 0 or 1 that determine the output. 
%                        For example, with '101' J and H are calculated but not J
%   OUTPUTS:
%       F              -> value of the non-linear constraints
%       J              -> Jacobian of F
%       H              -> cell containing the Hessian of each constraint
%
% David Leitao 2024, King's College London (david.leitao@kcl.ac.uk)


b_coeff = reshape(b_coeff,[],Nch);    
Nsp     = size(b_coeff,1);            
Nvop    = size(VOP,3);                

if output(1)=='1'
    F = zeros(Nvop+Nch*Nsp,1);
    num = 0;
    for k = 1:Nvop
        num = num + 1;
        F(num) = real(power_factor_s * sum(diag(b_coeff * VOP(:,:,k) * b_coeff')));  
    end
    for k = 1:Nch*Nsp
        num = num + 1;
        F(num) = sqrt(real(b_coeff(k) * conj(b_coeff(k))));
    end
else
    F = [];
end

if output(2)=='1'
    J = zeros(2*Nch*Nsp,Nvop);
    num = 0;
    for k = 1:Nvop
        num = num + 1;
        for p = 1:Nsp
            J(p:Nsp:Nch*Nsp,num)           = 2 * power_factor_s * VOP(:,:,k) * real(b_coeff(p,:))';
            J(Nch*Nsp+p:Nsp:2*Nch*Nsp,num) = 2 * power_factor_s * VOP(:,:,k) * imag(b_coeff(p,:))';
        end
    end
    J = cat(2,J,[diag(real(b_coeff(:))./abs(b_coeff(:))); ...
                 diag(imag(b_coeff(:))./abs(b_coeff(:)))]);
else
    J = [];
end

if output(3)=='1'
    H = cell(Nvop+Nch*Nsp,1);
    for k = 1:Nvop
        H{k} = zeros(2*Nch*Nsp);
        for p = 1:Nsp
            H{k}(p:Nsp:Nch*Nsp,p:Nsp:Nch*Nsp)                     = 2 * real(power_factor_s * VOP(:,:,k));
            H{k}(Nch*Nsp+p:Nsp:2*Nch*Nsp,Nch*Nsp+p:Nsp:2*Nch*Nsp) = 2 * real(power_factor_s * VOP(:,:,k));
        end
    end
    num = 0;
    for k = Nvop+1:Nvop+Nch*Nsp
        num = num + 1;
        H{k} = zeros(2*Nsp*Nch);
        H{k}(num,num)                 = (imag(b_coeff(num))^2 + 1i*real(b_coeff(num))*imag(b_coeff(num))) / abs(b_coeff(num))^3;
        H{k}(num,Nsp*Nch+num)         = 1i*(imag(b_coeff(num))^2 + 1i*real(b_coeff(num))*imag(b_coeff(num))) / abs(b_coeff(num))^3;
        H{k}(Nsp*Nch+num,num)         = 1i*(imag(b_coeff(num))^2 + 1i*real(b_coeff(num))*imag(b_coeff(num))) / abs(b_coeff(num))^3;
        H{k}(Nsp*Nch+num,Nsp*Nch+num) = (real(b_coeff(num))^2 + 1i*real(b_coeff(num))*imag(b_coeff(num))) / abs(b_coeff(num))^3;
    end
else
    H = [];
end

end

