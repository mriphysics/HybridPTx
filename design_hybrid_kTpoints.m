function [fval, b_opt, k_opt] = design_hybrid_kTpoints(alpha_des,beta_des,lambda,optim_struct)
%[fval, b_opt, k_opt] = design_hybrid_kTpoints(alpha_des,beta_des,lambda,optim_struct)
%   Optimizes a kT-points pulse using the hybrid design under the STA for
%   the flip angle calculation.
%   INPUTS:
%       alpha_des      -> column array containing the desired flip angle
%       beta_des       -> column array containing the desired B1rms
%       lambda         -> balancing parameter of hybrid design
%       optim_struct.* -> optimization structure with the following fields:
%         *.pos               -> matrix with spatial coordinates (dimensions: Nvoxels x Naxis; units: meters)
%         *.b0                -> column array with off-resonance values (dimensions: Nvoxels x 1; units: mT).
%         *.tx                -> matrix with transmit sensitivities (dimensions: Nvoxels x Nch; units: mT/V). 
%         *.weights           -> column array with voxels' weights (dimensions: Nvoxels x 1).    
%         *.VOP               -> VOP/average power matrices
%         *.constraint_limits -> constraint limits for dummy function
%         *.verbose           -> controls verbose of fmincon
%         *.maxiter           -> maximum fmincon iterations
%         *.quality           -> controls stopping conditions of fmincon
%         *.multistart        -> enable/disbale multi-starts
%         *.multistart_trials -> no. of multi-starts if enabled
%         *.smax              -> maximum slew rate (units: mT/m/ms)
%         *.gmax              -> maximum gradient amplitude (mT/m)
%         *.Nsp               -> no. of kT-points subpulses
%         *.subpulse_length_s -> duration of each RF sub-pulse (units: s)
%         *.subpulse_gap_s    -> duration of each gradient blip (units: s)
%         *.power_factor_s    -> integral of squared normalized RF waveform (units: s)
%         *.x0                -> cell containing multiple starting points or array with single starting point
%   OUTPUTS:
%       fval                  -> final cost function value (best of all multi-starts)
%       b_opt                 -> matrix with optimized RF coefficients (dimensions: Nsp x Nch; units: V)
%       k_opt                 -> matrix with optimized excitation k-space (dimensions: Naxis x Nsp; units: rad/m) 
%
% David Leitao 2024, King's College London (david.leitao@kcl.ac.uk)

gamma_uT = 267.522119; % units rad s^-1 uT^-1

pos = optim_struct.pos;
b0 = optim_struct.b0;
tx = optim_struct.tx;

Nch = size(tx,2); %no. transmit channels
Nsp = optim_struct.Nsp; %no. kT-points RF sub-pulses

Ttotal_s = Nsp*(optim_struct.subpulse_length_s+optim_struct.subpulse_gap_s); %duration of RF pulse [s] - note it includes gradient blips
Trf_s    = (0:Nsp-1)*(optim_struct.subpulse_length_s+optim_struct.subpulse_gap_s) + optim_struct.subpulse_length_s/2; %[s]

dt = optim_struct.subpulse_length_s;
T = Nsp * dt;

%%% Ensure weights is a column vector
W = optim_struct.weights(:).';

%%% Pre-calculate off-resonance term (constant for fixed vector t)
A1 = 1i * (1e3*gamma_uT) * dt .* exp(1i * (1e3*gamma_uT) * b0 * (Trf_s-Ttotal_s));    
Afull = cell(Nch,1);

%%% Auxiliar matrices for cost function & derivatives calculations:
% ri2cp -> transforms real/imag array into complex array
% idx_bfull -> stores indexes to expand RF voltages into a matrix
% fullTx -> matrix with real/imag of all TX channels repeated per sub-pulse
ri2cp = [eye(Nch*Nsp), 1i*eye(Nch*Nsp)]; 
idx_bfull = 1:Nsp*Nch+1:Nsp^2*Nch; for cc=1:Nch-1; idx_bfull = cat(2,idx_bfull,Nsp*cc + (1:Nsp*Nch+1:Nsp^2*Nch)); end
fullTx = []; for cc=1:Nch; fullTx = cat(2,fullTx,repmat(tx(:,cc),[1 Nsp])); end
fullTx = [fullTx, 1i*fullTx];

%% Set optimisation parameters

%%% Intialisation
if isempty(optim_struct.x0)
    CPmode = exp(-1i*2*pi*(0:Nch-1).'/Nch)/sqrt(Nch);
    %Acp/Bcp are the flip angle/B1rms distributions generated by CP mode @ 1V
    Acp = 180/pi*gamma_uT*Nsp*dt*abs(1e3*tx*CPmode);
    Bcp = sqrt(Nsp*dt/T)*abs(1e3*tx*CPmode);
    %analytical solution that minimizes hybrid cost function:
    base_voltage = ((1-lambda)*mean(Acp)/alpha_des(1) + lambda*mean(Bcp)/beta_des(1)) / ...
        ((1-lambda)*mean(Acp.^2)/alpha_des(1).^2 + lambda*mean(Bcp.^2)/beta_des(1).^2);
    if optim_struct.multistart
        x0 = cell(optim_struct.multistart_trials,1);
        for tt=1:optim_struct.multistart_trials
            x0{tt} = rand_init_kTpoints(base_voltage,Nsp,Nch,optim_struct.smax,optim_struct.gmax,optim_struct.subpulse_gap_s);
        end
    else
        x0 = cat(1,base_voltage*rand(2*Nch*Nsp,1),zeros(3*Nsp,1));
    end
else
    x0 = optim_struct.x0;
    if iscell(x0) && numel(x0)>1
        optim_struct.multistart        = true;
        optim_struct.multistart_trials = numel(x0);
    else
        optim_struct.multistart        = false;
    end
end

%denominators of the two cost function terms
alpha_norm = norm(alpha_des).^2;
beta_norm  = norm(beta_des).^2; 

f =@(x) cost(x,lambda,alpha_des,beta_des);
constr =@(x) nonlincon(x,optim_struct.VOP,optim_struct.power_factor_s,optim_struct.constraint_limits);

options = optimoptions('fmincon',...
                       'Display',optim_struct.verbose,... 
                       'Algorithm','interior-point',...
                       'MaxFunctionEvaluations',Inf,...
                       'MaxIterations',optim_struct.maxiter,...
                       'StepTolerance',10^(-optim_struct.quality),...
                       'FunctionTolerance',10^(-optim_struct.quality),...
                       'UseParallel',false,...
                       'SpecifyObjectiveGradient',true,...
                       'SpecifyConstraintGradient',true,...
                       'FiniteDifferenceType','central',...  
                       'CheckGradients',false,...
                       'Hessian','user-supplied', ...
                       'HessFcn',@(x, lag) hessian_function(x,lag,lambda,alpha_des,beta_des,optim_struct.VOP,optim_struct.power_factor_s));


%slew limited maximum gradient amplitude for the gradient blip duration:
gmax_slew = optim_struct.smax * (1e3*optim_struct.subpulse_gap_s)/2; %[mT/m]
if gmax_slew>optim_struct.gmax
	Gmax = optim_struct.gmax; %maximum gradient limited by gradient amplitude
else
	Gmax = gmax_slew; %maximum gradient limited by slew
end
%integral of gradient blip at maximum amplitude
garea_max = Gmax * optim_struct.subpulse_gap_s /2; %[mT.s/m]


auxAcons = zeros(Nsp,Nsp);
auxAcons(1:Nsp+1:end) = -1; auxAcons(Nsp+1:Nsp+1:end) = 1;
Acons = auxAcons;
for ss=2:3
	Acons = blkdiag(Acons,auxAcons);
end
Acons = cat(2,zeros(size(Acons,1),2*Nsp*Nch),Acons);
Acons = cat(1,Acons,-Acons);
bcons = (1e3*gamma_uT) * garea_max * ones(size(Acons,1),1); %gamma_uT*garea_max == [mT.s/m] * [rad/s/mT] == [rad/m]


%% Execute optimisation

if optim_struct.multistart
    
    all_xopt = cell(optim_struct.multistart_trials,1);
    all_fval = Inf(optim_struct.multistart_trials,1);
    parfor mm=1:optim_struct.multistart_trials
        [all_xopt{mm},all_fval(mm)] = fmincon(f,x0{mm},Acons,bcons,[],[],[],[],constr,options);      
    end
    idx_best = find(all_fval==min(all_fval),1,'first');
    x_opt = all_xopt{idx_best}; fval = all_fval(idx_best);
    
else
    
    [x_opt,fval] = fmincon(f,x0,Acons,bcons,[],[],[],[],constr,options);
    
end

b_opt = reshape(x_opt(1:Nsp*Nch)+1i*x_opt(Nsp*Nch+1:2*Nsp*Nch),[Nsp Nch]); 
k_opt = reshape(x_opt(2*Nsp*Nch+1:2*Nsp*Nch+3*Nsp), [Nsp 3])';


%% Cost function
    function [f, g] = cost(x,lambda,alpha_des,beta_des)
        %%% Extract RF and excitation k-space arrays
        b = x(1:Nsp*Nch)+1i*x(Nsp*Nch+1:2*Nsp*Nch); 
        k = reshape(x(2*Nsp*Nch+1:2*Nsp*Nch+3*Nsp), [Nsp 3]);

        %%% Flip angle calculation (STA approximation)
        if lambda<1
            A2 = pos * k';  A2 = exp(1i*A2);  A = A1 .* A2;
            Afull{1} = tx(:,1).*A; alpha = Afull{1}*b(1:Nsp);
            for nn=2:Nch; Afull{nn} = tx(:,nn).*A; alpha = alpha + Afull{nn}*b(1+Nsp*(nn-1):Nsp*nn); end
            alpha_diff = abs(alpha) - alpha_des; 
            alpha_err = W * abs(alpha_diff).^2;
        else
            alpha_err = 0;
        end
        
        %%% B1rms calculation (over pulse duration)
        if lambda>0
            b1   = 1e3 * tx * reshape(b,[Nsp Nch]).'; %[uT]
            b1sq = abs(b1).^2;                       
            b1rms = sqrt(sum(b1sq.*dt,2)/T); %assumes rectangular RF waveforms (p2=1)
            beta_diff = b1rms - beta_des;
            beta_err = W * abs(beta_diff).^2;
        else
            beta_err = 0;
        end
        
        f = (1-lambda)*alpha_err/alpha_norm + lambda*beta_err/beta_norm;
        
        if nargout>1
            if lambda<1
                Afull2 = [Afull{:}];

                dalpha_db = 1./abs(alpha) .* real(alpha.*conj(Afull2*ri2cp));
                ga_db = (1-lambda) * W * 2 * (alpha_diff.*dalpha_db) /alpha_norm;

                bfull = zeros(Nsp*Nch,Nsp); bfull(idx_bfull) = b; aux = Afull2*bfull;
                dalpha_dkx = 1./abs(alpha) .* imag(alpha .* conj(pos(:,1).*aux)); 
                dalpha_dky = 1./abs(alpha) .* imag(alpha .* conj(pos(:,2).*aux)); 
                dalpha_dkz = 1./abs(alpha) .* imag(alpha .* conj(pos(:,3).*aux));
                dalpha_dk = [dalpha_dkx, dalpha_dky, dalpha_dkz];

                ga_dk = (1-lambda) * W * 2 * (alpha_diff.*dalpha_dk) /alpha_norm;
                ga = [ga_db'; ga_dk'];
            else
                ga = 0;
            end
            
            if lambda>0
                dbetadb = 1e3 * dt /T * real(conj(repmat(b1,[1 2*Nch])) .* fullTx) ./b1rms;
                gb_db = lambda * W * 2 * (beta_diff.*dbetadb) /beta_norm;
                gb_dk = zeros(1, numel(k));
                gb = [gb_db'; gb_dk'];
            else
                gb = 0;
            end
            
            g = ga + gb;
        end
        
    end

%% Non-linear constraints
    function [c,ceq,gradc,gradceq] = nonlincon(x,VOP,power_factor_s,limits)
        b = [reshape(x(1:Nsp*Nch),[Nsp Nch])'; reshape(x(Nsp*Nch+1:2*Nsp*Nch),[Nsp Nch])'];
        b_coeff = transpose(b(1:Nch,:)+1i*b(Nch+1:end,:));
        b_coeff = b_coeff(:); %1st all TX for 1st subpulse, then all TX for 2nd subpulse and so on

        if nargout<=2
            cval = RF_constraints_dummy(b_coeff,VOP,Nch,power_factor_s,'100');
            c = cval - limits;
        else
            [cval,gradc] = RF_constraints_dummy(b_coeff,VOP,Nch,power_factor_s,'110');
            c = cval - limits;
            gradc = [gradc; zeros(3*Nsp,numel(cval))];
            gradceq = [];
        end
        ceq = [];
    end

%% Hessian
	function [H] = hessian_function(x,lag,lambda,alpha_des,beta_des,VOP,power_factor_s)
        b = x(1:Nsp*Nch)+1i*x(Nsp*Nch+1:2*Nsp*Nch); 
        k = reshape(x(2*Nsp*Nch+1:2*Nsp*Nch+3*Nsp), [Nsp 3]);

        %%% Flip angle calculation (STA approximation)
        if lambda<1
            A2 = pos * k';  A2 = exp(1i*A2); A = A1 .* A2;
            Afull{1} = bsxfun(@times,tx(:,1),A); alpha = Afull{1}*b(1:Nsp);
            for nn=2:Nch; Afull{nn} = bsxfun(@times,tx(:,nn),A); alpha = alpha + Afull{nn}*b(1+Nsp*(nn-1):Nsp*nn); end
            alpha_diff = abs(alpha) - alpha_des;
        end

        %%% B1rms calculation (over pulse duration)
        if lambda>0
            b1	 = 1e3 * tx * reshape(b,[Nsp Nch]).'; %[uT]
            b1sq = abs(b1).^2;                      
            b1rms = sqrt(sum(b1sq.* dt,2)/T);
            beta_diff = b1rms - beta_des;
        end

        if lambda<1
            Afull2 = [Afull{:}];
            Afull_ri = Afull2*ri2cp;

            dalpha_db = bsxfun(@times,1./abs(alpha),real(bsxfun(@times,alpha,conj(Afull_ri)))); 

            bfull = zeros(Nsp*Nch,Nsp); bfull(idx_bfull) = b; aux = Afull2*bfull;
            dalpha_dkx = bsxfun(@times,1./abs(alpha),imag(bsxfun(@times,alpha,conj(bsxfun(@times,pos(:,1),aux))))); 
            dalpha_dky = bsxfun(@times,1./abs(alpha),imag(bsxfun(@times,alpha,conj(bsxfun(@times,pos(:,2),aux))))); 
            dalpha_dkz = bsxfun(@times,1./abs(alpha),imag(bsxfun(@times,alpha,conj(bsxfun(@times,pos(:,3),aux))))); 
            dalpha_dk = [dalpha_dkx, dalpha_dky, dalpha_dkz];

            h_part1 = dalpha_db' * bsxfun(@times,W'/alpha_norm,dalpha_db);
            h_part2 = -real(bsxfun(@times,alpha,conj(Afull_ri)))' * bsxfun(@times,W'.*alpha_diff/alpha_norm,bsxfun(@times,dalpha_db,1./(abs(alpha).^2))) + ... 
                real(bsxfun(@times,1./abs(alpha),conj(Afull_ri)).' * bsxfun(@times,W'.*alpha_diff/alpha_norm,Afull_ri));
            Hga_dbdb = 2 * (1-lambda) * (h_part1 + h_part2);

            h_part1 = dalpha_dk' * bsxfun(@times,W'/alpha_norm,dalpha_db);
            h_part2_x = -imag(bsxfun(@times,alpha.*pos(:,1),conj(aux)))' * bsxfun(@times,W'.*alpha_diff/alpha_norm,bsxfun(@times,dalpha_db,1./(abs(alpha).^2))) + ... 
                imag(bsxfun(@times,pos(:,1),conj(aux)).' * bsxfun(@times,(bsxfun(@times,W',bsxfun(@times,alpha_diff/alpha_norm,1./abs(alpha)))), Afull_ri));
            h_part2_y = -imag(bsxfun(@times,alpha.*pos(:,2),conj(aux)))' * bsxfun(@times,W'.*alpha_diff/alpha_norm,bsxfun(@times,dalpha_db,1./(abs(alpha).^2))) + ... 
                imag(bsxfun(@times,pos(:,2),conj(aux)).' * bsxfun(@times,(bsxfun(@times,W',bsxfun(@times,alpha_diff/alpha_norm,1./abs(alpha)))), Afull_ri));
            h_part2_z = -imag(bsxfun(@times,alpha.*pos(:,3),conj(aux)))' * bsxfun(@times,W'.*alpha_diff/alpha_norm,bsxfun(@times,dalpha_db,1./(abs(alpha).^2))) + ... 
                imag(bsxfun(@times,pos(:,3),conj(aux)).' * bsxfun(@times,(bsxfun(@times,W',bsxfun(@times,alpha_diff/alpha_norm,1./abs(alpha)))), Afull_ri));
            h_part2 = cat(1,h_part2_x,h_part2_y,h_part2_z);
            h_part3 = [];
            aux_real = zeros(Nsp*Nch,Nsp); aux_imag = zeros(Nsp*Nch,Nsp);
            for pp = 1:3
                aux_pos = W * bsxfun(@times,alpha_diff.*alpha./abs(alpha)/alpha_norm.*pos(:,pp),conj(Afull_ri));
                aux_real(idx_bfull(:)) = aux_pos(1:Nch*Nsp);
                aux_imag(idx_bfull(:)) = aux_pos(Nch*Nsp+1:2*Nch*Nsp);
                h_part3 = cat(1,h_part3,imag(cat(1,aux_real,aux_imag).'));
            end
            Hga_dbdk = 2 * (1-lambda) * (h_part1 + h_part2 + h_part3);

            h_part1 = dalpha_dk' * bsxfun(@times,W'/alpha_norm,dalpha_dk);
            h_part2 = -imag([bsxfun(@times,alpha.*pos(:,1),conj(aux)), bsxfun(@times,alpha.*pos(:,2),conj(aux)), bsxfun(@times,alpha.*pos(:,3),conj(aux))]).' * bsxfun(@times,W'.*alpha_diff/alpha_norm,bsxfun(@times,dalpha_dk,1./(abs(alpha).^2))) + ...
                imag(([bsxfun(@times,pos(:,1),conj(aux)), bsxfun(@times,pos(:,2),conj(aux)), bsxfun(@times,pos(:,3),conj(aux))]).' * bsxfun(@times,W'.*alpha_diff./abs(alpha)/alpha_norm,1i*[bsxfun(@times,pos(:,1),aux), bsxfun(@times,pos(:,2),aux), bsxfun(@times,pos(:,3),aux)]));
            h_part3 = zeros(3*Nsp);
            for kk = 1:3*Nsp
                if kk<=Nsp
                    pp = 1;
                elseif kk<=2*Nsp
                    pp = 2;
                else
                    pp = 3;
                end
                Afull3 = zeros(size(Afull2));
                Afull3(:,(mod(kk-1,Nsp)+1):Nsp:Nsp*Nch) = Afull2(:,(mod(kk-1,Nsp)+1):Nsp:Nsp*Nch);
                h_part3(kk,:) = (W.*(alpha_diff./abs(alpha)/alpha_norm).') * imag(bsxfun(@times,alpha,[repmat(pos(:,1),[1 Nsp]), repmat(pos(:,2),[1 Nsp]), repmat(pos(:,3),[1 Nsp])]).*repmat(conj(1i*bsxfun(@times,pos(:,pp),Afull3)*bfull),[1 3]));
            end
            Hga_dkdk = 2 * (1-lambda) * (h_part1 + h_part2 + h_part3);
        else
            Hga_dbdb = zeros(2*Nch*Nsp,2*Nch*Nsp);
            Hga_dbdk = zeros(3*Nsp,2*Nch*Nsp);
            Hga_dkdk = zeros(3*Nsp,3*Nsp);
        end

        if lambda>0
            dbetadb = 1e3*dt/T*real(conj(repmat(b1,[1 2*Nch])).*fullTx)./b1rms;

            h_part1 = dbetadb' * bsxfun(@times,W'/beta_norm,dbetadb);
            N2 = W'.*beta_diff./beta_norm.*dt./T./(b1rms.^2);
            h_part2 = -real(T./dt*bsxfun(@times,b1rms,dbetadb)).' * bsxfun(@times,N2,dbetadb);
            aux = repmat(eye(Nsp),[2*Nch 1]);
            h_part3 = 1e6 * (aux*aux') .* real(fullTx' * bsxfun(@times,N2.*b1rms,fullTx));
            Hgb_dbdb = 2 * lambda * (h_part1 + h_part2 + h_part3);
        else
            Hgb_dbdb = zeros(2*Nch*Nsp);
        end
        Hgb_dbdk = zeros(3*Nsp,2*Nch*Nsp);
        Hgb_dkdk = zeros(3*Nsp,3*Nsp);

        H = [Hga_dbdb + Hgb_dbdb,    (Hga_dbdk + Hgb_dbdk)';
             Hga_dbdk + Hgb_dbdk,     Hga_dkdk + Hgb_dkdk];
         
        %%% Complete Hessian with Lagrange terms
        [~,~,Hconstr] = RF_constraints_dummy(b,VOP,Nch,power_factor_s,'001');
        for k = 1:numel(Hconstr)
            H = H + lag.ineqnonlin(k) * blkdiag(real(Hconstr{k}),zeros(3*Nsp));
        end

	end



end
