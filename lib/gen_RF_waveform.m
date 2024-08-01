function [bfull] = gen_RF_waveform(bcoeff,rf_form,rf_detail)
%[bfull] = gen_RF_waveform(bcoeff,rf_form,rf_detail)
%   Generates full RF waveforms based on the RF coefficients and RF shape &
%   number of samples passed on rf_detail and rf_form, respectively.
%   INPUTS:
%       bcoeff        -> RF samples coefficients (dimensions: Nsubpulses x
%                     Nch) [Volts]
%       rf_form       -> array with normalized RF waveform for each channel
%       rf_detail     -> matrix detailing number of samples in 1st column
%                     and sample type in 2nd column (No RF = 0, RF = 1)
%   OUTPUTS:
%       bfull         -> matrix with full RF waveform (dimensions: Nsamples 
%                     x Nch) [Volts]
%
% David Leitao 2024, King's College London (david.leitao@kcl.ac.uk)

Nch = size(bcoeff,2);
Nsamples = sum(rf_detail(:,1));
bfull = zeros(Nsamples,Nch);

n = 1; blip = 0;
while n<size(rf_detail,1)
    if rf_detail(n,2)==1
        blip = blip + 1;
        idx = sum(rf_detail(1:n-1,1))+1:sum(rf_detail(1:n,1));
        bfull(idx,:) = rf_form(idx)*bcoeff(blip,:);
    end
    n = n + 1;
end

end