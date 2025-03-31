function [C,mu,A,B,Theta]=iRRR_normal3(Y,X,lam1,paramstruct)
% This function uses consensus ADMM to fit the iRRR model. It is suitable
% for continuous outcomes (no missing or missing).
%
% Model:
% 1/(2n)*|Y-1*mu'-sum(X_i*B_i)|^2  + lam1*sum(w_i*|A_i|_*) (+0.5*lam0*sum(w_i^2*|B_i|^2_F))
% s.t.  A_i=B_i
% 
%
% input: 
%
%   Y       n*q continuous response data matrix 
%
%   X       1*K cell array, each cell is a n*p_i predictor data matrix
%           Note: X1,...XK may need some sort of standardization, because
%           we use a single lam0 and lam1 for different predictor sets.
%           Namely, we implicitly assume the coefficients are comparable.
% 
%   lam1    positive scalar, tuning for nuclear norm 
%
%   paramstruct
%          lam0     tuning for the ridge penalty, default=0
%r
%          weight   K*1 weight vector, default: a vector of 1; 
%                   By theory, we should use w(i)=(1/n)*max(svd(X{i}))*(sqrt(q)+sqrt(rank(X{i}))); where X is column centered 
%                   Hueristically, one could also use w(i)=|X_i|_F
%
%          randomstart         0=false (default); 1=true
%
%          varyrho  0=fixed rho (default); 1=adaptive rho
%          maxrho   5 (default): max rho. Unused if varyrho==0
%
%          rho      initial step size, default rho=0.1
%
%          Tol      default 1E-3, 
%
%          Niter	default 500
%
%          fig      1 (default) show checking figures; 0 no show
%
% Output: 
%
%   C       sum(p_i)*q coefficient matrix, potentially low-rank
%
%   mu      q*1 intercept vector (mean(Y,1)-mean(X,1)*hat{C})'
%           if X and Y are non-missing and column centered, mu=zeros(q,1)
%
%   A       cell arrays of length K, separate low-rank coefficient matrices
%
%   B       cell arrays of length K, separate coefficient matrices
%
%   Theta   cell arrays of length K, Lagrange parameter matrices
%
%
% 
% Modified from iRRR_normal2.m on 12/10/2017 by Gen Li
%   Note: allow Y to have missing values
%         when Y has missing, cannot directly center Y

% default parameters
K=length(X);
weight=ones(K,1);
Tol=1E-3; % stopping rule
Niter=500; % max iterations
varyrho=0;
rho=0.1;
lam0=0;
maxrho=5;
randomstart=0;
fig=1;
if nargin > 3 ;   %  then paramstruct is an argument
  if isfield(paramstruct,'lam0') ;   
    lam0 = paramstruct.lam0 ; 
  end ;
  if isfield(paramstruct,'weight') ;   
    weight = paramstruct.weight ; 
  end ;
  if isfield(paramstruct,'Tol') ;   
    Tol = paramstruct.Tol ; 
  end ;
  if isfield(paramstruct,'Niter') ;   
    Niter = paramstruct.Niter ; 
  end ;
  if isfield(paramstruct,'randomstart') ;   
    randomstart = paramstruct.randomstart ; 
  end ;
  if isfield(paramstruct,'varyrho') ;   
    varyrho = paramstruct.varyrho; 
  end ;
  if varyrho && isfield(paramstruct,'maxrho') ;
      maxrho = paramstruct.maxrho; 
  end;
  if isfield(paramstruct,'rho') ;   
    rho = paramstruct.rho; 
  end ;
  if isfield(paramstruct,'fig') ;   
    fig = paramstruct.fig ; 
  end ;
end;



% initialization
[n,q]=size(Y);
% % center Y
% meanY=mean(Y,1);
% Y=bsxfun(@minus,Y,meanY);
p=zeros(K,1);
cX=[]; % horizontally concatenated X
meanX=[];
for i=1:K
    [n_,p(i)]=size(X{i});
    if n_~=n
        error('Samples do not match!')
    end;
    % first, column center X{i}'s
    meanX=[meanX,mean(X{i},1)];
    X{i}=bsxfun(@minus,X{i},mean(X{i},1)); % this is important
    % second, normalize centered X{i}'s
    X{i}=X{i}/weight(i);
    cX=[cX,X{i}]; % column centered X
end;

% initial parameter estimates
mu=mean(Y,1,'omitnan')'; % q*1
% majorize Y to get a working Y
wY=Y;
temp=ones(n,1)*mu';
wY(isnan(wY))=temp(isnan(wY)); % wY should be a complete matrix 
mu=mean(wY,1)'; % new est of mu, b/c cX is col centered
wY1=bsxfun(@minus,wY,mu'); % column centered wY
%
B=cell(K,1); 
Theta=cell(K,1); % Lagrange params for B
cB=zeros(sum(p),q);% vertically concatenated B
for i=1:K
    if randomstart
        B{i}=randn(p(i),q);
    else
        B{i}=pinv(X{i}'*X{i})*X{i}'*wY1; %  OLS with generalized inverse
    end;
    Theta{i}=zeros(p(i),q);
    cB((sum(p(1:(i-1)))+1):sum(p(1:i)),:)=B{i};
end;
A=B; % low-rank alias
cA=cB; 
cTheta=zeros(sum(p),q);
%
[~,D_cX,V_cX]=svd((1/sqrt(n))*cX,'econ');
if ~varyrho % fixed rho
    DeltaMat=V_cX*diag(1./(diag(D_cX).^2+lam0+rho))*V_cX'+...
        (eye(sum(p))-V_cX*V_cX')/(lam0+rho);   % inv(1/n*X'X+(lam0+rho)I)
end;


% check obj value
obj=ObjValue1(Y,X,mu,A,lam0,lam1); % full objective function (with penalties) on observed data
obj_ls=ObjValue1(Y,X,mu,A,0,0); % only the least square part on observed data



%%%%%%%%%%%%%%%
% ADMM
niter=0;
diff=inf;
rec_obj=[obj;obj_ls]; % record obj value
rec_Theta=[]; % record the Fro norm of Theta{1}
rec_primal=[]; % record total primal residual
rec_dual=[]; % record total dual residual
while niter<Niter  && abs(diff)>Tol
    niter=niter+1;
    cB_old=cB;
    
    
    %%%%%%%%%%%%% Majorization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Eta = ones(n,1)*mu' + cX*cB; % current linear predictor
    wY = Y;
    wY(isnan(wY))=Eta(isnan(wY)); % working response 
    mu=mean(wY,1)'; % new est of mu
    wY1=bsxfun(@minus,wY,mu'); % column centered wY

    
    % est concatenated B
    if varyrho
        DeltaMat=V_cX*diag(1./(diag(D_cX).^2+lam0+rho))*V_cX'+...
            (eye(sum(p))-V_cX*V_cX')/(lam0+rho); 
    end;
    cB=DeltaMat*((1/n)*cX'*wY1+rho*cA+cTheta);
    for i=1:K
        B{i}=cB((sum(p(1:(i-1)))+1):sum(p(1:i)),:);
    end;    
    
    
           
    % est A_i in parallel
    % update Theta_i in parallel right after est A_i
    parfor i=1:K
        % est A
        temp=B{i}-Theta{i}/rho;
        [tempU,tempD,tempV]=svd(temp,'econ');
        A{i}=tempU*SoftThres(tempD,lam1/rho)*tempV';
        
        % update Theta
        Theta{i}=Theta{i}+rho*(A{i}-B{i});
    end;
    % reshape cA and cTheta
    for i=1:K
        cA((sum(p(1:(i-1)))+1):sum(p(1:i)),:)=A{i};
        cTheta((sum(p(1:(i-1)))+1):sum(p(1:i)),:)=Theta{i};
    end;
    
        
    % update rho
    if varyrho
        rho=min(maxrho,1.1*rho); % steadily increasing rho
    end;
    

    
    % stopping rule
    % primal and dual residuals
    primal=norm(cA-cB,'fro')^2;
    rec_primal=[rec_primal,primal];
    dual=norm(cB-cB_old,'fro')^2;
    rec_dual=[rec_dual,dual];  

    % objective function value
    obj=ObjValue1(Y,X,mu,A,lam0,lam1);
    obj_ls=ObjValue1(Y,X,mu,A,0,0);
    rec_obj=[rec_obj,[obj;obj_ls]];
    
    % stopping rule
    diff=primal; 
%     diff=dual;
%     diff=rec_obj(1,end-1)-rec_obj(1,end);



    % Check Figures
    if fig==1
        % obj fcn values
        figure(101);clf; 
        plot(0:niter,rec_obj(1,:),'bo-');
        hold on
        plot(0:niter,rec_obj(2,:),'ro-');
        legend('Full Obj Value','LS Obj Value')
        title(['Objective function value (decrease in full=',num2str(rec_obj(1,end-1)-rec_obj(1,end)),')']);
        drawnow;
 
        % primal and dual residuals
        figure(102);clf;
        subplot(1,2,1)
        plot(1:niter,rec_primal,'o-');
        title(['|A-B|^2: ',num2str(primal1)]);
        subplot(1,2,2)
        plot(1:niter,rec_dual,'o-');
        title(['Dual residual |B-B|^2: ',num2str(dual)]);
        drawnow
    
        figure(103);clf;
        rec_Theta=[rec_Theta,norm(Theta{1},'fro')];
        plot(rec_Theta,'o-');
        title(['Theta: Lagrange multiplier for B1']);
        drawnow
    end;



end;


if niter==Niter
    disp(['iRRR does NOT converge after ',num2str(Niter),' iterations!']);
else
    disp(['iRRR converges after ',num2str(niter),' iterations.']);      
end;


% output
% rescale parameter estimate, and add back mean
C=[];
for i=1:K
    A{i}=A{i}/weight(i);
    B{i}=B{i}/weight(i);
    C=[C;A{i}];
end;
clear cA cB;
mu=(mu'-meanX*C)';
 
end






function Dout=SoftThres(Din,lam)
% this function soft thresholds the diagonal values of Din
% Din is a diagonal matrix
% lam is a positive threshold
% Dout is also a diagonal matrix
d=diag(Din);
d(d>0)=max(d(d>0)-lam,0);
d(d<0)=min(d(d<0)+lam,0);
Dout=diag(d);
end








% function obj=ObjValue(Y,X,B,lam0,lam1)
% % Calc 1/(2n)|Y-sum(Xi*Bi)|^2 + lam0/2*sum(|Bi|_F^2) + lam1*sum(|Bi|_*) 
% % with column centered and complete Y and Xi's 
% [n,q]=size(Y);
% K=length(X);
% obj=0;
% pred=0;
% for i=1:K
%     pred=pred+X{i}*B{i};
%     obj=obj+lam0/2*norm(B{i},'fro')^2+lam1*sum(svd(B{i}));
% end;
% obj=obj+(1/(2*n))*norm(Y-pred,'fro')^2;
% end

function obj=ObjValue1(Y,X,mu,B,lam0,lam1)
% Calc 1/(2n)|Y-1*mu'-sum(Xi*Bi)|^2 + lam0/2*sum(|Bi|_F^2) + lam1*sum(|Bi|_*) 
% with column centered  Xi's and (potentially non-centered and missing) Y
[n,q]=size(Y);
K=length(X);
obj=0;
pred=ones(n,1)*mu';
for i=1:K
    pred=pred+X{i}*B{i};
    obj=obj+lam0/2*norm(B{i},'fro')^2+lam1*sum(svd(B{i}));
end;
obj=obj+(1/(2*n))*sum(sum((Y-pred).^2,'omitnan'),'omitnan');
end
