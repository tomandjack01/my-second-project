function out = Pate(a,XT)
    out=zeros(1,3);
    d1=a(:,1);
    d2=a(:,2);
    XT=8000; %时间序列长度
%     XT=10000;
  
    
    grotMIN=prctile(d1,10); %百分之10
    grotMAX=prctile(d1,90); 
    d1b=max(min((d1-grotMIN)/(grotMAX-grotMIN),1),0);
    grotMIN=prctile(d2,10); 
    grotMAX=prctile(d2,90);  
    d2b=max(min((d2-grotMIN)/(grotMAX-grotMIN),1),0);
    d1b=double(d1b>0.75); d2b=double(d2b>0.75);
    theta1=d1b'*d2b/XT;
    theta2=d1b'*(1-d2b)/XT;  
    theta3=d2b'*(1-d1b)/XT;  
    theta4=(1-d1b)'*(1-d2b)/XT;
    EEE=(theta1+theta2)*(theta1+theta3);
    max_theta1=min(theta1+theta2,theta1+theta3); 
    min_theta1=max(0,2*theta1+theta2+theta3-1);
    if (theta1>EEE), DDD=0.5+(theta1-EEE)/(2*(max_theta1-EEE));
        else DDD=0.5-(theta1-EEE)/(2*(EEE-min_theta1));
    end;
    kappa=(theta1-EEE)/(DDD*(max_theta1-EEE) + (1-DDD)*(EEE-min_theta1));
   % urgh(kappa,'Patel Kappa');
    out(1)=kappa;
    if (theta2>theta3), tau_12=1-(theta1+theta3)/(theta1+theta2); 
        else tau_12=(theta1+theta2)/(theta1+theta3)-1; 
    end;
    %urgh(-tau_12,'Patel Tau');
    out(2)=-tau_12;
    %urgh(-kappa * tau_12,'Tau x Kappa');
    out(3)=-kappa * tau_12;
    output=out(3);
    



