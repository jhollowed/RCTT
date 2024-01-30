function [Xt]=TrajModel_mult(x,y,u,v,t,gridx,gridy,Dt,h)

% function [Xt]=TrajModel_mult(x,y,u,v,t,gridx,gridy,Dt,[h])
% 2-D Trajectory Model, using 4th order Runge-Kutta Scheme. Input is inital
% positions x and y (arrays) and windfields u and v, and the total time (in seconds) of
% integration t. The input 'gridx'/'gridy' specifies the grid of u and v, and
% Dt the temporal resolution of the wind fields in hours.  The timestep of
% integration (h) is set to 1 hour if not otherwise sepcified. Winds are in m/s, and u 
% and v are 3-d arrays: u(x,y,t).


if (nargin==8)
    h=60*60; % 1hour
end


S = size(u);
gridt = [0:Dt*60*60:(S(3)-1)*Dt*60*60]'; % in seconds
[meshx,meshy,mesht]=ndgrid(gridx,gridy,gridt);

N = length(x); %number of trajectories;

try    
    Xt = zeros(round(t/h)+1,2,N);
    Xt(1,1,:)=x;
    Xt(1,2,:)=y;   
    for ts=0:h:t-h %timesteps in seconds      
        tsV = ones(size(x))*ts; 
        % outside of domain?: reset
	x = resetz(x,min(gridx),max(gridx));
	y = resetz(y,min(gridy),max(gridy));
	% interpolation
	k1 = h*interpn(meshx,meshy,mesht,u,x,y,tsV,'linear',0);
	xk1 = resetz(x+k1/2,min(gridx),max(gridx));
	k2 = h*interpn(meshx,meshy,mesht,u,xk1,y,tsV+h/2,'linear',0);
	xk2 = resetz(x+k2/2,min(gridx),max(gridx));
	k3 = h*interpn(meshx,meshy,mesht,u,xk2,y,tsV+h/2,'linear',0);
	xk3 = resetz(x+k3,min(gridx),max(gridx));
	k4 = h*interpn(meshx,meshy,mesht,u,xk3,y,tsV+h,'linear',0);
	xnew = x +k1/6 +k2/3 +k3/3+ k4/6;      
	k1 = h*interpn(meshx,meshy,mesht,v,x,y,tsV,'linear',0);
	yk1 = resetz(y+k1/2,min(gridy),max(gridy));
	k2 = h*interpn(meshx,meshy,mesht,v,x,yk1,tsV+h/2,'linear',0);
	yk2 = resetz(y+k2/2,min(gridy),max(gridy));	
	k3 = h*interpn(meshx,meshy,mesht,v,x,yk2,tsV+h/2,'linear',0);
	yk3 = resetz(y+k3,min(gridy),max(gridy));	
	k4 = h*interpn(meshx,meshy,mesht,v,x,yk3,tsV+h,'linear',0);
	y = y +k1/6 +k2/3 +k3/3+ k4/6;
	x = xnew;
	Xt(round((ts+2*h)/h),1,:)=x;
	Xt(round((ts+2*h)/h),2,:)=y;
    end    
catch
    disp(lasterr);
end

end

function z1 = resetz(z,zmin,zmax)
  z1 = z;
  if (any(z>zmax))	  	      
    z1(z>zmax) = zmax; 
  end
  if (any(z<zmin));	      
    z1(z<zmin) = zmin;
  end
end