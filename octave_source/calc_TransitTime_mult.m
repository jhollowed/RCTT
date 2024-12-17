function TTime=calc_TransitTime_mult(latgr,plevgr,lats,plevs,v,w,tropP,ifsave,Path,ifmatrix)

% function calc_TransitTime_mult(latgr,plevgr,lats,plevs,v,w,tropoP,ifsave,Path,ifmatrix)
%
% as calc_TransitTime but calculates trajectories at same time, which is much more time efficient.
% (and fixed some bugs!)
%
% calculates transit time along residual circulation trajectories, starting 
% from each point in the vectors of latgr and plevgr. The monthly mean values 
% of v [m/s] and w [m/s] are given on the grid of lats [deg] and plevs [hPa].
% tropP is the monthly mean tropopause pressure for each latitude in Pa. 
% The fileds v,w and tropP should be given for 10 years, and the transit 
% time is claculated for the first  timestep in these arrays.
% v,w: NlatxNlevxNtime
% tropP: NlatxNtime
% the optional ifsave argument is to be set to true if the trajectories should
% be saved, in which case a path must be given. Otherwise the trajectories
% are not saved.
%if ifmatrix is set to false, latgr and plevgr are used as single vectors. If true (default),
%TTime is calculated on a matrix of latgrxplevgr.

if (nargin<8)
  ifsave = false;
elseif (nargin==8 & ifsave==true)
  disp('if you want to save the trajectories you have to speciy a path');
  stop;
end

if(nargin<10)
 ifmatrix = true;
end

%resolution of calculation in days:
resday = 5;

%some set ups
nlat = length(lats);
nlev = length(plevs);

disp('interpolation of tropopoause pressure');
%interpolate to resolution of trajectory in time and convert to Z
tropZ = -7000*log(tropP/101300);
for lat=1:nlat
  tropZint(lat,:) = interp1([1:30:1+119*30],tropZ(lat,:),[1:resday:3600]);  
end

% transform latitude in m and pressure into height in m!
latsR = lats*111000; %m
plevs = -7000*log(plevs/1013);

if (ifmatrix)
  TTime = zeros(length(latgr),length(plevgr));

  Ntraj = length(latgr)*length(plevgr);
  latgr = latgr(:)';
  lat0 = repmat(latgr,1,length(plevgr));
  p0 = reshape(repmat(plevgr,length(latgr),1),1,Ntraj);
else
  if (length(latgr) ~= length(plevgr))
    disp('if you choose ifmatrix=flase, latgr and plevgr must be of same length')
    stop;
  end
  TTime = zeros(length(latgr),1);
  Ntraj = length(latgr);
  lat0 = latgr(:);
  p0 = plevgr(:);
  plevgr = ones(1,1);
end

days=360;
x0 = lat0*111000; %in m
y0 = -7000*log(p0/1013);

disp(['calculate ' int2str(Ntraj) ' trajectories first year']);
Xt = TrajModel_mult(x0,y0,-1*v(:,:,1:13),-1*w(:,:,1:13),60*60*24*days,latsR,plevs,24*30,resday*24*60*60);

len = size(Xt,1);
% transform back in lats:
Xt(:,1,:)=Xt(:,1,:)/111000;


tropR = zeros(length(latgr),length(plevgr));
%search for crossing of tropopause for each time step.
disp('search for tropopause crossings')
for ilat=1:length(latgr)
  for ip=1:length(plevgr)
   i = (ip-1)*length(latgr)+ilat;   
   for t=1:size(Xt,1)	
     if ( Xt(t,2,i) <= tropZint(GetIndforVal(Xt(t,1,i),lats),t))          
	  if (t==1)
	    TTime(ilat,ip) = 0;
	  else		
	    %interpolate to day of tropopause crossing	 
           trop0(1) = tropZint(GetIndforVal(Xt((t-1),1,i),lats),(t-1));      
           trop0(2) = tropZint(GetIndforVal(Xt(t,1,i),lats),t);           
	   TTime(ilat,ip) = interp1(Xt([(t-1):t],2,i)-trop0',[(t-1)*resday (t)*resday],0);                                                   
  	  end
          disp(['Transit time at ' int2str(lat0(i)) ' deg and ' int2str(p0(i)) 'hPa: ' ...
	          num2str(TTime(ilat,ip)) ' days']);
	  tropR(ilat,ip) = 1;
	  Xt(t:end,1,i) = Xt(t,1,i);
	  Xt(t:end,2,i) = Xt(t,2,i);
	  break
      end
    end % end loop through timesteps
  end
end


year = 1;
while (any(any(tropR==0)) & year < 9)	
  len1 = size(Xt,1);
  disp(['calculate trajectories year ' int2str(year+1) ]);
  Xt2 = TrajModel_mult(Xt(len1,1,:)*111000,Xt(len1,2,:),...
      -1*v(:,:,year*12+[1:13]),-1*w(:,:,year*12+[1:13]),60*60*24*days,...
      latsR,plevs,24*30,resday*24*60*60);
  % transform back in lats:
  Xt2(:,1,:)=Xt2(:,1,:)/111000;
  Xt = [Xt; Xt2];	 	

  %search for crossing of tropopause
  for ilat=1:length(latgr)
    for ip=1:length(plevgr)
      if (tropR(ilat,ip) == 0)
	i = (ip-1)*length(latgr)+ilat; 
	for t=len1:size(Xt,1)	    
	  if (Xt(t,2,i) <= tropZint(GetIndforVal(Xt(t,1,i),lats),t))
            trop0(1) = tropZint(GetIndforVal(Xt((t-1),1,i),lats),(t-1));      
            trop0(2) = tropZint(GetIndforVal(Xt(t,1,i),lats),t);           
	    TTime(ilat,ip) = interp1(Xt([(t-1):t],2,i)-trop0',[(t-1)*resday (t)*resday],0);	    
	    disp(['Transit time at ' int2str(lat0(i)) ' deg and ' int2str(p0(i)) 'hPa: ' ...
		    num2str(TTime(ilat,ip)) ' days']);
	    tropR(ilat,ip) = 1;
	    Xt(t:end,1,i) = Xt(t,1,i);
	    Xt(t:end,2,i) = Xt(t,2,i);	
	    break
	  end
	end 
      end % only if not reached yet.
    end
  end
  year = year+1;  
end

      
if (any(any(tropR==0)))        	  
    TTime(tropR==0) = 10*360; %10 years
    disp('tropopause not reached, set time to 10 years for remaining trajectories');
end

if (ifsave)
  for ilat=1:length(latgr)
    for ip=1:length(plevgr)
      i = (ip-1)*length(latgr)+ilat; 
      if (tropR(ilat,ip)==0)
	dlmwrite([Path 'ResidTraj_' int2str(p0(i)) 'hPa_' ...
	  FInt2Str(2,x0(i)/111000) 'deg.dat'],Xt(1:end,:,i),' ');
      else
	dlmwrite([Path 'ResidTraj_' int2str(p0(i)) 'hPa_' ...
	  FInt2Str(2,x0(i)/111000) 'deg.dat'],Xt(1:ceil(TTime(ilat,ip)/5.0),:,i),' ');
      end
    end
  end
end

  

end