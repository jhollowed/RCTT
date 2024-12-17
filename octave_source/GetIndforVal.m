function ind = GetIndforVal(val,Level);

% function ind = GetIndforVal(val,Level);
% get (nearest) Index ind for level 'val' within the specified list of
% 'Level'

len = length(Level);

% 1. decreasing levels
if (Level(1) > Level(len))
    if (val<=Level(len));
        ind = len;
    elseif (val >= Level(1));
        ind = 1;
    else
        ind=1;
        i=1;
        while ( abs(val-Level(i+1)) < abs(val-Level(i)))
            ind=i+1;
            i=i+1;
            if (ind==len)
                break
            end
        end
    end
else
    if (val<=Level(1));
        ind = 1;
    elseif (val >= Level(len));
        ind = len;
    else
        ind=1;
        i=1;
        while ( abs(val-Level(i+1)) < abs(val-Level(i)))
            ind=i+1;
            i=i+1;
            if (ind==len)
                break
            end
        end
    end
end