% Start brainstorm to use database engine
if ~brainstorm('status')
    brainstorm nogui
end

ProtocolName = 'test01';
% Get the protocol index
iProtocol = bst_get('Protocol', ProtocolName);
if isempty(iProtocol)
    error(['Unknown protocol: ' ProtocolName]);
end
% Select the current procotol
gui_brainstorm('SetCurrentProtocol', iProtocol);

% Select all Studies
tt = bst_get('ProtocolStudies');

% Filter out to only fif's entries
iStudy = [];
for i = 1:length(tt.Study)
    if (tt.Study(i).iHeadModel)
        iStudy(end+1) = i;
    end
end

% cerate a struct for HeadModel recalc
sMethod = [];
sMethod.Comment       = 'Overlapping spheres';
sMethod.HeadModelType = 'surface';
sMethod.MEGMethod     = 'os_meg';
sMethod.EEGMethod     = '';
sMethod.ECOGMethod    = '';
sMethod.SEEGMethod    = '';
sMethod.SaveFile      = 1;

% Recalc Head Model for all subjects 
panel_headmodel('ComputeHeadModel', iStudy, sMethod);

% Close brainstorm
brainstorm stop