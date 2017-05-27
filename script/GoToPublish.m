% evaluate publish
published_doc_name = publish('D:\science\Brain\MSMM\script\DisplayResultTimeCources.m', 'format', 'pdf', 'outputDir', 'D:\science\Brain\MSMM\script\publish\', 'showCode', false);
%published_doc_name = 'D:\science\Brain\MSMM\script\publish\DisplayResultTimeCources.pdf';
% get current time to append to filename
t = datestr((datetime('now')));
% get rid of spaces and ':'
t(t==' ') = '_';
t(t==':') = '_';

% get filename decomposition
[pathstr,name,ext] = fileparts(published_doc_name);
% append to filename
newFile = [pathstr filesep() name '_' t ext];
[success,msg,msgid] = movefile(published_doc_name,newFile);
if ~success
    error(msgid,msg);
end

