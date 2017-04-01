function DisplayActivityInteractive(mesh, J)
	    time = 1;

	    mn = min(min(J));
	    mx = max(max(J));

	    f = figure;

	    b = uicontrol('Parent',f,'Style','slider','Position',[81,54,419,23], 'min',1, 'max',size(J,2), 'value', 1, 'SliderStep', [1/size(J,2) 0.1], 'Callback', @slider_callback);

	    bgcolor = f.Color;
		bl1 = uicontrol('Parent',f,'Style','text','Position',[50,54,23,23],...
		                'String','1','BackgroundColor',bgcolor);
		bl2 = uicontrol('Parent',f,'Style','text','Position',[500,54,23,23],...
		                'String',num2str(size(J,2)),'BackgroundColor',bgcolor);
		bl3 = uicontrol('Parent',f,'Style','text','Position',[240,25,100,23],...
		                'String','Time (1)','BackgroundColor',bgcolor);
        

        cla; axis off;

        ptch = patch('vertices',mesh.Vertices,'faces',mesh.Faces,'FaceVertexCData',J(:,time));

        set(ptch,'FaceColor',[.5 .5 .5],'EdgeColor','none');
        shading interp;
        lighting gouraud;
        %camlight;
        zoom off;
        lightangle(0,270);lightangle(270,0),lightangle(90,0),lightangle(0,45),lightangle(0,135);
        material([.1 .1 .4 .5 .4]);
        %material([.1 .1 .4 .5 .4]);
        view(140,15);
        caxis([mn mx]);
        %caxis([0 1]);
        colormap('jet');
        
        set( findall( f, '-property', 'Units' ), 'Units', 'Normalized' );


function slider_callback(hObj,eventdata)
	time = round(get(hObj,'Value'));

	set(ptch,'FaceVertexCData', J(:,time));
    set(bl3, 'String', ['Time (', num2str(time),')']);

    %mn = min(min(J(:,time)));
	%mx = max(max(J(:,time)));
    %caxis([mn mx]);
end

end