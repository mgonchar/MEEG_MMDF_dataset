function DisplayActivity(mesh, J, time)

	    mn = min(min(J(:, time)));
	    mx = max(max(J(:, time)));

	    figure;        

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
end