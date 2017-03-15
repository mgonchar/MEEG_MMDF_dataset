function Wmat = my_interpolation(srcSurfFile, destSurfFile, nbNeighbors)

    if nargin < 3
        % Evaluate number of vertices to use
        nbNeighbors = 8;
    end

    % Load source surface file
    srcSurfMat  = load(srcSurfFile);
    % Load destination surface file
    destSurfMat = load(destSurfFile);
    
    % Get source and destination subjects
    %sSrcSubj  = bst_get('SurfaceFile', srcSurfFile);
    %sDestSubj = bst_get('SurfaceFile', destSurfFile);
    
    % Number of vertices
    nSrc  = size(srcSurfMat.Vertices,  1);
    nDest = size(destSurfMat.Vertices, 1);

    Wmat = [];   

    % ===== USE FREESURFER SPHERES =====
    % If the registered spheres are available in both surfaces
    
    SrcDataAvailable = isfield(srcSurfMat,  'Reg') && isfield(srcSurfMat.Reg,  'Sphere') && isfield(srcSurfMat.Reg.Sphere,  'Vertices') && ~isempty(srcSurfMat.Reg.Sphere.Vertices);
    DstDataAvailable = isfield(destSurfMat, 'Reg') && isfield(destSurfMat.Reg, 'Sphere') && isfield(destSurfMat.Reg.Sphere, 'Vertices') && ~isempty(destSurfMat.Reg.Sphere.Vertices);
    
    if SrcDataAvailable && DstDataAvailable
       
        % Allocate interpolation matrix
        Wmat = spalloc(nDest, nSrc, nbNeighbors * nDest);
        % Split hemispheres
        [rHsrc, lHsrc,  isConnected(1)] = tess_hemisplit(srcSurfMat);
        [rHdest,lHdest, isConnected(2)] = tess_hemisplit(destSurfMat);
        % Get vertices
        srcVert  = double(srcSurfMat.Reg.Sphere.Vertices);
        destVert = double(destSurfMat.Reg.Sphere.Vertices);
        % If hemispheres are connected: process all at once
        if any(isConnected)
            rHsrc  = 1:nSrc;
            rHdest = 1:nDest;
            lHsrc  = [];
            lHdest = [];
        end
        % Re-interpolate using the sphere and the shepards algorithm
        Wmat(rHdest,rHsrc) = bst_shepards(destVert(rHdest,:), srcVert(rHsrc,:), nbNeighbors, 0);
        if ~isempty(lHdest)
            Wmat(lHdest,lHsrc) = bst_shepards(destVert(lHdest,:), srcVert(lHsrc,:), nbNeighbors, 0);
        end
        
    else
        if ~SrcDataAvailable
            fprintf('FreeSurfer Spheres are not available in Source file: %s\n', srcSurfFile);
        end
        
        if ~DstDataAvailable
            fprintf('FreeSurfer Spheres are not available in Source file: %s\n', destSurfFile);
        end     
    end
end
