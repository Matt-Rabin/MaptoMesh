# MaptoMesh
Takes a flat geographic world map and converts it into a textured globe mesh.

V1.0
Fun experiment to see how far I can push some pin-hole camera modeling on CPU. Obviously not optimal solution since we are moving a pseudo-spherical mesh rather than just warping and moving a UV, but this is entirely academic and for my entertainment. Uses Numpy for calculations, numba to accelerate some of the rasterization calculations, and Pygame to handle UI and keystrokes.

To change images, change the image file in line 418. Camera properties can be modified in line 430, and more intrinsic changes to behavior can be changed in the camera class. Images should be equirectangular projections. You can actually use this as a 360 image viewer if you comment out the hemisphere and back culling, and change the min_zoom in 430 to 0.

Currently on a laptop, 3 subdivions should create a decently round mesh @ 45-60fps on CPU (depends also on texture quality). Rasterization is the main time sink obviously, so a GPU implementation with proper buffers would fix.
