# Findings — scene reconstruction on DGX Spark

Running log of results and gotchas. Newest first.

## Capture / camera

- **iPhone 15: use the 1× (main) camera, not 0.5× (ultra-wide).**
  - 1× footage reconstructs cleanly in COLMAP with both `OPENCV` and `OPENCV_FISHEYE`
    camera models — worked great.
  - 0.5× ultra-wide fails / behaves badly in COLMAP. Likely because Apple applies
    internal lens-distortion correction (dewarping) to the ultra-wide, so the frames
    don't match a clean parametric camera model COLMAP can solve for.

## Pipeline
_(none yet)_

## Gotchas log
_(record anything that bites during setup)_
