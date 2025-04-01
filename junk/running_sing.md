# instructions for running dev singularity container


>to build the container from `freesurfer_dev.def` file:

`singularity build freesurfer_dev.sif freesurfer_dev.def`

>to run and test this file

`singularity exec freesurfer_dev.sif recon-all --version`


