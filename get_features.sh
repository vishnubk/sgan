#!/usr/bin/env bash
module load singularity/latest
singularity exec -H /fred/oz002/vishnu:/home1 -B /fred/oz002/vishnu/:/fred/oz002/vishnu/ /fred/oz002/vishnu/sap4pulsars_pics_ai_dev10_new.simg python /fred/oz002/vishnu/sgan/extract_features_for_classification.py 
