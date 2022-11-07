def get_id(filename):
    #SUBJECT_ENCOUNTER_DEVICE_CAPTURE_RESOLUTION_FGRP.EXT
    return filename.split('_')[0]

def get_fgrp(filename):
    filename = filename[:-4]
    return filename.split('_')[-1]
    #return filename.replace('.', '_').split('_')[-2]

def get_sensor(filename):
    return filename.split('_')[1]
